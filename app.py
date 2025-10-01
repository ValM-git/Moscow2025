from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import base64, io, secrets
import json, time, os
import cv2
import numpy as np
from tool_vision import _segment_components_from_bgr_kit, _build_expected_ref_map_kit, _segment_components_from_bgr, _build_expected_ref_map, _load_bgr_from_uploads, _compute_desc, _match_components_to_expected, _mask_from_roi, _suppress_nested_components

import cfg
from cfg import DEFAULT_WEIGHTS, KIT_WEIGHTS

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

DEBUG_DRAW = True
DEBUG_MATCH = True


app = Flask(__name__)

cfg.init_app(app)
BASE_DIR = cfg.BASE_DIR

db = SQLAlchemy(app)

# ===================== MODELS =====================

class ToolType(db.Model):
    __tablename__ = 'tool_type'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(160), nullable=False)
    photo_filename = db.Column(db.String(255), nullable=False)
    items = db.relationship('ToolItem', backref='type', cascade='all, delete-orphan')

    def photo_url(self):
        return url_for('uploaded_file', folder='types', filename=self.photo_filename) if self.photo_filename else None

    @property
    def total_count(self):
        return len(self.items)

    @property
    def available_count(self):
        return len([i for i in self.items if not i.is_blocked])

class ToolItem(db.Model):
    __tablename__ = 'tool_item'
    id = db.Column(db.Integer, primary_key=True)
    type_id = db.Column(db.Integer, db.ForeignKey('tool_type.id'), nullable=False)
    inv = db.Column(db.String(64), nullable=False, unique=True)
    photo_filename = db.Column(db.String(255), nullable=False)
    is_blocked = db.Column(db.Boolean, default=False)

    ref_photos = db.relationship('ToolItemRefPhoto', backref='tool_item', cascade='all, delete-orphan')

    def photo_url(self):
        return url_for('uploaded_file', folder='items', filename=self.photo_filename) if self.photo_filename else None

class Request(db.Model):
    __tablename__ = 'request'
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(64), nullable=False, unique=True)
    status = db.Column(db.String(16), default='open')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    closed_at = db.Column(db.DateTime, nullable=True)
    # связь с конкретными экземплярами
    items = db.relationship('RequestToolItem', backref='request', cascade='all, delete-orphan')
    photos = db.relationship('RequestPhoto', backref='request', cascade='all, delete-orphan')

class RequestToolItem(db.Model):
    __tablename__ = 'request_tool_item'
    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.Integer, db.ForeignKey('request.id'), nullable=False)
    item_id = db.Column(db.Integer, db.ForeignKey('tool_item.id'), nullable=False)
    item = db.relationship('ToolItem')

class RequestPhoto(db.Model):
    __tablename__ = 'request_photo'
    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.Integer, db.ForeignKey('request.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)

class ToolMovement(db.Model):
    __tablename__ = 'tool_movement'
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('tool_item.id'), nullable=False)
    request_id = db.Column(db.Integer, db.ForeignKey('request.id'), nullable=False)
    action = db.Column(db.String(32), nullable=False)
    ts = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    item = db.relationship('ToolItem')
    request = db.relationship('Request')

class ToolItemRefPhoto(db.Model):
    __tablename__ = 'tool_item_ref_photo'
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('tool_item.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)

    item = db.relationship('ToolItem')


# ===================== HELPERS =====================

def save_ref_photo(file):
    return save_uploaded(file, app.config['UPLOAD_FOLDER_ITEMS'])

def gen_code(prefix='REQ'):
    now = datetime.utcnow()
    base = now.strftime('%Y%m%d-%H%M%S')
    return f'{prefix}-{base}-{secrets.token_hex(3)}'

def save_uploaded(file, target_folder):
    filename = secure_filename(file.filename)
    name, ext = os.path.splitext(filename)
    unique = f"{name}-{secrets.token_hex(4)}{ext}"
    path = os.path.join(target_folder, unique)
    file.save(path)
    return unique

def seed_example_if_empty():
    if ToolType.query.count() > 0:
        return

def try_copy_samples(filenames):
    src_dir = os.path.join(BASE_DIR, 'static', 'samples')
    for fname in filenames:
        src = os.path.join(src_dir, fname)
        if not os.path.exists(src):
            continue
        if fname == filenames[0]:
            dst = os.path.join(app.config['UPLOAD_FOLDER_TYPES'], fname)
        else:
            dst = os.path.join(app.config['UPLOAD_FOLDER_ITEMS'], fname)
        if not os.path.exists(dst):
            with open(src, 'rb') as s, open(dst, 'wb') as d:
                d.write(s.read())

def log_movement(item_id: int, request_id: int, action: str):
    db.session.add(ToolMovement(item_id=item_id, request_id=request_id, action=action))

def _safe_imwrite_png(path: str, img: np.ndarray) -> bool:
    try:
        ok, buf = cv2.imencode('.png', img)
        if ok:
            buf.tofile(path)
            return True
    except Exception:
        pass
    try:
        return bool(cv2.imwrite(path, img))
    except Exception:
        return False

def _analyze_json_path(req_id: int) -> str:
    return os.path.join(app.config['UPLOAD_FOLDER_REQ'], f"__analyze_{req_id}.json")

def _save_analyze_result(req_id: int, payload) -> None:
    if isinstance(payload, dict):
        items = payload.get("items", [])
        thr = payload.get("threshold")
        try:
            thr = float(thr) if thr is not None else None
        except Exception:
            thr = None
        data = {
            "req_id": req_id,
            "ts": int(time.time()),
            "threshold": thr,
            "items": items,
        }
    elif isinstance(payload, list):
        data = {
            "req_id": req_id,
            "ts": int(time.time()),
            "threshold": None,
            "items": payload,
        }
    else:
        raise ValueError("Unsupported payload type for _save_analyze_result")

    with open(_analyze_json_path(req_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _load_analyze_result(req_id: int, req_created_at=None, expected_invs: list[str]|None=None) -> dict | None:
    p = _analyze_json_path(req_id)
    if not os.path.exists(p):
        return None

    try:
        if req_created_at is not None:
            if os.path.getmtime(p) < req_created_at.timestamp():
                return None
    except Exception:
        pass

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items")
        thr = data.get("threshold", None)
        if not isinstance(items, list):
            return None

        if expected_invs is not None:
            file_invs = {r.get("inv") for r in items if r.get("inv")}
            if file_invs and set(expected_invs) != file_invs:
                return None

        return {"threshold": thr, "items": items}
    except Exception:
        return None

def _is_kit_request(req: 'Request') -> bool:
    if len(req.items) != 1:
        return False
    tname = (req.items[0].item.type.name or '').strip().lower()
    return tname.startswith('набор')

def _annot_name(fn: str) -> str:
    name, _ = os.path.splitext(fn)
    return f"{name}__annot.png"

def _dummy_match_label(idx: int):
    return f"Segment {idx+1}", 0.99

# ===================== ROUTES =====================

@app.route('/')
def index():
    return redirect(url_for('types_list'))

@app.route('/uploads/<path:folder>/<path:filename>')
def uploaded_file(folder, filename):
    if folder == 'types':
        directory = app.config['UPLOAD_FOLDER_TYPES']
    elif folder == 'items':
        directory = app.config['UPLOAD_FOLDER_ITEMS']
    elif folder == 'requests':
        directory = app.config['UPLOAD_FOLDER_REQ']
    else:
        return '', 404
    return send_from_directory(directory, filename)

@app.route('/types', methods=['GET', 'POST'])
def types_list():
    if request.method == 'POST':
        # === СОЗДАНИЕ ТИПА + ЭКЗЕМПЛЯРЫ (файлы refs_i[] + кадры data_url_i[]) ===
        name = (request.form.get('name') or '').strip()
        t_photo = request.files.get('type_photo')

        if not name or not t_photo:
            flash('Укажите название и загрузите общее фото типа', 'error')
            return redirect(url_for('types_list'))

        inv_list = request.form.getlist('inv[]')
        if not inv_list:
            flash('Нужно добавить минимум один экземпляр', 'error')
            return redirect(url_for('types_list'))

        existing = ToolItem.query.filter(ToolItem.inv.in_(inv_list)).first()
        if existing:
            flash(f'Инвентарный номер «{existing.inv}» уже существует', 'error')
            return redirect(url_for('types_list'))

        try:
            type_fname = save_uploaded(t_photo, app.config['UPLOAD_FOLDER_TYPES'])
            t = ToolType(name=name, photo_filename=type_fname)
            db.session.add(t)
            db.session.flush()

            from werkzeug.datastructures import FileStorage
            import base64, io, secrets

            for i, inv in enumerate(inv_list):
                inv = (inv or '').strip()
                if not inv:
                    db.session.rollback()
                    flash('Инвентарный номер не может быть пустым', 'error')
                    return redirect(url_for('types_list'))

                files = [f for f in request.files.getlist(f'refs_{i}[]') if f and f.filename]

                data_urls = request.form.getlist(f'data_url_{i}[]') or []
                for du in data_urls:
                    if du and du.startswith('data:image/'):
                        header, b64 = du.split(',', 1)
                        img_bytes = base64.b64decode(b64)
                        fs = FileStorage(stream=io.BytesIO(img_bytes),
                                         filename=f'cam-{secrets.token_hex(4)}.png',
                                         content_type='image/png')
                        files.append(fs)

                if len(files) < 3:
                    db.session.rollback()
                    flash(f'Для инв. «{inv}» добавьте минимум 3 фото (файлами и/или с камеры).', 'error')
                    return redirect(url_for('types_list'))

                saved = []
                for f in files:
                    fn = save_ref_photo(f)
                    saved.append(fn)

                main_photo = saved[0]
                item = ToolItem(type_id=t.id, inv=inv, photo_filename=main_photo)
                db.session.add(item)
                db.session.flush()
                for fn in saved:
                    db.session.add(ToolItemRefPhoto(item_id=item.id, filename=fn))

            db.session.commit()
            flash('Тип создан.', 'ok')
            return redirect(url_for('types_list'))

        except Exception as e:
            db.session.rollback()
            flash(f'Ошибка при создании типа: {e}', 'error')
            return redirect(url_for('types_list'))

    q = (request.args.get('q') or '').strip()
    if q:
        types = (ToolType.query
                 .filter(ToolType.name.ilike(f'%{q}%'))
                 .order_by(ToolType.name.asc())
                 .all())
    else:
        types = ToolType.query.order_by(ToolType.name.asc()).all()

    return render_template('types.html', types=types, q=q)

@app.route('/types/<int:type_id>/edit', methods=['GET', 'POST'])
def type_edit(type_id):
    t = ToolType.query.get_or_404(type_id)

    if request.method == 'POST':
        action = request.form.get('action')

        has_items = len(t.items) > 0

        if action == 'add_item':
            inv = (request.form.get('inv_new') or '').strip()

            files = [f for f in request.files.getlist('refs_new[]') if f and f.filename]

            from werkzeug.datastructures import FileStorage
            import base64, io, secrets

            data_urls = request.form.getlist('data_url[]') or []
            for du in data_urls:
                if du and du.startswith('data:image/'):
                    try:
                        _, b64 = du.split(',', 1)
                        img_bytes = base64.b64decode(b64)
                    except Exception:
                        continue
                    files.append(FileStorage(
                        stream=io.BytesIO(img_bytes),
                        filename=f'cam-{secrets.token_hex(4)}.png',
                        content_type='image/png'
                    ))

            if not inv:
                flash('Введите инвентарный номер', 'error')
                return redirect(url_for('type_edit', type_id=type_id))
            if len(files) < 3:
                flash('Нужно минимум 3 фото (файлами и/или с камеры).', 'error')
                return redirect(url_for('type_edit', type_id=type_id))
            if ToolItem.query.filter_by(inv=inv).first():
                flash(f'Инвентарный номер «{inv}» уже существует', 'error')
                return redirect(url_for('type_edit', type_id=type_id))

            try:
                saved = [save_ref_photo(f) for f in files]
                item = ToolItem(type_id=type_id, inv=inv, photo_filename=saved[0])
                db.session.add(item)
                db.session.flush()
                for fn in saved:
                    db.session.add(ToolItemRefPhoto(item_id=item.id, filename=fn))
                db.session.commit()
                flash('Экземпляр добавлен.', 'ok')
            except Exception as e:
                db.session.rollback()
                flash(f'Ошибка при добавлении экземпляра: {e}', 'error')

            return redirect(url_for('type_edit', type_id=type_id))

        if action == 'update_items':
            proposed = {}
            dup_check = set()

            for item in t.items:
                if item.is_blocked:
                    continue
                new_inv = (request.form.get(f'inv_{item.id}') or '').strip()
                if not new_inv:
                    new_inv = item.inv
                proposed[item.id] = new_inv
                if new_inv in dup_check:
                    flash(f'Дубликат инв. номера в форме: {new_inv}', 'error')
                    return redirect(url_for('type_edit', type_id=type_id))
                dup_check.add(new_inv)

            for item in t.items:
                if item.is_blocked:
                    continue
                new_inv = proposed.get(item.id, item.inv)
                if new_inv != item.inv:
                    conflict = ToolItem.query.filter(ToolItem.inv == new_inv, ToolItem.id != item.id).first()
                    if conflict:
                        flash(f'Инв. номер {new_inv} уже используется другим экземпляром', 'error')
                        return redirect(url_for('type_edit', type_id=type_id))

            for item in t.items:
                if item.is_blocked:
                    continue
                item.inv = proposed[item.id]
                file = request.files.get(f'photo_{item.id}')
                if file and file.filename:
                    item.photo_filename = save_uploaded(file, app.config['UPLOAD_FOLDER_ITEMS'])

            db.session.commit()
            flash('Экземпляры обновлены', 'ok')
            return redirect(url_for('type_edit', type_id=type_id))

        if action == 'delete_item':
            item_id = int(request.form.get('item_id'))
            item = ToolItem.query.get_or_404(item_id)
            if item.is_blocked:
                flash('Нельзя удалить: экземпляр находится в открытой заявке', 'error')
                return redirect(url_for('type_edit', type_id=type_id))
            db.session.delete(item)
            db.session.commit()
            flash('Экземпляр удалён', 'ok')
            return redirect(url_for('type_edit', type_id=type_id))

        if action == 'delete_type':
            if has_items:
                flash('Нельзя удалить тип: сначала удалите все экземпляры этого типа.', 'error')
                return redirect(url_for('type_edit', type_id=type_id))
            db.session.delete(t)
            db.session.commit()
            flash('Тип удалён', 'ok')
            return redirect(url_for('types_list'))

        if action == 'update_main':
            item_id = int(request.form.get('item_id') or 0)
            item = ToolItem.query.get_or_404(item_id)
            if item.is_blocked:
                flash('Экземпляр заблокирован (в заявке) — заменять фото нельзя', 'error')
                return redirect(url_for('type_edit', type_id=type_id))
            f = request.files.get('photo')
            if not f or not f.filename:
                flash('Файл не выбран', 'error')
                return redirect(url_for('type_edit', type_id=type_id))
            fname = save_ref_photo(f)  # сохраняем в ту же папку /uploads/items
            item.photo_filename = fname
            db.session.commit()
            flash('Основное фото обновлено', 'ok')
            return redirect(url_for('type_edit', type_id=type_id))

    return render_template('edit_type.html', t=t)

@app.route('/items/<int:item_id>/refs/add', methods=['POST'])
def add_item_refs(item_id):
    item = ToolItem.query.get_or_404(item_id)
    if item.is_blocked:
        flash('Экземпляр заблокирован (в заявке) — добавление эталонов запрещено', 'error')
        return redirect(url_for('type_edit', type_id=item.type_id))

    files = [f for f in request.files.getlist('refs[]') if f and f.filename]

    data_urls = request.form.getlist('data_url[]') or []
    for du in data_urls:
        if du and du.startswith('data:image/'):
            try:
                _, b64 = du.split(',', 1)
                img_bytes = base64.b64decode(b64)
            except Exception:
                continue
            files.append(FileStorage(
                stream=io.BytesIO(img_bytes),
                filename=f'cam-{secrets.token_hex(4)}.png',
                content_type='image/png'
            ))

    if not files:
        flash('Добавьте хотя бы одно фото (файлом и/или с камеры).', 'error')
        return redirect(url_for('type_edit', type_id=item.type_id))

    try:
        added = 0
        for f in files:
            fname = save_ref_photo(f)
            db.session.add(ToolItemRefPhoto(item_id=item.id, filename=fname))
            added += 1
        db.session.commit()
        flash(f'Добавлено эталонных фото: {added}', 'ok')
    except Exception as e:
        db.session.rollback()
        flash(f'Ошибка при добавлении эталонов: {e}', 'error')

    return redirect(url_for('type_edit', type_id=item.type_id))

@app.route('/items/<int:item_id>/refs/<int:ref_id>/delete', methods=['POST'])
def delete_item_ref(item_id, ref_id):
    item = ToolItem.query.get_or_404(item_id)
    ref = ToolItemRefPhoto.query.get_or_404(ref_id)
    if ref.item_id != item.id:
        return '', 404
    if item.is_blocked:
        flash('Экземпляр заблокирован (в заявке) — удаление эталонов запрещено', 'error')
        return redirect(url_for('type_edit', type_id=item.type_id))

    db.session.delete(ref)
    db.session.commit()
    flash('Эталонное фото удалено', 'ok')
    return redirect(url_for('type_edit', type_id=item.type_id))

@app.route('/requests/new', methods=['GET', 'POST'])
def new_request():
    types = ToolType.query.order_by(ToolType.id.asc()).all()
    if request.method == 'POST':
        picks = []
        for t in types:
            qty_str = request.form.get(f'type_{t.id}')
            if not qty_str:
                continue
            try:
                qty = int(qty_str)
            except:
                qty = 0
            if qty <= 0:
                continue
            if qty > t.available_count:
                flash(f'Запрошено {qty}, но доступно только {t.available_count} для типа «{t.name}»', 'error')
                return redirect(url_for('new_request'))
            picks.append((t, qty))
        if not picks:
            flash('Выберите хотя бы один тип и количество', 'error')
            return redirect(url_for('new_request'))

        req = Request(code=gen_code('REQ'), status='open')
        db.session.add(req)
        db.session.flush()
        try:
            stale = _analyze_json_path(req.id)
            if os.path.exists(stale):
                os.remove(stale)
        except Exception:
            pass

        for t, qty in picks:
            free_items = [i for i in t.items if not i.is_blocked]
            chosen = free_items[:qty]
            for item in chosen:
                item.is_blocked = True
                link = RequestToolItem(request_id=req.id, item_id=item.id)
                db.session.add(link)
                log_movement(item_id=item.id, request_id=req.id, action='reserved')

        db.session.commit()
        flash(f'Заявка {req.code} создана и экземпляры зарезервированы', 'ok')
        return redirect(url_for('requests_list'))

    return render_template('new_request.html', types=types)

@app.route('/requests')
def requests_list():
    reqs = Request.query.order_by(Request.created_at.desc()).all()

    missing_by_req = {}
    for r in reqs:
        missing = set()
        if r.status in ('closed_missing', 'closed'):
            analyze = _load_analyze_result(r.id) or {}
            for it in analyze.get('items', []):
                if it.get('user_status') == 'absent':
                    inv = it.get('inv')
                    if inv:
                        missing.add(inv)
        missing_by_req[r.id] = missing

    return render_template('requests.html', reqs=reqs, missing_by_req=missing_by_req)

@app.route('/requests/<int:req_id>/close', methods=['GET', 'POST'])
def close_request(req_id):
    req = Request.query.get_or_404(req_id)
    if req.status == 'closed':
        flash('Заявка уже закрыта', 'ok')
        return redirect(url_for('requests_list'))

    if request.method == 'POST':
        files = request.files.getlist('photos')
        added = 0
        for f in files:
            if f and f.filename:
                fname = save_uploaded(f, app.config['UPLOAD_FOLDER_REQ'])
                db.session.add(RequestPhoto(request_id=req.id, filename=fname))
                added += 1
        if added:
            db.session.commit()
            flash(f'Фото добавлены: {added}', 'ok')
            return redirect(url_for('close_request', req_id=req.id))

        if request.form.get('action') == 'confirm':
            req.status = 'closed'
            req.closed_at = datetime.utcnow()
            for link in req.items:
                link.item.is_blocked = False
                log_movement(item_id=link.item.id, request_id=req.id, action='returned')
            db.session.commit()
            flash(f'Заявка {req.code} закрыта.', 'ok')
            return redirect(url_for('requests_list'))

        return redirect(url_for('close_request', req_id=req.id))

    expected_invs = [link.item.inv for link in req.items]
    analyze = _load_analyze_result(
        req.id,
        req_created_at=req.created_at,
        expected_invs=expected_invs
    ) or {}

    thr_val = analyze.get("threshold") if isinstance(analyze, dict) else None
    try:
        threshold_default = float(thr_val) if thr_val is not None else 0.50
    except Exception:
        threshold_default = 0.50

    return render_template(
        "close_request.html",
        req=req,
        analyze=analyze,
        threshold_default=threshold_default
    )

@app.route('/items/<int:item_id>/history')
def item_history(item_id):
    item = ToolItem.query.get_or_404(item_id)
    moves = ToolMovement.query.filter_by(item_id=item_id).order_by(ToolMovement.ts.desc()).all()
    return render_template('item_history.html', item=item, moves=moves)


@app.route('/requests/<int:req_id>/check', methods=['POST'])
def check_request(req_id):
    req = Request.query.get_or_404(req_id)
    file = request.files.get('photo')
    if not file or file.filename == '':
        data_url = request.form.get('data_url')
        if data_url and data_url.startswith('data:image/'):
            import base64, io
            header, b64 = data_url.split(',', 1)
            img_bytes = base64.b64decode(b64)
            from werkzeug.datastructures import FileStorage
            file = FileStorage(stream=io.BytesIO(img_bytes), filename=f'camera-{secrets.token_hex(4)}.png', content_type='image/png')
        else:
            return jsonify({'ok': False, 'msg': 'Нет изображения'}), 400

    fname = save_uploaded(file, app.config['UPLOAD_FOLDER_REQ'])
    db.session.add(RequestPhoto(request_id=req.id, filename=fname))
    db.session.commit()

    expected = len(req.items)
    summary = f'Проверка-заглушка: ожидается {expected} инструмент(а). Фото сохранено: {fname}'
    flash(summary, 'ok')

    return redirect(url_for('close_request', req_id=req.id))


@app.route('/requests/<int:req_id>/analyze', methods=['POST'])
def analyze_request(req_id):
    req = Request.query.get_or_404(req_id)

    thr_raw = (request.form.get('threshold') or '').strip()
    try:
        threshold = float(thr_raw)
    except Exception:
        threshold = 0.50
    threshold = max(0.0, min(1.0, threshold))

    is_kit = _is_kit_request(req)
    weights = KIT_WEIGHTS if is_kit else DEFAULT_WEIGHTS

    expected_map = _build_expected_ref_map_kit(req) if is_kit else _build_expected_ref_map(req)
    if not expected_map:
        flash('Нет эталонных фото у ожидаемых инструментов — разметка невозможна.', 'error')
        return redirect(url_for('close_request', req_id=req.id))

    photos = list(req.photos)
    if not photos:
        flash('Фото для проверки не добавлены', 'error')
        return redirect(url_for('close_request', req_id=req.id))

    raw_components = []
    for ph in photos:
        bgr = _load_bgr_from_uploads('requests', ph.filename)
        if bgr is None:
            continue
        segs = _segment_components_from_bgr_kit(bgr) if is_kit else _segment_components_from_bgr(bgr)
        for seg in segs:
            desc = _compute_desc(bgr, seg['mask'])
            if not desc:
                continue
            raw_components.append({
                'bgr': bgr,
                'mask': seg['mask'],
                'desc': desc,
                'bbox': seg['bbox'],
                'photo_id': ph.id,
            })

    if is_kit:
        components = raw_components[:]
    else:
        components = []
        by_photo_raw = {}
        for comp in raw_components:
            by_photo_raw.setdefault(comp['photo_id'], []).append(comp)
        for pid, lst in by_photo_raw.items():
            filtered = _suppress_nested_components(lst, iou_thr=0.6, inside_thr=0.85, area_min_frac=0.03)
            components.extend(filtered)

    if not components:
        flash('На добавленных фото не удалось выделить объекты', 'error')
        return redirect(url_for('close_request', req_id=req.id))

    assignments, missing_invs, extras_idx = _match_components_to_expected(components, expected_map, threshold, weights=weights, allow_margin=(not is_kit))
    comp_score_display = {}
    for a in assignments:
        s_raw = float(a['score'])
        comp_idx = a['component_index']
        if (not is_kit) and s_raw < threshold:
            s_show = threshold
        else:
            s_show = s_raw
        comp_score_display[comp_idx] = s_show

    by_photo = {}
    for i, comp in enumerate(components):
        pid = comp['photo_id']
        if pid not in by_photo:
            src = _load_bgr_from_uploads('requests', [p for p in photos if p.id == pid][0].filename)
            if src is None:
                continue
            by_photo[pid] = [src.copy(), []]
        by_photo[pid][1].append(i)

    matched_for_comp = {a['component_index']: a for a in assignments}

    green = (16, 163, 74)
    red   = (220, 38, 38)
    added = 0

    for ph in photos:
        if ph.id not in by_photo:
            continue
        canvas, comp_ids = by_photo[ph.id]

        for i in comp_ids:
            if i not in matched_for_comp:
                continue

            comp = components[i]
            x, y, w, h = comp.get('bbox', (0, 0, 0, 0))
            if w <= 0 or h <= 0:
                cnt = comp['desc'].get('cnt')
                if cnt is None or len(cnt) < 3:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)

            if i in matched_for_comp:
                inv = matched_for_comp[i]['inv']
                score = comp_score_display.get(i, matched_for_comp[i]['score'])
                color = green
                label = f"{inv}  {score:.2f}"

            roi = comp['bgr'][y:y + h, x:x + w]
            m = _mask_from_roi(roi)
            rcnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if rcnts:
                c_local = max(rcnts, key=cv2.contourArea)
                c_global = c_local + np.array([[x, y]])
                cv2.drawContours(canvas, [c_global], -1, color, 2)

            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
            cv2.putText(canvas, label, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (17, 17, 17), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        out_name = _annot_name(ph.filename)
        out_path = os.path.join(app.config['UPLOAD_FOLDER_REQ'], out_name)
        _safe_imwrite_png(out_path, canvas)
        added += 1

    if missing_invs:
        miss = ', '.join(missing_invs)
        flash(f'Анализ выполнен. Не найдены: {miss}', 'warn')
    else:
        flash(f'Анализ выполнен. Все ожидаемые экземпляры найдены.', 'ok')
    annot_by_photo_id = {}
    for ph in photos:
        annot_by_photo_id[ph.id] = _annot_name(ph.filename)

    best_for_inv = {}
    for a in assignments:
        inv = a['inv']
        comp_idx = a['component_index']
        s = float(a['score'])
        if inv not in best_for_inv or s > best_for_inv[inv][1]:
            best_for_inv[inv] = (comp_idx, s)

    per_inv = []
    for link in req.items:
        inv = link.item.inv
        if inv in best_for_inv:
            ci, s = best_for_inv[inv]
            comp = components[ci]
            pid = comp['photo_id']
            x, y, w, h = comp.get('bbox', (0, 0, 0, 0))
            s_show = comp_score_display.get(ci, s)

            per_inv.append({
                "inv": inv,
                "found": True,
                "score": round(float(s_show), 3),
                "photo_annot": annot_by_photo_id.get(pid),
                "photo_id": int(pid),
                "bbox": [int(x), int(y), int(w), int(h)],
            })
        else:
            per_inv.append({
                "inv": inv,
                "found": False,
                "score": None,
                "photo_annot": None,
                "photo_id": None,
                "bbox": None,
            })

    _save_analyze_result(req.id, {
        "threshold": threshold,
        "items": per_inv,
    })

    return redirect(url_for('close_request', req_id=req.id))


@app.route('/requests/<int:req_id>/photos/add', methods=['POST'])
def add_request_photos(req_id):
    req = Request.query.get_or_404(req_id)

    hdr = (request.headers.get('X-Requested-With') or '').lower()
    accept = (request.headers.get('Accept') or '').lower()
    is_fetch = ('fetch' in hdr) or ('xmlhttprequest' in hdr) or ('application/json' in accept)

    mode = (request.form.get('mode') or '').lower()
    if mode == 'replace' and req.photos:
        for ph in list(req.photos):
            db.session.delete(ph)
        db.session.flush()

    files = [f for f in request.files.getlist('photos[]') if f and f.filename]

    data_urls = request.form.getlist('data_url[]') or []
    for du in data_urls:
        if du and du.startswith('data:image/'):
            try:
                header, b64 = du.split(',', 1)
                img_bytes = base64.b64decode(b64)
            except Exception:
                continue
            files.append(FileStorage(
                stream=io.BytesIO(img_bytes),
                filename=f'cam-{secrets.token_hex(4)}.png',
                content_type='image/png'
            ))

    added = 0
    for f in files:
        fname = save_uploaded(f, app.config['UPLOAD_FOLDER_REQ'])
        db.session.add(RequestPhoto(request_id=req.id, filename=fname))
        added += 1
    db.session.commit()

    if is_fetch:
        return jsonify({'ok': True, 'added': added})

    flash(f'Добавлено фото: {added}' if added else 'Фото не добавлены',
          'ok' if added else 'error')
    return redirect(url_for('close_request', req_id=req.id))


@app.route('/requests/<int:req_id>/photos/<int:photo_id>/delete', methods=['POST'])
def delete_request_photo(req_id, photo_id):
    ph = RequestPhoto.query.get_or_404(photo_id)
    if ph.request_id != req_id:
        abort(404)

    try:
        path = os.path.join(app.config['UPLOAD_FOLDER_REQ'], ph.filename)
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

    try:
        apath = os.path.join(app.config['UPLOAD_FOLDER_REQ'], _annot_name(ph.filename))
        if os.path.exists(apath):
            os.remove(apath)
    except Exception:
        pass

    db.session.delete(ph)
    db.session.commit()
    flash('Фото удалено', 'ok')
    return redirect(url_for('close_request', req_id=req_id))


@app.route('/requests/<int:req_id>/decision', methods=['POST'])
def set_item_decision(req_id):
    req = Request.query.get_or_404(req_id)
    data = _load_analyze_result(req_id) or {"threshold": None, "items": []}

    expected_invs = [link.item.inv for link in req.items]

    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"ok": False, "msg": "bad json"}), 400

    inv = (payload or {}).get('inv')
    action = (payload or {}).get('action')  # 'manual_found' | 'absent' | 'clear'
    if inv not in expected_invs:
        return jsonify({"ok": False, "msg": "unknown inv"}), 400
    if action not in ('manual_found', 'absent', 'clear'):
        return jsonify({"ok": False, "msg": "bad action"}), 400

    items = data.get("items", [])
    rec = next((r for r in items if r.get("inv") == inv), None)
    if rec is None:
        rec = {"inv": inv, "found": False, "score": None, "photo_annot": None, "photo_id": None, "bbox": None}
        items.append(rec)

    rec["user_status"] = None if action == 'clear' else action

    _save_analyze_result(req.id, {"threshold": data.get("threshold"), "items": items})

    def is_resolved(r):
        return bool(r.get("found")) or (r.get("user_status") in ("manual_found", "absent"))

    by_inv = {r.get("inv"): r for r in items}
    resolved_all = all(is_resolved(by_inv.get(x, {})) for x in expected_invs)

    return jsonify({"ok": True, "resolved_all": resolved_all})

@app.route('/requests/<int:req_id>/finalize', methods=['POST'])
def finalize_request(req_id):
    req = Request.query.get_or_404(req_id)
    analyze = _load_analyze_result(req.id) or {"items": []}
    by_inv = {r.get("inv"): r for r in analyze.get("items", [])}

    expected_invs = [link.item.inv for link in req.items]

    def resolve_state(rec: dict) -> str:
        us = rec.get("user_status")
        if us == "absent":
            return "absent"
        if us == "manual_found":
            return "found"
        if bool(rec.get("found")):
            return "found"
        return "unresolved"

    unresolved = [inv for inv in expected_invs if resolve_state(by_inv.get(inv, {})) == "unresolved"]
    if unresolved:
        flash("Есть позиции без решения (ни найдено, ни отсутствует): " + ", ".join(unresolved), "error")
        return redirect(url_for('close_request', req_id=req.id))

    missing = []
    for link in req.items:
        inv = link.item.inv
        state = resolve_state(by_inv.get(inv, {}))
        if state == "found":
            link.item.is_blocked = False
            log_movement(item_id=link.item.id, request_id=req.id, action='returned')
        elif state == "absent":
            link.item.is_blocked = True
            missing.append(inv)
            log_movement(item_id=link.item.id, request_id=req.id, action='missing')

    req.closed_at = datetime.utcnow()
    req.status = 'closed' if not missing else 'closed_missing'
    db.session.commit()

    if missing:
        from markupsafe import Markup
        abs_list = ", ".join(missing)
        flash(Markup(
            "Заявка <b>закрыта</b> с недостающими позициями: "
            f"<b style='color:#b91c1c'>{abs_list}</b>"
        ), "error")
    else:
        flash("Заявка закрыта. Все позиции сданы.", "ok")

    return redirect(url_for('requests_list'))

@app.template_filter('annot_name')
def j_annot_name(fn: str) -> str:
    return _annot_name(fn)

@app.template_filter('req_exists')
def j_req_exists(fn: str) -> bool:
    path = os.path.join(app.config['UPLOAD_FOLDER_REQ'], fn)
    return os.path.exists(path)

# ===================== INIT =====================

@app.before_request
def init_db():
    db.create_all()
    seed_example_if_empty()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)