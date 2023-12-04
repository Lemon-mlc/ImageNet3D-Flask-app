from datetime import datetime
import glob
import logging
import os
import shutil
import sqlite3
import uuid

from flask import Flask, render_template, request, url_for, jsonify, redirect, g
import flask_login
import numpy as np

from config import DATABASE, DATA_PATH, CAD_PATH, NUM_MODELS, SAVE_PATH, step_size
from render import matplotlib_line_render

save_path = SAVE_PATH
os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
os.makedirs('static/tmp', exist_ok=True)

dt = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=os.path.join(save_path, "logs", f"log_{dt}.txt"),
    filemode="w",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

NEW_DATABASE = f'database_{dt}.sqlite'
shutil.copyfile(DATABASE, NEW_DATABASE)

app = Flask(__name__, static_url_path='')
app.secret_key = 'there is no secret'

login_manager = flask_login.LoginManager()
login_manager.init_app(app)


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = sqlite3.connect(NEW_DATABASE)
        db.row_factory = sqlite3.Row
        g._database = db
    return db


def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return rv


def call_db(cmd):
    conn = get_db()
    c = conn.cursor()
    c.execute(cmd)
    conn.commit()


def add_annotator(annotator_id):
    conn = get_db()
    c = conn.cursor()
    c.execute(f'''insert into annotators (annotator_id, pass_test, assigned_annotations, total_annotations) VALUES ("{annotator_id}", 0, 100, 0)''')
    conn.commit()


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


class User(flask_login.UserMixin):
    pass


def get_annotator(annotator_id):
    return query_db(f'select * from annotators where annotator_id = "{annotator_id}"')


def get_annotator_tasks(annotator_id):
    return query_db(f'select * from tasks where annotator_id = "{annotator_id}"') + query_db(f'select * from quality_tasks where annotator_id = "{annotator_id}"')


def get_task_questions_eval(task_id):
    return query_db(f'select * from questions_eval where task_id = "{task_id}"')


def get_task_questions(task_id, quality=False):
    if quality:
        return query_db(f'select * from quality_questions where task_id = "{task_id}"')
    else:
        return query_db(f'select * from questions where task_id = "{task_id}"')


def increment_task_progress(task_id, quality=False):
    if quality:
        raise NotImplementedError
    else:
        task = query_db(f'select * from tasks where task_id = "{task_id}"')[0]
        curr_progress = task['num_finished']
        conn = get_db()
        c = conn.cursor()
        c.execute(f'''update tasks set num_finished = {curr_progress+1} where task_id = "{task_id}"''')
        conn.commit()


def decrement_task_progress(task_id, quality=False):
    if quality:
        raise NotImplementedError
    else:
        task = query_db(f'select * from tasks where task_id = "{task_id}"')[0]
        curr_progress = task['num_finished']
        conn = get_db()
        c = conn.cursor()
        c.execute(f'''update tasks set num_finished = {curr_progress-1} where task_id = "{task_id}"''')
        conn.commit()


def increment_user_progress(annotator_id):
    annotator = query_db(f'select * from annotators where annotator_id = "{annotator_id}"')[0]
    curr_progress = annotator['finished_annotations']
    conn = get_db()
    c = conn.cursor()
    c.execute(f'''update annotators set finished_annotations = {curr_progress+1} where annotator_id = "{annotator_id}"''')
    conn.commit()


def decrement_user_progress(annotator_id):
    annotator = query_db(f'select * from annotators where annotator_id = "{annotator_id}"')[0]
    curr_progress = annotator['finished_annotations']
    conn = get_db()
    c = conn.cursor()
    c.execute(f'''update annotators set finished_annotations = {curr_progress-1} where annotator_id = "{annotator_id}"''')
    conn.commit()


def set_question_finished_eval(q_id, finished):
    assert finished in [0, 1], f'finished = {finished}'
    conn = get_db()
    c = conn.cursor()
    c.execute(f'''update questions_eval set finished = {finished} where question_id = "{q_id}"''')
    conn.commit()


def set_question_finished(q_id, finished):
    assert finished in [0, 1], f'finished = {finished}'
    conn = get_db()
    c = conn.cursor()
    c.execute(f'''update questions set finished = {finished} where question_id = "{q_id}"''')
    conn.commit()


def set_question_rejected(q_id, rejected):
    assert rejected in [0, 1], f'rejected = {rejected}'
    conn = get_db()
    c = conn.cursor()
    c.execute(f'''update questions set rejected = {rejected} where question_id = "{q_id}"''')
    conn.commit()


@login_manager.user_loader
def user_loader(annotator_id):
    annotator = query_db(f'select * from annotators where annotator_id = "{annotator_id}"')
    if len(annotator) == 0:
        return None
    user = User()
    user.id = annotator[0]['annotator_id']
    return user


@login_manager.request_loader
def request_loader(request):
    annotator = get_annotator(request.form.get('annotator_id'))
    if len(annotator) == 0:
        return None
    user = User()
    user.id = annotator[0]['annotator_id']
    return user


@app.route('/', methods=["GET", "POST"])
def login(warning=''):
    if request.method == 'GET':
        if flask_login.current_user.is_authenticated:
            return redirect(url_for('account'))
        else:
            return render_template('login.html', warning=warning)

    annotator_id = request.form['annotator_id']
    annotator = get_annotator(annotator_id)
    if len(annotator) == 0:
        return render_template('login.html', warning=annotator_id)

    user = User()
    user.id = annotator_id
    flask_login.login_user(user)
    return redirect(url_for('account'))


@app.route('/account')
@flask_login.login_required
def account():
    annotator_id = flask_login.current_user.id
    annotator = get_annotator(annotator_id)
    assert len(annotator) == 1
    pass_test = annotator[0]['pass_test']
    total_anno = annotator[0]['assigned_annotations']
    finished_anno = annotator[0]['finished_annotations']

    tasks = get_annotator_tasks(annotator_id)

    url, task_total_anno, task_finished_anno, task_progress, task_name = [], [], [], [], []
    for t in tasks:
        if 'quality' in t['task_id']:
            url.append(url_for('quality', task=t['task_id']))
        elif 'eval' in t['task_id']:
            url.append(url_for('eval', task=t['task_id']))
        else:
            url.append(url_for('annotate', task=t['task_id']))
        task_total_anno.append(t['num_questions'])
        task_finished_anno.append(t['num_finished'])
        task_progress.append(f"{t['num_finished']/t['num_questions']*100:.2f}%")
        task_name.append(t['task_id'])

    return render_template(
        'account.html',
        annotator_id=annotator_id,
        pass_test=pass_test,
        total_anno=total_anno,
        finished_anno=finished_anno,
        progress=f'{finished_anno/total_anno*100:.2f}%' if total_anno > 0 else '100%',
        tasks=zip(url, task_total_anno, task_finished_anno, task_progress, task_name))


@app.route('/test', methods=["GET", "POST"])
@flask_login.login_required
def test():
    if request.method == 'GET':
        return render_template('test.html', annotator_id=flask_login.current_user.id)

    annotator_id = flask_login.current_user.id
    call_db(f'update annotators set pass_test = 1 where annotator_id = {annotator_id}')

    return redirect(url_for('account'))


@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('unauthorized.html'), 401


def get_prev_question_eval(task, idx):
    questions = get_task_questions_eval(task)
    if idx == 0:
        return None, False, -1
    else:
        return questions[idx-1], True, idx-1


def get_prev_question(task, idx, quality=False):
    questions = get_task_questions(task, quality)
    if idx == 0:
        return None, False, -1
    else:
        return questions[idx-1], True, idx-1


def get_next_question_eval(task, idx):
    questions = get_task_questions_eval(task)
    if idx+1 >= len(questions):
        return None, False, -1
    else:
        return questions[idx+1], True, idx+1


def get_next_question(task, idx, quality=False):
    questions = get_task_questions(task, quality)
    if idx+1 >= len(questions):
        return None, False, -1
    else:
        return questions[idx+1], True, idx+1


def get_next_unannotated_question_eval(task, idx):
    questions = get_task_questions_eval(task)

    for i in range(0, len(questions)):
        curr_idx = (idx + i) % len(questions)
        if questions[curr_idx]['finished']:
            continue
        else:
            return questions[curr_idx], True, curr_idx

    return None, False, -1


def get_next_unannotated_question(task, idx, quality=False):
    questions = get_task_questions(task, quality)

    for i in range(0, len(questions)):
        curr_idx = (idx + i) % len(questions)
        if questions[curr_idx]['finished']:
            continue
        else:
            return questions[curr_idx], True, curr_idx

    return None, False, -1


def get_question_eval(task, idx):
    questions = get_task_questions_eval(task)
    if idx < 0:
        return questions[0], 0
    elif idx > len(questions):
        return questions[-1], len(questions) - 1
    else:
        return questions[idx], idx


def get_question(task, idx, quality=False):
    questions = get_task_questions(task, quality)
    if idx < 0:
        return questions[0], 0
    elif idx > len(questions):
        return questions[-1], len(questions) - 1
    else:
        return questions[idx], idx


def get_model_id(cad_path):
    c = os.path.basename(os.path.dirname(cad_path))
    m = os.path.basename(cad_path)
    return f'{c}_{m[:-4]}'


@app.route('/save_eval', methods=["POST"])
@flask_login.login_required
def save_eval():
    task = request.args.get('task', type=str)
    q_idx = request.args.get('q_idx', type=int)
    question, q_idx = get_question_eval(task, q_idx)

    pred = request.args.get('pred', type=str)
    annotator = request.args.get('annotator', default='none', type=str)

    if pred == 'undefined' or pred not in ['eval_good_1', 'eval_good_2']:
        return jsonify({'success': 0})

    logging.info(f'SystemMsg: Annotator [{annotator}] save annotation for question [{question["question_id"]}] from task [{task}]')

    eval_path = os.path.join(DATA_PATH, 'eval', question['anno_path'].replace('npz', 'txt').replace('annotations/', ''))
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)
    with open(eval_path, 'w') as fp:
        fp.write('0' if pred == 'eval_good_2' else '1')
    finished = question['finished']

    if not finished:
        set_question_finished_eval(question['question_id'], 1)
        increment_task_progress(task)
        increment_user_progress(annotator)

    return jsonify({'success': 1})


@app.route('/save', methods=["POST"])
@flask_login.login_required
def save():
    task = request.args.get('task', type=str)
    q_idx = request.args.get('q_idx', type=int)
    question, q_idx = get_question(task, q_idx)

    azim = request.args.get('azim', default=0.0, type=float) / 180.0 * np.pi
    elev = request.args.get('elev', default=0.0, type=float) / 180.0 * np.pi
    theta = request.args.get('theta', default=0.0, type=float) / 180.0 * np.pi
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    viewport = request.args.get('viewport', type=float)
    object_status = request.args.get('object_status', type=str)
    dense = request.args.get('dense', type=str)

    model_id = int(model_id.split('_')[-1])

    if object_status == 'undefined' or dense == 'undefined':
        return jsonify({'success': 0})

    logging.info(f'SystemMsg: Annotator [{annotator}] save annotation for question [{question["question_id"]}] from task [{task}]')

    img_path = os.path.join(DATA_PATH, question['img_path'])
    anno_path = os.path.join(DATA_PATH, question['anno_path'])
    initialized = question['initialized']
    finished = question['finished']

    if finished:
        npy = dict(np.load(anno_path.replace('annotations', 'annotations_new'), allow_pickle=True))
    else:
        npy = dict(np.load(anno_path, allow_pickle=True))
    npy['azimuth'] = azim
    npy['elevation'] = elev
    npy['theta'] = theta
    npy['distance'] = dist
    npy['px'] = px
    npy['py'] = py
    npy['cad_index'] = model_id
    npy['viewport'] = viewport
    npy['annotator'] = annotator
    npy['object_status'] = object_status
    npy['dense'] = dense
    os.makedirs(os.path.dirname(anno_path.replace('annotations', 'annotations_new')), exist_ok=True)
    np.savez(anno_path.replace('annotations', 'annotations_new'), **npy)

    if not finished:
        set_question_finished(question['question_id'], 1)
        increment_task_progress(task)
        increment_user_progress(annotator)

    return jsonify({'success': 1})


@app.route('/action_clear', methods=["POST"])
@flask_login.login_required
def action_clear():
    task = request.args.get('task', type=str)
    q_idx = request.args.get('q_idx', type=int)
    question, q_idx = get_question(task, q_idx)

    logging.info(f'SystemMsg: Annotator [{flask_login.current_user.id}] clear annotation for question [{question["question_id"]}] from task [{task}]')

    img_path = os.path.join(DATA_PATH, question['img_path'])
    anno_path = os.path.join(DATA_PATH, question['anno_path'])
    initialized = question['initialized']

    npy = dict(np.load(anno_path, allow_pickle=True))
    pose = {
        'azim': round(float(npy['azimuth'])/np.pi*180.0, 2),
        'elev': round(float(npy['elevation'])/np.pi*180.0, 2),
        'theta': round(float(npy['theta'])/np.pi*180.0, 2),
        'dist': round(float(npy['distance']), 2),
        'px': round(float(npy['px']), 2),
        'py': round(float(npy['py']), 2)}
    model_idx = npy['cad_index']
    viewport = npy['viewport']

    if question['finished']:
        os.remove(anno_path.replace('annotations', 'annotations_new'))
        set_question_finished(question['question_id'], 0)
        decrement_task_progress(task)
        decrement_user_progress(flask_login.current_user.id)

    cad_path = os.path.join(CAD_PATH, question['cad_path'], f'{model_idx:02d}.off')

    rendered_img = _render(
        img_path, cad_path, annotator=flask_login.current_user.id, viewport=viewport,
        **pose)
    pose_img = _render_pose(
        cad_path, annotator=flask_login.current_user.id, viewport=viewport,
        **pose)

    return jsonify({
        'rendered_img': rendered_img,
        'pose_img': pose_img,
        'status_initialized': initialized,
        'status_finished': 0,
        'status_rejected': 0,
        'model_id': get_model_id(cad_path),
        **pose
    })


@app.route('/eval_next')
@flask_login.login_required
def eval_next():
    task = request.args.get('task', type=str)
    q_idx = request.args.get('q_idx', type=int)
    question, found_next, q_idx = get_next_question_eval(task, q_idx)
    if not found_next:
        return redirect(url_for('account'))
    else:
        return redirect(url_for('eval', task=task, q_idx=q_idx))


@app.route('/annotated_next')
@flask_login.login_required
def annotate_next():
    task = request.args.get('task', type=str)
    q_idx = request.args.get('q_idx', type=int)
    question, found_next, q_idx = get_next_question(task, q_idx)
    if not found_next:
        return redirect(url_for('account'))
    else:
        return redirect(url_for('annotate', task=task, q_idx=q_idx))


@app.route('/eval_prev')
@flask_login.login_required
def eval_prev():
    task = request.args.get('task', type=str)
    q_idx = request.args.get('q_idx', type=int)
    question, found_next, q_idx = get_prev_question_eval(task, q_idx)
    if not found_next:
        return redirect(url_for('account'))
    else:
        return redirect(url_for('eval', task=task, q_idx=q_idx))


@app.route('/annotate_prev')
@flask_login.login_required
def annotate_prev():
    task = request.args.get('task', type=str)
    q_idx = request.args.get('q_idx', type=int)
    question, found_next, q_idx = get_prev_question(task, q_idx)
    if not found_next:
        return redirect(url_for('account'))
    else:
        return redirect(url_for('annotate', task=task, q_idx=q_idx))


@app.route('/eval_next_unfinished')
@flask_login.login_required
def eval_next_unfinished():
    task = request.args.get('task', type=str)
    q_idx = request.args.get('q_idx', type=int)
    question, found_next, q_idx = get_next_unannotated_question_eval(task, q_idx+1)
    if not found_next:
        return redirect(url_for('account'))
    else:
        return redirect(url_for('eval', task=task, q_idx=q_idx))


@app.route('/annotate_next_unfinished')
@flask_login.login_required
def annotate_next_unfinished():
    task = request.args.get('task', type=str)
    q_idx = request.args.get('q_idx', type=int)
    question, found_next, q_idx = get_next_unannotated_question(task, q_idx+1)
    if not found_next:
        return redirect(url_for('account'))
    else:
        return redirect(url_for('annotate', task=task, q_idx=q_idx))


@app.route('/quality')
@flask_login.login_required
def quality():
    task = request.args.get('task', type=str)
    q_idx = request.args.get('q_idx', default=-1, type=int)

    q_idx = 0
    question, q_idx = get_question(task, q_idx, quality=True)

    img_path = os.path.join(DATA_PATH, question['img_path'])
    anno_path = os.path.join(DATA_PATH, question['anno_path'])
    use_gt = question['use_gt']
    labeled = question['labeled']

    npy = dict(np.load(anno_path, allow_pickle=True))
    pose = {
        'azim': round(float(npy['azimuth'])/np.pi*180.0, 2),
        'elev': round(float(npy['elevation'])/np.pi*180.0, 2),
        'theta': round(float(npy['theta'])/np.pi*180.0, 2),
        'dist': round(float(npy['distance']), 2),
        'px': round(float(npy['px']), 2),
        'py': round(float(npy['py']), 2)}
    model_idx = npy['cad_index']

    cad_path = os.path.join(CAD_PATH, question['cad_path'], f'{model_idx:02d}.off')

    rendered_img = _render(
        img_path, cad_path, annotator=flask_login.current_user.id,
        **pose)
    pose_img = _render_pose(
        cad_path, annotator=flask_login.current_user.id,
        **pose)

    annotator_id = flask_login.current_user.id
    annotator = get_annotator(annotator_id)
    if not annotator[0]['pass_test']:
        return redirect(url_for('account'))
    else:
        return render_template(
            'quality.html', **pose, annotator_id=annotator_id,
            rendered_img=rendered_img, pose_img=pose_img,
            model_id=get_model_id(cad_path),
            img_path=question['img_path'],
            anno_path=question['anno_path'],
            task_id=task,
            question_id=q_idx,
            use_gt=use_gt,
            labeled=labeled)


@app.route('/annotate')
@flask_login.login_required
def annotate():
    task = request.args.get('task', type=str)
    q_idx = request.args.get('q_idx', default=-1, type=int)
    if q_idx == -1:
        question, found_next, q_idx = get_next_unannotated_question(task, 0)
        if not found_next:
            return redirect(url_for('account'))
    else:
        question, q_idx = get_question(task, q_idx)

    img_path = os.path.join(DATA_PATH, question['img_path'])
    anno_path = os.path.join(DATA_PATH, question['anno_path'])
    initialized = question['initialized']
    finished = question['finished']

    if finished:
        npy = dict(np.load(anno_path.replace('annotations', 'annotations_new'), allow_pickle=True))
    else:
        npy = dict(np.load(anno_path, allow_pickle=True))

    pose = {
        'azim': round(float(npy['azimuth'])/np.pi*180.0, 2),
        'elev': round(float(npy['elevation'])/np.pi*180.0, 2),
        'theta': round(float(npy['theta'])/np.pi*180.0, 2),
        'dist': round(float(npy['distance']), 2),
        'px': round(float(npy['px']), 2),
        'py': round(float(npy['py']), 2)}
    model_idx = npy['cad_index']
    viewport = npy['viewport']

    cad_path = os.path.join(CAD_PATH, question['cad_path'], f'{model_idx:02d}.off')

    rendered_img = _render(
        img_path, cad_path, annotator=flask_login.current_user.id, viewport=viewport,
        **pose)
    pose_img = _render_pose(
        cad_path, annotator=flask_login.current_user.id, viewport=viewport,
        **pose)

    annotator_id = flask_login.current_user.id
    annotator = get_annotator(annotator_id)
    if not annotator[0]['pass_test']:
        return redirect(url_for('account'))
    else:
        return render_template(
            'index.html', **pose, annotator_id=annotator_id,
            rendered_img=rendered_img, pose_img=pose_img,
            model_id=get_model_id(cad_path),
            img_path=question['img_path'],
            anno_path=question['anno_path'],
            task_id=task,
            question_id=q_idx,
            status_initialized=initialized,
            status_annotated=finished,
            cate=question['cad_path'],
            viewport=viewport,
            object_status=npy['object_status'] if 'object_status' in npy else 'undefined',
            dense=npy['dense'] if 'dense' in npy else 'undefined')


@app.route('/eval')
@flask_login.login_required
def eval():
    task = request.args.get('task', type=str)
    q_idx = request.args.get('q_idx', default=-1, type=int)
    if q_idx == -1:
        question, found_next, q_idx = get_next_unannotated_question_eval(task, 0)
        if not found_next:
            return redirect(url_for('account'))
    else:
        question, q_idx = get_question_eval(task, q_idx)

    img_path = os.path.join(DATA_PATH, question['img_path'])
    anno_path = os.path.join(DATA_PATH, question['anno_path'])
    finished = question['finished']
    gt = question['answer']

    if finished:
        with open(os.path.join(DATA_PATH, 'eval', question['anno_path'].replace('npz', 'txt').replace('annotations/', '')), 'r') as fp:
            pred = fp.read().strip()[0]
    else:
        pred = 'None'

    npy = dict(np.load(anno_path, allow_pickle=True))
    pose = {
        'azim': round(float(npy['azimuth'])/np.pi*180.0, 2),
        'elev': round(float(npy['elevation'])/np.pi*180.0, 2),
        'theta': round(float(npy['theta'])/np.pi*180.0, 2),
        'dist': round(float(npy['distance']), 2),
        'px': round(float(npy['px']), 2),
        'py': round(float(npy['py']), 2)}
    model_idx = npy['cad_index']
    viewport = npy['viewport']
    cad_path = os.path.join(CAD_PATH, question['cad_path'], f'{model_idx:02d}.off')
    rendered_img1 = _render(
        img_path, cad_path, annotator=flask_login.current_user.id+'eval1', viewport=viewport,
        **pose)

    npy = dict(np.load(anno_path.replace('annotations', 'annotations_new'), allow_pickle=True))
    pose = {
        'azim': round(float(npy['azimuth'])/np.pi*180.0, 2),
        'elev': round(float(npy['elevation'])/np.pi*180.0, 2),
        'theta': round(float(npy['theta'])/np.pi*180.0, 2),
        'dist': round(float(npy['distance']), 2),
        'px': round(float(npy['px']), 2),
        'py': round(float(npy['py']), 2)}
    model_idx = npy['cad_index']
    viewport = npy['viewport']
    cad_path = os.path.join(CAD_PATH, question['cad_path'], f'{model_idx:02d}.off')
    rendered_img2 = _render(
        img_path, cad_path, annotator=flask_login.current_user.id+'eval2', viewport=viewport,
        **pose)

    if gt == 1:
        rendered_img1, rendered_img2 = rendered_img2, rendered_img1

    _src = os.path.join(DATA_PATH, question['img_path'])
    _dst = f'tmp/orig_{question["cad_path"]}_{os.path.basename(question["img_path"])}'
    shutil.copyfile(_src, f'static/{_dst}')

    annotator_id = flask_login.current_user.id
    annotator = get_annotator(annotator_id)
    if not annotator[0]['pass_test']:
        return redirect(url_for('account'))
    else:
        return render_template(
            'eval.html', annotator_id=annotator_id,
            img1=rendered_img1, img2=rendered_img2,
            model_id=get_model_id(cad_path),
            img_path=question['img_path'],
            img0=_dst,
            anno_path=question['anno_path'],
            task_id=task,
            question_id=q_idx,
            status_annotated=finished,
            cate=question['cad_path'],
            pred=pred)


@app.route('/logout')
def logout():
    flask_login.logout_user()
    return redirect(url_for('login'))


@app.route('/doc')
def doc():
    return render_template('doc.html')


def _render_pose(cad_path, azim, elev, theta, dist, px, py, annotator, viewport):
    pose = {
        'azimuth': float(azim),
        'elevation': float(elev),
        'theta': float(theta),
        'distance': 8.0,
        'px': 192.0,
        'py': 192.0}
    img = matplotlib_line_render(None, cad_path, pose, color=(39, 158, 255), alpha=1.0, add_margin=True, viewport=viewport)
    [os.remove(f) for f in glob.glob(f'static/tmp/{annotator}_*.jpg')]
    fname = f'tmp/{annotator}_{uuid.uuid4().hex}.jpg'
    img.save(f'static/{fname}')
    return fname


def _render(img_path, cad_path, azim, elev, theta, dist, px, py, annotator, viewport):
    pose = {
        'azimuth': float(azim),
        'elevation': float(elev),
        'theta': float(theta),
        'distance': float(dist),
        'px': float(px),
        'py': float(py)}
    img = matplotlib_line_render(img_path, cad_path, pose, color=(39, 158, 255), alpha=0.75, viewport=viewport)
    [os.remove(f) for f in glob.glob(f'static/tmp/render_{annotator}_*.jpg')]
    fname = f'tmp/render_{annotator}_{uuid.uuid4().hex}.jpg'
    img.save(f'static/{fname}')
    return fname


@app.route('/azim_dec', methods=["POST"])
def azim_dec():
    azim = request.args.get('azim', default=0.0, type=float)
    elev = request.args.get('elev', default=0.0, type=float)
    theta = request.args.get('theta', default=0.0, type=float)
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    img_path = os.path.join(DATA_PATH, request.args.get('img_path', type=str))
    viewport = request.args.get('viewport', type=float)

    new_azim = azim - step_size['azim']

    cad_path = os.path.join(CAD_PATH, model_id.replace('_', '/')+'.off')

    return jsonify({
        'rendered_img': _render(img_path, cad_path, new_azim, elev, theta, dist, px, py, annotator, viewport),
        'pose_img': _render_pose(cad_path, new_azim, elev, theta, dist, px, py, annotator, viewport),
        'key': 'azim',
        'value': new_azim})


@app.route('/azim_inc', methods=["POST"])
def azim_inc():
    azim = request.args.get('azim', default=0.0, type=float)
    elev = request.args.get('elev', default=0.0, type=float)
    theta = request.args.get('theta', default=0.0, type=float)
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    img_path = os.path.join(DATA_PATH, request.args.get('img_path', type=str))
    viewport = request.args.get('viewport', type=float)

    new_azim = azim + step_size['azim']

    cad_path = os.path.join(CAD_PATH, model_id.replace('_', '/')+'.off')

    return jsonify({
        'rendered_img': _render(img_path, cad_path, new_azim, elev, theta, dist, px, py, annotator, viewport),
        'pose_img': _render_pose(cad_path, new_azim, elev, theta, dist, px, py, annotator, viewport),
        'key': 'azim',
        'value': new_azim})


@app.route('/elev_dec', methods=["POST"])
def elev_dec():
    azim = request.args.get('azim', default=0.0, type=float)
    elev = request.args.get('elev', default=0.0, type=float)
    theta = request.args.get('theta', default=0.0, type=float)
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    img_path = os.path.join(DATA_PATH, request.args.get('img_path', type=str))
    viewport = request.args.get('viewport', type=float)

    new_elev = elev - step_size['elev']

    cad_path = os.path.join(CAD_PATH, model_id.replace('_', '/')+'.off')

    return jsonify({
        'rendered_img': _render(img_path, cad_path, azim, new_elev, theta, dist, px, py, annotator, viewport),
        'pose_img': _render_pose(cad_path, azim, new_elev, theta, dist, px, py, annotator, viewport),
        'key': 'elev',
        'value': new_elev})


@app.route('/elev_inc', methods=["POST"])
def elev_inc():
    azim = request.args.get('azim', default=0.0, type=float)
    elev = request.args.get('elev', default=0.0, type=float)
    theta = request.args.get('theta', default=0.0, type=float)
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    img_path = os.path.join(DATA_PATH, request.args.get('img_path', type=str))
    viewport = request.args.get('viewport', type=float)

    new_elev = elev + step_size['elev']

    cad_path = os.path.join(CAD_PATH, model_id.replace('_', '/')+'.off')

    return jsonify({
        'rendered_img': _render(img_path, cad_path, azim, new_elev, theta, dist, px, py, annotator, viewport),
        'pose_img': _render_pose(cad_path, azim, new_elev, theta, dist, px, py, annotator, viewport),
        'key': 'elev',
        'value': new_elev})


@app.route('/theta_dec', methods=["POST"])
def theta_dec():
    azim = request.args.get('azim', default=0.0, type=float)
    elev = request.args.get('elev', default=0.0, type=float)
    theta = request.args.get('theta', default=0.0, type=float)
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    img_path = os.path.join(DATA_PATH, request.args.get('img_path', type=str))
    viewport = request.args.get('viewport', type=float)

    new_theta = theta - step_size['theta']

    cad_path = os.path.join(CAD_PATH, model_id.replace('_', '/')+'.off')

    return jsonify({
        'rendered_img': _render(img_path, cad_path, azim, elev, new_theta, dist, px, py, annotator, viewport),
        'pose_img': _render_pose(cad_path, azim, elev, new_theta, dist, px, py, annotator, viewport),
        'key': 'theta',
        'value': new_theta})


@app.route('/theta_inc', methods=["POST"])
def theta_inc():
    azim = request.args.get('azim', default=0.0, type=float)
    elev = request.args.get('elev', default=0.0, type=float)
    theta = request.args.get('theta', default=0.0, type=float)
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    img_path = os.path.join(DATA_PATH, request.args.get('img_path', type=str))
    viewport = request.args.get('viewport', type=float)

    new_theta = theta + step_size['theta']

    cad_path = os.path.join(CAD_PATH, model_id.replace('_', '/')+'.off')

    return jsonify({
        'rendered_img': _render(img_path, cad_path, azim, elev, new_theta, dist, px, py, annotator, viewport),
        'pose_img': _render_pose(cad_path, azim, elev, new_theta, dist, px, py, annotator, viewport),
        'key': 'theta',
        'value': new_theta})


@app.route('/dist_dec', methods=["POST"])
def dist_dec():
    azim = request.args.get('azim', default=0.0, type=float)
    elev = request.args.get('elev', default=0.0, type=float)
    theta = request.args.get('theta', default=0.0, type=float)
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    img_path = os.path.join(DATA_PATH, request.args.get('img_path', type=str))
    viewport = request.args.get('viewport', type=float)

    new_dist = dist - step_size['dist']

    cad_path = os.path.join(CAD_PATH, model_id.replace('_', '/')+'.off')

    return jsonify({
        'rendered_img': _render(img_path, cad_path, azim, elev, theta, new_dist, px, py, annotator, viewport),
        'pose_img': _render_pose(cad_path, azim, elev, theta, new_dist, px, py, annotator, viewport),
        'key': 'dist',
        'value': new_dist})


@app.route('/dist_inc', methods=["POST"])
def dist_inc():
    azim = request.args.get('azim', default=0.0, type=float)
    elev = request.args.get('elev', default=0.0, type=float)
    theta = request.args.get('theta', default=0.0, type=float)
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    img_path = os.path.join(DATA_PATH, request.args.get('img_path', type=str))
    viewport = request.args.get('viewport', type=float)

    new_dist = dist + step_size['dist']

    cad_path = os.path.join(CAD_PATH, model_id.replace('_', '/')+'.off')

    return jsonify({
        'rendered_img': _render(img_path, cad_path, azim, elev, theta, new_dist, px, py, annotator, viewport),
        'pose_img': _render_pose(cad_path, azim, elev, theta, new_dist, px, py, annotator, viewport),
        'key': 'dist',
        'value': new_dist})


@app.route('/px_dec', methods=["POST"])
def px_dec():
    azim = request.args.get('azim', default=0.0, type=float)
    elev = request.args.get('elev', default=0.0, type=float)
    theta = request.args.get('theta', default=0.0, type=float)
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    img_path = os.path.join(DATA_PATH, request.args.get('img_path', type=str))
    viewport = request.args.get('viewport', type=float)

    new_px = px - step_size['px']

    cad_path = os.path.join(CAD_PATH, model_id.replace('_', '/')+'.off')

    return jsonify({
        'rendered_img': _render(img_path, cad_path, azim, elev, theta, dist, new_px, py, annotator, viewport),
        'pose_img': _render_pose(cad_path, azim, elev, theta, dist, new_px, py, annotator, viewport),
        'key': 'px',
        'value': new_px})


@app.route('/px_inc', methods=["POST"])
def px_inc():
    azim = request.args.get('azim', default=0.0, type=float)
    elev = request.args.get('elev', default=0.0, type=float)
    theta = request.args.get('theta', default=0.0, type=float)
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    img_path = os.path.join(DATA_PATH, request.args.get('img_path', type=str))
    viewport = request.args.get('viewport', type=float)

    new_px = px + step_size['px']

    cad_path = os.path.join(CAD_PATH, model_id.replace('_', '/')+'.off')

    return jsonify({
        'rendered_img': _render(img_path, cad_path, azim, elev, theta, dist, new_px, py, annotator, viewport),
        'pose_img': _render_pose(cad_path, azim, elev, theta, dist, new_px, py, annotator, viewport),
        'key': 'px',
        'value': new_px})


@app.route('/py_dec', methods=["POST"])
def py_dec():
    azim = request.args.get('azim', default=0.0, type=float)
    elev = request.args.get('elev', default=0.0, type=float)
    theta = request.args.get('theta', default=0.0, type=float)
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    img_path = os.path.join(DATA_PATH, request.args.get('img_path', type=str))
    viewport = request.args.get('viewport', type=float)

    new_py = py - step_size['py']

    cad_path = os.path.join(CAD_PATH, model_id.replace('_', '/')+'.off')

    return jsonify({
        'rendered_img': _render(img_path, cad_path, azim, elev, theta, dist, px, new_py, annotator, viewport),
        'pose_img': _render_pose(cad_path, azim, elev, theta, dist, px, new_py, annotator, viewport),
        'key': 'py',
        'value': new_py})


@app.route('/py_inc', methods=["POST"])
def py_inc():
    azim = request.args.get('azim', default=0.0, type=float)
    elev = request.args.get('elev', default=0.0, type=float)
    theta = request.args.get('theta', default=0.0, type=float)
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    img_path = os.path.join(DATA_PATH, request.args.get('img_path', type=str))
    viewport = request.args.get('viewport', type=float)

    new_py = py + step_size['py']

    cad_path = os.path.join(CAD_PATH, model_id.replace('_', '/')+'.off')

    return jsonify({
        'rendered_img': _render(img_path, cad_path, azim, elev, theta, dist, px, new_py, annotator, viewport),
        'pose_img': _render_pose(cad_path, azim, elev, theta, dist, px, new_py, annotator, viewport),
        'key': 'py',
        'value': new_py})


@app.route('/model_dec', methods=["POST"])
def model_dec():
    azim = request.args.get('azim', default=0.0, type=float)
    elev = request.args.get('elev', default=0.0, type=float)
    theta = request.args.get('theta', default=0.0, type=float)
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    img_path = os.path.join(DATA_PATH, request.args.get('img_path', type=str))
    viewport = request.args.get('viewport', type=float)

    c = '_'.join(model_id.split('_')[:-1])
    m = model_id.split('_')[-1]
    m = int(m)
    if m == 1:
        m = NUM_MODELS[c]
    else:
        m -= 1
    new_model_id = f'{c}_{m:02d}'
    cad_path = os.path.join(CAD_PATH, new_model_id.replace('_', '/')+'.off')

    return jsonify({
        'rendered_img': _render(img_path, cad_path, azim, elev, theta, dist, px, py, annotator, viewport),
        'pose_img': _render_pose(cad_path, azim, elev, theta, dist, px, py, annotator, viewport),
        'key': 'model_id',
        'value': new_model_id})


@app.route('/model_inc', methods=["POST"])
def model_inc():
    azim = request.args.get('azim', default=0.0, type=float)
    elev = request.args.get('elev', default=0.0, type=float)
    theta = request.args.get('theta', default=0.0, type=float)
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    img_path = os.path.join(DATA_PATH, request.args.get('img_path', type=str))
    viewport = request.args.get('viewport', type=float)

    c = '_'.join(model_id.split('_')[:-1])
    m = model_id.split('_')[-1]
    m = int(m)
    if m == NUM_MODELS[c]:
        m = 1
    else:
        m += 1
    new_model_id = f'{c}_{m:02d}'
    cad_path = os.path.join(CAD_PATH, new_model_id.replace('_', '/')+'.off')

    return jsonify({
        'rendered_img': _render(img_path, cad_path, azim, elev, theta, dist, px, py, annotator, viewport),
        'pose_img': _render_pose(cad_path, azim, elev, theta, dist, px, py, annotator, viewport),
        'key': 'model_id',
        'value': new_model_id})


@app.route('/text', methods=["POST"])
def text():
    azim = request.args.get('azim', default=0.0, type=float)
    elev = request.args.get('elev', default=0.0, type=float)
    theta = request.args.get('theta', default=0.0, type=float)
    dist = request.args.get('dist', default=0.0, type=float)
    px = request.args.get('px', default=0.0, type=float)
    py = request.args.get('py', default=0.0, type=float)
    model_id = request.args.get('model_id', default=1, type=str)
    annotator = request.args.get('annotator', default='none', type=str)
    img_path = os.path.join(DATA_PATH, request.args.get('img_path', type=str))
    viewport = request.args.get('viewport', type=float)

    cad_path = os.path.join(CAD_PATH, model_id.replace('_', '/')+'.off')

    return jsonify({
        'rendered_img': _render(img_path, cad_path, azim, elev, theta, dist, px, py, annotator, viewport),
        'pose_img': _render_pose(cad_path, azim, elev, theta, dist, px, py, annotator, viewport)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6260, debug=False)
