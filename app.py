import os
from flask import Flask, send_file, request, render_template, redirect, url_for, send_from_directory, jsonify
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
import label_video
from biggan import gen_biggan_arr, create_latent_vectors_from_video, create_class_vectors, test_process
import cv2
import threading
import re

ALLOWED_EXTENSIONS = set(['mp4','avi','mov','jpg', 'jpeg'])
FLOYD_API_URL = "https://www.floydlabs.com/serve/cliveunger/projects/biggan-videos/"
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#OUTPUT_FOLDER = 'static/outputs/' os.path.dirname(os.path.abspath(__file__)) +
OUTPUT_FOLDER = 'static/outputs' #'/output/'
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

print("app.config['UPLOAD_FOLDER'] = ", app.config['UPLOAD_FOLDER'])
print("app.config['OUTPUT_FOLDER'] = ",  app.config['OUTPUT_FOLDER'])

tasks = []
results = []

@app.route('/', methods=["GET", "POST"])
def home():
    return render_template("home.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        input_file = request.files.get('file')
        if not input_file:
            return BadRequest("File not present in request")

        # if user does not select file, browser also
        # submit an empty part without filename
        filename = secure_filename(input_file.filename)
        if filename == '':
            return BadRequest("File name is not present in request")
        if not allowed_file(filename):
            return BadRequest("Invalid file type")

        input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = "processed_" + filename.split('.')[0] + ".mp4"
        output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        input_file.save(input_filepath)


        new_task_id = len(tasks)
        x = threading.Thread(target=create_biggan_video,
                                args=(input_filepath, output_filepath, new_task_id))
        x.start()
        tasks.append(x)
        results.append(None)
        #create_biggan_video(input_filepath, output_filepath)

        return jsonify({'task_id': new_task_id,
            'task_endpoint': FLOYD_API_URL + "/task/" + str(new_task_id), #+ url_for('task_status', task_id = new_task_id),
            'total_tasks': len(tasks)})
        #redirect(url_for('home'))
        #return send_from_directory(app.config['OUTPUT_FOLDER'], output_filename, as_attachment=True)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route("/task/<int:task_id>", methods=["GET"])
def task_status(task_id):
    assert 0 <= task_id < len(tasks)
    if tasks[task_id].is_alive():
        return jsonify({"status": "running"})
    try:
        tasks[task_id].join()
        return jsonify({"status": "finished", \
         "result": FLOYD_API_URL + "/outputs/" + results[task_id],  #url_for('output_file', filename=results[task_id]), \
         "path": url_for('output_file', filename=results[task_id]) })
        #return send_from_directory(app.config['OUTPUT_FOLDER'], results[task_id], as_attachment=True)
    except RuntimeError:
        return jsonify({"status": "not started"})


@app.route('/outputs/<filename>')
def output_file(filename):
   return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def write_video(file_name, frame_array, fps):
  frame_width = frame_array[0].shape[1]
  frame_height = frame_array[0].shape[0]
  #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
  fourcc = cv2.VideoWriter_fourcc(*'DIVX')
  #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
  #fourcc = cv2.VideoWriter_fourcc('i', 'Y', 'U', 'V')
  out = cv2.VideoWriter(file_name, fourcc, fps, (frame_width,frame_height))
  for f in frame_array:
    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
  out.release()


def create_biggan_video(input_filepath, output_filepath, task_id):
    frame_arr, fps = label_video.read_video(input_filepath)
    predictions = label_video.make_frame_predictions(frame_arr)
    avg_predictions = label_video.average_predictions(predictions)

    latent_vs = create_latent_vectors_from_video(frame_arr)
    print("Latent Vectors: ", latent_vs.shape)
    class_vs, label_list = create_class_vectors(avg_predictions)
    print("Class Vectors: ", class_vs.shape)


    #gan_arr = gen_biggan_arr(latent_vs, class_vs, label_list)
    gan_arr = test_process(frame_arr, label_list)
    print("Length of GAN Array: ", len(gan_arr))
    print("Shape of GAN frame: ", gan_arr[0].shape)

    write_video(output_filepath, gan_arr, fps)

    results[task_id] = output_filepath.replace('\\', '/').split('/')[-1]
    print(results[task_id])



if __name__ == '__main__':
    app.run()
