from flask import Flask,render_template,redirect,request,send_from_directory
import tensorflow as tf
from keras import layers;
import os
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

model_file = "cnn.hdf5"

model = tf.keras.models.load_model(model_file)

app = Flask(__name__,template_folder='templates')
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 

class_names=np.array(['Monkey Pox', 'Others'])

def makePredictions(path):
  
    img=tf.io.read_file(path)
    img=tf.image.decode_image(img)
    img=tf.image.resize(img,size=[224,224])
    img=img/255.
    pred=model.predict(tf.expand_dims(img,axis=0))
    #print(pred)
    if len(pred[0])>1:
        pred_class=class_names[tf.argmax(pred[0])]
    else:
        pred_class=class_names[int(tf.round(pred[0]))]
        
    if pred_class=='Monkey Pox':
        a="monkeypox"
    else:
        a="others"
    
    return a

@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='POST':
        if 'img' not in request.files:
            return render_template('home.html',filename="unnamed.png",message="Please upload an file")
        f = request.files['img'] 
        filename = secure_filename(f.filename) 
        if f.filename=='':
            return render_template('home.html',filename="unnamed.png",message="No file selected")
        if not ('jpeg' in f.filename or 'png' in f.filename or 'jpg' in f.filename):
            return render_template('home.html',filename="unnamed.png",message="please upload an image with .png or .jpg/.jpeg extension")
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if len(files)==1:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        else:
            files.remove("unnamed.png")
            file_ = files[0]
            os.remove(app.config['UPLOAD_FOLDER']+'/'+file_)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        predictions = makePredictions(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        return render_template('home.html',filename=f.filename,message=predictions,show=True)
    return render_template('home.html',filename='unnamed.png')

@app.route('/index.html',methods=['GET','POST'])
def profile():
    return render_template('index.html')

@app.route("/analysis")
def analysis():
    return render_template("analysis.html") 

@app.route("/stages")
def stages():
    return render_template("stages.html")

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)