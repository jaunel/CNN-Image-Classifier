from flask import Flask,render_template,request
import tensorflow as tf
import cv2

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['file']                     #accessing file from html
    imagefile.save(f'images/{imagefile.filename}')        #saving image on the given path in parenthesis
    mymodel = tf.keras.models.load_model('mymodel.h5')    #loading the model
    test_img = cv2.imread(f'images/{imagefile.filename}') #loading the photo from images directory
    test_img = cv2.resize(test_img, (256, 256))           #resizing
    test_img = test_img.reshape(1,256,256,3)
    result = mymodel.predict(test_img)[0]
    if result == 1:
        return render_template('index.html',label = "It's a Dog!")
    else:
        return render_template('index.html',label= "It's a Cat!")

if __name__ == '__main__':
    app.run(port=9000,debug = True)

