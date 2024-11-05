from __future__ import division, print_function
import numpy as np
import os
import tensorflow as tf
from flask import Flask, request, render_template
from Model import final2
from PIL import Image
app = Flask(__name__)
detector_output,image_string_placeholder,decoded_image,init_ops,sess,draw_boxes=final2()
def model_detect(img_path):
    sample_image_path = img_path  

        # Load our sample image into a binary string
    with tf.compat.v1.gfile.Open(sample_image_path, "rb") as binfile:
            image_string = binfile.read()

        # Run the graph we just created
    sess.run(init_ops)
    result_out, image_out = sess.run(
                [detector_output, decoded_image],feed_dict={image_string_placeholder: image_string}
            )
    
    image_with_boxes = draw_boxes(
    np.array(image_out), result_out["detection_boxes"],
    result_out["detection_class_entities"], result_out["detection_scores"])
    
    return image_with_boxes


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        print("save sucessfully")
        print(type(f.filename))
        preds = model_detect(f.filename)
        data = Image.fromarray(preds)
        data.save("static/preds.jpg")
        os.remove(f.filename)
    return render_template("image.html")

if __name__ == '__main__':
    app.run()
