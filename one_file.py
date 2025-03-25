import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

from deepface import DeepFace

img_path = 'test_img/your_image.jpg'

prediction = DeepFace.analyze(img_path, actions=['age', 'gender', 'race'], align=True, detector_backend ='mtcnn')

print(prediction)