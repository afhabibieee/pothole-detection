import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
from process import obj_detector
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2

# load  a model; pre-trained on COCO
saved_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, pretrained_backbone=False)
WEIGHTS_FILE = "faster_rcnn_state.pth"

# get number of input features for the classifier
in_features = saved_model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
saved_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the traines weights
saved_model.load_state_dict(torch.load(WEIGHTS_FILE))
saved_model = saved_model.to(device)


st.title('Pothole detection')

st.markdown(
    'This is a web application that allows us to detect potholes'
)
st.markdown('---')

motif = st.markdown(
    f"<h1 style='text-align: center; color: red;'>Hmm, how many holes were detected?</h1>", 
    unsafe_allow_html=True
)

st.markdown('---')

img_file_buffer = st.file_uploader(
    "Upload an image", 
    type=[ "jpg", "jpeg",'png']
)

st.markdown('---')

thresh = st.number_input('Insert a number', value=0.6)
st.write('The current threshold is ', thresh)

st.markdown('---')

if img_file_buffer is not None:
    image = cv2.imdecode(np.frombuffer(img_file_buffer.getvalue(), np.uint8), -1)
    #model = torch.jit.load(MODEL_PATH)
    boxes,sample = obj_detector(image, saved_model, device, thresh)
    
    motif.write(
        f"<h1 style='text-align: center; color: green;'>{boxes.shape[0]}</h1>", 
        unsafe_allow_html=True
    )

    p = plt.figure(figsize=(16, 8))
    for box in boxes:
        cv2.rectangle(sample,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (220, 0, 0), 2)
    plt.axis('off')
    plt.imshow((sample * 255).astype(np.uint8))
    st.pyplot(p)