import streamlit as st
import cv2
import torch
from utils.hubconf import custom
import numpy as np
import tempfile
import time
from collections import Counter
import json
import pandas as pd
from model_utils import get_yolo, color_picker_fn, get_system_stat
# from ultralytics import YOLO

p_time = 0
col1, col2 = st.columns(2)

st.sidebar.title('Settings')
# Choose the model
model_type = st.sidebar.selectbox(
    'Choose YOLO Model', ('YOLO Model', 'YOLOv8')
)


with col1:
    st.title('Webcam')
    sample_img = cv2.imread('logo.jpg')
    FRAME_WINDOW = st.image(sample_img, channels='BGR')
    cap = None

if not model_type == 'YOLO Model':
    path_model_file = st.sidebar.text_input(
        f'path to {model_type} Model:',
        f'best.pt'
    )
    if st.sidebar.checkbox('Load Model'):

        # YOLOv8 Model
        if model_type == 'YOLOv8':
            from ultralytics import YOLO
            model = YOLO(path_model_file)

        # Load Class names
        class_labels = model.names

        # Inference Mode
        options = st.sidebar.radio(
            'Options:', ('Webcam', 'Image'), index=1)

        # Confidence
        confidence = st.sidebar.slider(
            'Detection Confidence', min_value=0.0, max_value=1.0, value=0.25)

        # Draw thickness
        draw_thick = st.sidebar.slider(
            'Draw Thickness:', min_value=1,
            max_value=20, value=3
        )
        
        color_pick_list = []
        for i in range(len(class_labels)):
            classname = class_labels[i]
            color = color_picker_fn(classname, i)
            color_pick_list.append(color)

        # Image
        if options == 'Image':  
            upload_img_file = st.sidebar.file_uploader(
                'Upload Image', type=['jpg', 'jpeg', 'png'])
            if upload_img_file is not None:
                pred = st.sidebar.checkbox(f'Predict Using {model_type}')
                file_bytes = np.asarray(
                    bytearray(upload_img_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                FRAME_WINDOW.image(img, channels='BGR')

                if pred:
                    img, current_no_class = get_yolo(img, model_type, model, confidence, color_pick_list, class_labels, draw_thick)
                    FRAME_WINDOW.image(img, channels='BGR')

                    # Current number of classes
                    class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
                    class_fq = json.dumps(class_fq, indent = 4)
                    class_fq = json.loads(class_fq)
                    df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
                    
                    # Updating Inference results
                    st.dataframe(df_fq, use_container_width=True)

                    # Result messages based on detected classes
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'Happy' in df_fq['Class'].values:
                            st.write("**Aku Sangat Gembira!**")
                    with col2:
                        # Cek apakah terdapat kelas 'Sad' dalam DataFrame
                        if 'Sad' in df_fq['Class'].values:
                            st.markdown("**Aku Tidak Gembira.**")
                        
        # Web-cam
                            
        if options == 'Webcam':
            cam_options = st.sidebar.selectbox('Webcam Channel',
                                            ('Select Channel', '0', '1', '2', '3'))
        
            if not cam_options == 'Select Channel':
                pred = st.sidebar.checkbox(f'Predict Using {model_type}')
                cap = cv2.VideoCapture(int(cam_options))

with col2:
    st.title('SOP Tindakan')
    prev_emotion = None  # Inisialisasi variabel untuk menyimpan emosi sebelumnya
    message_container = st.empty()  # Membuat area kosong untuk pesan
    message_container2 = st.empty()  # Membuat area kosong untuk pesan
    message_container3 = st.empty()  # Membuat area kosong untuk pesan
    if (cap != None) and pred:
        while True:
            success, img = cap.read()
            if not success:
                st.sidebar.error(
                    f"{options} NOT working\nCheck {options} properly!!",
                    icon="🚨"
                )
                break

            img, current_no_class = get_yolo(img, model_type, model, confidence, color_pick_list, class_labels, draw_thick)
            FRAME_WINDOW.image(img, channels='BGR')
            
            # Current number of classes
            class_fq = dict(Counter(i for sub in current_no_class for i in set(sub)))
            class_fq = json.dumps(class_fq, indent=4)
            class_fq = json.loads(class_fq)
            df_fq = pd.DataFrame(class_fq.items(), columns=['Class', 'Number'])
    
            # Ambil emosi yang terdeteksi
            current_emotion = None
            if 'Happy' in df_fq['Class'].values:
                current_emotion = 'Happy'
            elif 'Anger' in df_fq['Class'].values:
                current_emotion = 'Anger'
            # ... tambahkan kondisi untuk emosi lainnya

            # Cek apakah emosi berubah sejak deteksi sebelumnya
            if current_emotion != prev_emotion:
                # Hapus pesan sebelumnya
                message_container.empty()

                # Tampilkan pesan baru dan perbarui prev_emotion
                if current_emotion == 'Happy':
                    st.write("")
                    st.write("")
                    st.write("")
                    message_container.write("- Tanyakan Kebutuhan Lainnya")
                    message_container2.write("- Tawarkan Produk Lain!")
                    message_container3.write("- Teruskan Suasana yang Baik")
                elif current_emotion == 'Anger':
                    st.write("")
                    st.write("")
                    st.write("")
                    message_container.write("- Jangan Mencela Perkataannya")
                    message_container2.write("- Coba Tenangkan Bapak/Ibu Tersebut")
                    message_container3.write("- Berikan Solusi Lain")
                # ... tambahkan pesan untuk emosi lainnya
                
                prev_emotion = current_emotion


