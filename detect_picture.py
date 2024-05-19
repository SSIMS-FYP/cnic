import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import easyocr
import json
from scipy.spatial import distance
import os
import csv

# Load YOLOv8 model for ID card detection
id_card_detector = YOLO('best.pt')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Output folder for cropped ID card images
output_folder = 'id_card_images'
os.makedirs(output_folder, exist_ok=True)

def detect_id_card_in_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Detect ID cards using YOLOv8
    id_cards = id_card_detector(image)[0]

    for id_card in id_cards.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = id_card

        # Crop the ID card region
        id_card_crop = image[int(y1):int(y2), int(x1):int(x2), :]

        # Save the cropped ID card image
        image_name = f"id_card_{os.path.basename(image_path)}"
        cv2.imwrite(os.path.join(output_folder, image_name), id_card_crop)

        # Extract information from the ID card
        extracted_data = extract_cnic_information(os.path.join(output_folder, image_name))

        # Store extracted data in a CSV file
        csv_filename = 'extracted_data.csv'
        write_to_csv(csv_filename, extracted_data)

def normalize(img,result):
    w,h = img.shape[:-1]
    normalize_bbx = []
    detected_labels = []
    for (bbox, text, prob) in result:
        (tl, tr, br, bl) = bbox
        tl[0],tl[1] = round(tl[0] / h,3),round(tl[1] / w,3)
        tr[0],tr[1] = round(tr[0] / h,3),round(tr[1] / w,3)
        br[0],br[1] = round(br[0] / h,3),round(br[1] / w,3)
        bl[0],bl[1] = round(bl[0] / h,3),round(bl[1] / w,3)
        normalize_bbx.append([tl,tr,br,bl])
        detected_labels.append(text)
    return normalize_bbx,detected_labels

def calculate_distance(key,bbx):
    euc_sum = 0
    for val1,val2 in zip(key,bbx):
        euc_sum = euc_sum + distance.euclidean(val1,val2)
        return euc_sum

def get_value(key,normalize_output):
    distances = {}
    for bbx,text in normalize_output:
        distances[text] = calculate_distance(key,bbx)
    return distances

def extract_cnic_information(image_path):
    # Read the image
    im = Image.open(image_path)
    image = cv2.imread(image_path)

    # Apply OCR to the image
    result = reader.readtext(image_path)

    # Normalize bounding boxes
    norm_boxes, labels = normalize(image, result)
    normalize_output = list(zip(norm_boxes, labels))

    # Define template boxes for specific fields
    name_value = [[0.283, 0.271], [0.415, 0.271], [0.415, 0.325], [0.283, 0.325]]
    father_value = [[0.29, 0.456], [0.494, 0.456], [0.494, 0.514], [0.29, 0.514]]
    dob_value = [[0.529, 0.751], [0.648, 0.751], [0.648, 0.803], [0.529, 0.803]]
    doi_value = [[0.285, 0.857], [0.404, 0.857], [0.404, 0.908], [0.285, 0.908]]
    doe_value = [[0.531, 0.859], [0.65, 0.859], [0.65, 0.911], [0.531, 0.911]]
    id_card = [[0.285, 0.742], [0.643, 0.742], [0.643, 0.791], [0.285, 0.791]]

    # name_value = [[0.283, 0.271], [0.415, 0.271], [0.415, 0.325], [0.283, 0.325]]
    # # father_value = [[0.29, 0.456], [0.494, 0.456], [0.494, 0.514], [0.29, 0.514]]
    # father_value = [[0.29, 0.446], [0.514, 0.446], [0.514, 0.524], [0.29, 0.524]]
    # # dob_value = [[0.529, 0.751], [0.648, 0.751], [0.648, 0.803], [0.529, 0.803]]
    # dob_value = [[0.529, 0.751], [0.748, 0.751], [0.748, 0.803], [0.529, 0.803]]
    # # doi_value = [[0.285, 0.857], [0.404, 0.857], [0.404, 0.908], [0.285, 0.908]]
    # doi_value = [[0.285, 0.857], [0.48, 0.857], [0.48, 0.908], [0.285, 0.908]]
    # # doe_value = [[0.531, 0.859], [0.65, 0.859], [0.65, 0.911], [0.531, 0.911]]
    # doe_value = [[0.531, 0.859], [0.726, 0.859], [0.726, 0.911], [0.531, 0.911]]
    # # id_card = [[0.285, 0.742], [0.643, 0.742], [0.643, 0.791], [0.285, 0.791]]
    # id_card = [[0.285, 0.742], [0.658, 0.742], [0.658, 0.791], [0.285, 0.791]]

    # Extract information based on template boxes
    dict_data = {}
    output_dict = {
        'Name': name_value,
        'Father Name': father_value,
        'Date of Birth': dob_value,
        'Date of Issue': doi_value,
        'Date of Expiry': doe_value,
        'Card Number': id_card
    }

    for key, value in output_dict.items():
        output_dict = get_value(value, normalize_output)
        answer = list(min(output_dict.items(), key=lambda x: x[1]))[0]
        dict_data[key] = answer

    return dict_data

def write_to_csv(csv_filename, data):
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if os.stat(csv_filename).st_size == 0:
            writer.writeheader()
        writer.writerow(data)

if __name__ == "__main__":
    input_image_path = 'id_card_images\id_card_test4.jpg'
    detect_id_card_in_image(input_image_path)
