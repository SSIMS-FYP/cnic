import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from scipy.spatial import distance
import csv
import re
import string
import random

# Load YOLOv8 model for ID card detection
id_card_detector = YOLO('best.pt')

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def normalize(img, result):
    w, h = img.shape[:-1]
    normalize_bbx = []
    detected_labels = []
    for (bbox, text, prob) in result:
        (tl, tr, br, bl) = bbox
        tl = [round(tl[0] / h, 3), round(tl[1] / w, 3)]
        tr = [round(tr[0] / h, 3), round(tr[1] / w, 3)]
        br = [round(br[0] / h, 3), round(br[1] / w, 3)]
        bl = [round(bl[0] / h, 3), round(bl[1] / w, 3)]
        normalize_bbx.append([tl, tr, br, bl])
        detected_labels.append((text, prob))  # Include confidence score along with label
    return normalize_bbx, detected_labels

def calculate_distance(key, bbx):
    euc_sum = 0
    for val1, val2 in zip(key, bbx):
        euc_sum += distance.euclidean(val1, val2)
    return euc_sum

def get_value(key, normalize_output):
    distances = {}
    for bbx, (text, prob) in normalize_output:
        distances[text] = (calculate_distance(key, bbx), prob)  # Include confidence score along with distance
    return distances

def add_bogus_letters(text, length):
    return text + ''.join(random.choices(string.ascii_letters, k=length))

def validate_and_correct_field(field, value):
    if field in ['Name', 'Father Name']:
        # Regular expression for a valid name containing only alphabets and spaces
        if re.match(r"^[A-Za-z ]+$", value):
            return value
        # If invalid, add bogus letters
        corrected_value = ''.join([char if char.isalpha() or char.isspace() else random.choice(string.ascii_letters) for char in value])
        return corrected_value

    elif field == 'Card Number':
        # Regular expression for a valid ID card number in the format 13101-6356172-4
        if re.match(r"^\d{5}-\d{7}-\d{1}$", value):
            return value
        # If invalid, correct the format
        digits = re.findall(r'\d', value)
        if len(digits) < 13:
            digits += random.choices(string.digits, k=13 - len(digits))
        elif len(digits) > 13:
            digits = digits[:13]
        corrected_value = f"{digits[0]}{digits[1]}{digits[2]}{digits[3]}{digits[4]}-{digits[5]}{digits[6]}{digits[7]}{digits[8]}{digits[9]}{digits[10]}{digits[11]}-{digits[12]}"
        return corrected_value

    elif field in ['Date of Birth', 'Date of Issue', 'Date of Expiry']:
        # Regular expression for a valid date in the format dd.mm.yyyy
        if re.match(r"^\d{2}\.\d{2}\.\d{4}$", value):
            return value
        # If invalid, correct the format
        corrected_value = re.sub(r'[^\d]', '', value)  # Remove non-digit characters
        if len(corrected_value) >= 8:
            corrected_value = f"{corrected_value[:2]}.{corrected_value[2:4]}.{corrected_value[4:8]}"
            return corrected_value
        # Add bogus digits if necessary
        while len(corrected_value) < 8:
            corrected_value += random.choice(string.digits)
        corrected_value = f"{corrected_value[:2]}.{corrected_value[2:4]}.{corrected_value[4:8]}"
        return corrected_value

    return value  # Return the original value if no specific validation is defined

def extract_cnic_information(image):
    # Apply OCR to the image
    result = reader.readtext(image)

    # Normalize bounding boxes
    norm_boxes, labels = normalize(image, result)
    normalize_output = list(zip(norm_boxes, labels))

    # Define template boxes for specific fields
    name_value = [[0.283, 0.271], [0.415, 0.271], [0.415, 0.325], [0.283, 0.325]]
    father_value = [[0.29, 0.446], [0.514, 0.446], [0.514, 0.524], [0.29, 0.524]]
    dob_value = [[0.529, 0.751], [0.748, 0.751], [0.748, 0.803], [0.529, 0.803]]
    doi_value = [[0.285, 0.857], [0.48, 0.857], [0.48, 0.908], [0.285, 0.908]]
    doe_value = [[0.531, 0.859], [0.726, 0.859], [0.726, 0.911], [0.531, 0.911]]
    id_card = [[0.285, 0.742], [0.658, 0.742], [0.658, 0.791], [0.285, 0.791]]

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

    max_confidence = {}  # Dictionary to store max confidence for each field
    for key, value in output_dict.items():
        output_dict = get_value(value, normalize_output)
        if output_dict:  # Check if output_dict is not empty
            answer, confidence = min(output_dict.items(), key=lambda x: x[1][0])  # Get the answer with the minimum distance
            answer = validate_and_correct_field(key, answer)
            dict_data[key] = (answer, confidence[1])  # Store answer and confidence score
            if key not in max_confidence or confidence[1] > max_confidence[key]:
                max_confidence[key] = confidence[1]  # Update max confidence score for this field
        else:
            dict_data[key] = ("Not found", 0.0)  # Placeholder if no information is found
            max_confidence[key] = 0.0  # Set max confidence to 0 if no information found

    return dict_data, max_confidence

def write_to_csv(csv_filename, data):
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(data)

def live_id_card_detection():
    # Open camera
    cap = cv2.VideoCapture(0)

    # Open CSV file for writing
    csv_filename = 'extracted_data_live.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Name', 'Father Name', 'Date of Birth', 'Date of Issue', 'Date of Expiry', 'Card Number', 'Confidence'])
        writer.writeheader()

        processed_cards = set()  # Set to store processed card IDs
        accurate_extraction = False  # Flag to indicate accurate extraction for the current card
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Detect ID cards using YOLOv8
            id_cards = id_card_detector(frame)[0]

            for id_card in id_cards.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = id_card

                # Generate a unique ID for the card based on its position and size
                card_id = (x1, y1, x2, y2)

                if card_id not in processed_cards:
                    # Draw bounding box around the ID card
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Crop the ID card region
                    id_card_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                    # Extract information from the ID card
                    extracted_data, max_confidence = extract_cnic_information(id_card_crop)

                    # Display the extracted information
                    print(extracted_data)

                    # Check if confidence is above threshold
                    if max(max_confidence.values()) > 0.8:
                        # Check if accurate information has already been extracted for the current card
                        if not accurate_extraction:
                            # Write extracted data to CSV
                            writer.writerow({key: value[0] for key, value in extracted_data.items()})
                            accurate_extraction = True  # Set flag to indicate accurate extraction
                            processed_cards.add(card_id)  # Add the processed card ID to the set
                            break  # Stop processing this card and move to the next one

            # Reset the flag if no card is detected in the frame
            if len(id_cards) == 0:
                accurate_extraction = False
            
            # Display the resulting frame
            cv2.imshow('frame', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_id_card_detection()