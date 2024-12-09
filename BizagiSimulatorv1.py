from PIL import Image
import pytesseract
import cv2
import numpy as np

def analyze_bpmn_diagram(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale for shape detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect contours (shapes)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Recognize text using Tesseract OCR
    ocr_data = pytesseract.image_to_data(Image.open(image_path), output_type=pytesseract.Output.DICT)

    # Map shapes and text to elements
    elements = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(contour)

        # Extract text within the bounding box of the shape
        associated_text = []
        for i in range(len(ocr_data['text'])):
            text_x, text_y, text_w, text_h = (ocr_data['left'][i], ocr_data['top'][i], 
                                              ocr_data['width'][i], ocr_data['height'][i])
            if (text_x >= x and text_x + text_w <= x + w) and (text_y >= y and text_y + text_h <= y + h):
                associated_text.append(ocr_data['text'][i])

        # Determine element type based on shape
        element_type = "Unknown"
        if len(approx) == 4:  # Rectangle (e.g., task)
            element_type = "Task/Activity"
        elif len(approx) > 4:  # Circle or oval (e.g., event)
            element_type = "Event"
        elif len(approx) == 3:  # Diamond (e.g., gateway)
            element_type = "Gateway"

        # Add element details to the list
        if associated_text:
            elements.append({
                "Type": element_type,
                "Text": " ".join(associated_text).strip()
            })

    # Print results
    print("\nIdentified BPMN Elements:")
    for element in elements:
        print(f"Type: {element['Type']}, Text: {element['Text']}")

# Example usage
image_path = "C:/Test Data/Bizagi/5.5.13 Real Property-Monthly_Reviews.tiff"  # Replace with the path to your BPMN diagram
analyze_bpmn_diagram(image_path)
