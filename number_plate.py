import cv2
import os
import pytesseract
import numpy as np

# Tesseract path configuration for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if not os.path.exists("Plates"):
    os.makedirs("Plates")

def enhance_plate(img_roi):
    # Resize image to a larger size for better OCR
    img_roi = cv2.resize(img_roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp = cv2.filter2D(gray, -1, kernel)
    
    blur = cv2.GaussianBlur(sharp, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    padded = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    return padded

def process_plate(img_roi):
    try:
        processed = enhance_plate(img_roi)
        
        # Multiple OCR attempts with different configurations
        configs = [
            '--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--oem 1 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ]
        
        results = []
        for config in configs:
            text = pytesseract.image_to_string(processed, config=config)
            cleaned = ''.join(c for c in text if c.isalnum())
            if len(cleaned) >= 5:  
                results.append(cleaned)

        if results:
            text = max(results, key=len)
            text = text.replace('O', '0').replace('I', '1').replace('S', '5')

            return text
        
        return ""
        
    except Exception as e:
        print(f"Error in OCR processing: {e}")
        return ""

def detect_plate_color(img_roi):
    blurred_img = cv2.GaussianBlur(img_roi, (5, 5), 0)
    
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    # Define broader color ranges for more sensitivity
    # Color ranges for vehicle types
    colors = {
        "Private Vehicle (White)": ([0, 0, 180], [180, 50, 255]),
        "Commercial Vehicle (Yellow)": ([15, 70, 70], [35, 255, 255]),
        "Electric Vehicle (Green)": ([30, 50, 50], [85, 255, 255]),
        "Government Vehicle (Blue)": ([90, 50, 50], [130, 255, 255]),
        "Rental Vehicle (Red)": ([0, 70, 50], [10, 255, 255]),
        "Rental Vehicle (Red 2)": ([170, 70, 50], [180, 255, 255])
    }

    color_percentages = {}
    total_pixels = img_roi.shape[0] * img_roi.shape[1]

    # Calculate the percentage of pixels for each color
    for color, (lower, upper) in colors.items():
        mask = cv2.inRange(hsv_img, np.array(lower, np.uint8), np.array(upper, np.uint8))
        color_percent = np.sum(mask) / (255 * total_pixels)
        color_percentages[color] = color_percent

    threshold = 0.2  # 20% of pixels need to be of a particular color
    detected_color = "Unknown"
    
    for color, percent in color_percentages.items():
        if percent > threshold and (detected_color == "Unknown" or percent > color_percentages[detected_color]):
            detected_color = color

    return detected_color

def main():
    harcascade = "model/haarcascade_russian_plate_number.xml"
    if not os.path.exists(harcascade):
        print(f"Error: Cascade file not found at {harcascade}")
        return

    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set higher resolution for better image capture
    cap.set(3, 1280)  # width
    cap.set(4, 720)   # height
    min_area = 500
    count = 0

    print("Starting license plate detection...")
    print("Press 's' to save detected plate")
    print("Press 'q' to quit")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break
            
        try:
            plate_cascade = cv2.CascadeClassifier(harcascade)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
            
            for (x, y, w, h) in plates:
                area = w * h
                if area > min_area:
                    # Draw rectangle around plate
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Extract plate region with padding
                    padding = 5
                    y_start = max(y - padding, 0)
                    y_end = min(y + h + padding, img.shape[0])
                    x_start = max(x - padding, 0)
                    x_end = min(x + w + padding, img.shape[1])
                    img_roi = img[y_start:y_end, x_start:x_end]
                    
                    try:
                        # Process plate and get text
                        plate_text = process_plate(img_roi)
                        
                        # Detect plate color
                        plate_type = detect_plate_color(img_roi)
                        
                        # Display the text and plate type above the rectangle
                        if plate_text:
                            display_text = f"{plate_text} | {plate_type}"
                            cv2.putText(img, display_text, (x, max(y - 10, 20)), 
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                        
                        cv2.imshow("License Plate", img_roi)
                        
                        processed_img = enhance_plate(img_roi)
                        cv2.imshow("Processed Plate", processed_img)
                        
                    except Exception as e:
                        print(f"Error processing individual plate: {e}")
                        continue
            
            cv2.imshow("License Plate Detection", img)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Save plate when 's' is pressed
            if key == ord('s'):
                try:
                    save_folder = os.path.join("Plates", f"plate_{count}")
                    os.makedirs(save_folder, exist_ok=True)
                    
                    orig_path = os.path.join(save_folder, "original.jpg")
                    proc_path = os.path.join(save_folder, "processed.jpg")
                    text_path = os.path.join(save_folder, "text.txt")
                    
                    cv2.imwrite(orig_path, img_roi)
                    cv2.imwrite(proc_path, processed_img)
                    
                    with open(text_path, 'w') as f:
                        f.write(f"Detected Text: {plate_text}\n")
                        f.write(f"Detected Plate Type: {plate_type}\n")
                    
                    print(f"Saved plate details to {save_folder}")
                    count += 1
                    
                except Exception as e:
                    print(f"Error saving plate: {e}")

            # Exit when 'q' is pressed
            if key == ord('q'):
                break
        
        except Exception as e:
            print(f"Error in main loop: {e}")
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
