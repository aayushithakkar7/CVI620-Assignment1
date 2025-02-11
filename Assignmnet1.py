import cv2
import numpy as np

# ----------------------- Part I: Capture and Save Images Manually -----------------------
def capture_images():
    # Initialize webcam
    webcam = cv2.VideoCapture(0)
    
    if not webcam.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    # Instructions
    print("Press 's' to capture an image, and 'q' to quit after capturing both images.")
    
    image_count = 0

    while True:
        # Read a frame from the webcam
        ret, frame = webcam.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Display the live video feed
        cv2.imshow('Capture Images (Press s to save, q to quit)', frame)

        # Wait for key input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Capture and save the current frame as an image
            image_count += 1
            filename = f'image{image_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Image {image_count} captured and saved as {filename}")

            if image_count == 2:
                print("Both images captured successfully.")
                break  # Exit the loop after capturing two images

        elif key == ord('q'):
            print("Exiting without capturing further images.")
            break

    # Release the webcam and close windows
    webcam.release()
    cv2.destroyAllWindows()

# Call the capture function to manually take two images
capture_images()

# ----------------------- Part II: Image Arithmetic -----------------------

# Load the saved images
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# Check if the images loaded successfully
def check_image(image, image_name):
    if image is None:
        print(f"Error: Could not load {image_name}. Check the file path.")
        exit()

check_image(image1, 'image1.jpg')
check_image(image2, 'image2.jpg')

# ----- Step 1: Brightness Adjustment -----
bright_image1 = cv2.add(image1, np.array([100.0]))
cv2.imshow('Brightened Image', bright_image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ----- Step 2: Contrast Adjustment -----
contrast_image1 = cv2.convertScaleAbs(image1, alpha=1.5, beta=0)
cv2.imshow('Contrast Image', contrast_image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ----- Step 3: Linear Blending -----
# Resize image2 to match the dimensions of image1
image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# Ask for blending factor
alpha = float(input("Enter a blending factor (alpha) between 0 and 1: "))
blended_image = cv2.addWeighted(image1, 1 - alpha, image2_resized, alpha, 0)
cv2.imshow('Blended Image', blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save Part II results
cv2.imwrite('bright_image1.jpg', bright_image1)
cv2.imwrite('contrast_image1.jpg', contrast_image1)
cv2.imwrite('blended_image.jpg', blended_image)

# ----------------------- Part III: Drawing Application -----------------------

# ----- 1.1 Draw a rectangle on image1 -----
rect_image = cv2.rectangle(image1.copy(), (50, 50), (300, 300), (0, 255, 0), 4)
cv2.imshow('Rectangle with Thickness 4', rect_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ----- 1.2 Draw a filled rectangle on image1 -----
filled_rect_image = cv2.rectangle(image1.copy(), (50, 50), (300, 300), (0, 255, 0), -1)
cv2.imshow('Filled Rectangle', filled_rect_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ----- 1.3 Add text inside the filled rectangle -----
text_image = cv2.putText(filled_rect_image, 'Sample Text', (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imshow('Rectangle with Text', text_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save Part III results
cv2.imwrite('rect_image.jpg', rect_image)
cv2.imwrite('filled_rect_image.jpg', filled_rect_image)
cv2.imwrite('text_image.jpg', text_image)

print("All parts completed successfully.")
