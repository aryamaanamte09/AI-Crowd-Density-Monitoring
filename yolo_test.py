from ultralytics import YOLO
import cv2

#first, we load the pretrained YOLO model
#i am using the nano version because it is faster to run and performs faster on this laptop
model = YOLO("yolo26n.pt")

#this is a test image that I have set the yolo to analyze
image_path = "bus.jpg"

#opening the image using OpenCV
image = cv2.imread(image_path)

#run the same image through yolo
#verbose=False only to keep output clean
results = model(image, verbose=False)

#checking all YOLO detections loop
for result in results:

    for box in result.boxes:

        #get the class number and convert it to the actual class name -> understand what is being identified
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        #set a confidence score to analyze how sure the model is about the identification
        confidence = float(box.conf[0])

        #collect boundfing box coords
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        #drawing the bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        #display detected class and confidence above the box
        cv2.putText(
            image,
            f"{class_name} {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

#show the final image with detections
cv2.imshow("YOLO Detection", image)

#wait until a key is pressed before closing window, more application feel
cv2.waitKey(0)
cv2.destroyAllWindows()