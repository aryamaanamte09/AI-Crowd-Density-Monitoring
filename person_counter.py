import cv2
import time
import psutil
from ultralytics import YOLO

#this lets the python program send the density result to the simulated arduino
from arduino_simulator import send_to_arduino

#load the pretrained yolo model
model = YOLO("yolo26n.pt")

#model architecture and computational requirements
model.info()

#this gets information about the current python process for memory measurement
process = psutil.Process()

#location of the video that will be used as the camera input is specified (do not change!)
video_path = "crowd.mp4"

#opens the video with opencv
import os
import cv2

video_path = os.path.join(
    "videos",
    "BLK-HDPTZ12 Security Camera Parkng Lot Surveillance Video (online-video-cutter.com).mp4"
)

cap = cv2.VideoCapture(video_path)

print("Current working directory:", os.getcwd())
print("Video path:", video_path)
print("File exists:", os.path.exists(video_path))

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: OpenCV could not open the video.")
    exit()

print("Video opened successfully!")

#check whether opencv could open the video (check format for specific mp4 or naming conflict)
if not cap.isOpened():
    print("ERROR: OpenCV could not open the video.")
    exit()

print("Video opened successfully!")

#this stores the last density level sent to the arduino
#we use it so that the same command is not sent repeatedly for every frame, makes terminal cleaner
previous_density = ""

#these variables are used to calculate the average inference time
total_inference_time = 0
total_processing_time = 0
frame_count = 0

#this stores the highest memory usage recorded during the test
peak_memory = 0


while True:

    #start measuring the total processing time for this frame
    frame_start_time = time.perf_counter()

    #read one frame from the video
    ret, frame = cap.read()

    #if no more frames, stop the loop
    if not ret:
        print("Video ended or frame could not be read.")
        break

    # measure how long yolo takes to process one frame

    # measure how long yolo takes to process one frame
    start_time = time.perf_counter()

    results = model(
        frame,
        classes=[0, 2, 7],
        conf=0.15,
        imgsz=640,
        verbose=False
    )

    end_time = time.perf_counter()
    inference_time = end_time - start_time

    # add the current inference time to the total
    total_inference_time += inference_time

    #start counting each object class in this frame
    person_count = 0
    car_count = 0
    truck_count = 0

    #go through the detection results returned by yolo
    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            #get the coordinates of the object's bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            #count the detected object and choose its box colour
            if class_name == "person":
                person_count += 1
                box_color = (0, 255, 0)

            elif class_name == "car":
                car_count += 1
                box_color = (255, 150, 0)

            elif class_name == "truck":
                truck_count += 1
                box_color = (255, 0, 255)

            else:
                continue

            #draw a rectangle around the detected object
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            #display the object name and confidence
            label = f"{class_name}: {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

    #calculate density using the number of people detected
    #current limits are experimental
    if person_count <= 5:
        density = "LOW"
        density_color = (0, 255, 0)

    elif person_count <= 15:
        density = "MEDIUM"
        density_color = (0, 255, 255)

    elif person_count <= 30:
        density = "HIGH"
        density_color = (0, 165, 255)

    else:
        density = "CRITICAL"
        density_color = (0, 0, 255)

    #only send a new command when the density level changes
    if density != previous_density:
        send_to_arduino(density)

        #save the current density for comparison with the next frame
        previous_density = density

    #display the number of people detected
    cv2.putText(
        frame,
        f"People: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    #display the number of cars detected
    cv2.putText(
        frame,
        f"Cars: {car_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    #display the number of trucks detected
    cv2.putText(
        frame,
        f"Trucks: {truck_count}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    #display the current crowd-density level
    cv2.putText(
        frame,
        f"Density: {density}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        density_color,
        2
    )


    #calculate the total time taken to process this frame
    frame_end_time = time.perf_counter()
    processing_time = frame_end_time - frame_start_time
    total_processing_time += processing_time

    #count how many frames have been processed
    frame_count += 1

    #check the current memory usage of the python process
    current_memory = process.memory_info().rss

    #keep track of the highest memory usage during the test
    if current_memory > peak_memory:
        peak_memory = current_memory

    #show the processed frame
    cv2.imshow("Crowd Density Monitor", frame)

    #press q to stop the video
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


#release the video file after the loop finishes
cap.release()

#close all opencv windows
cv2.destroyAllWindows()

#calculate and display the average inference time
if frame_count > 0:

    #calculate the average time taken by yolo for one frame
    average_inference_time = total_inference_time / frame_count

    #average inference time turns into frames per second
    average_fps = 1 / average_inference_time

    #calculate the average time taken to process one complete frame
    average_processing_time = total_processing_time / frame_count

    #average processing time turns into overall processing frames per second
    processing_fps = 1 / average_processing_time

    #convert the peak memory usage from bytes to megabytes
    peak_memory_mb = peak_memory / (1024 * 1024)

    print("PERFORMANCE RESULTS")
    print(f"Frames processed: {frame_count}")
    print(f"Average inference time: {average_inference_time * 1000:.2f} ms")
    print(f"Average inference FPS: {average_fps:.2f}")
    print(f"Average processing latency: {average_processing_time * 1000:.2f} ms")
    print(f"Overall processing FPS: {processing_fps:.2f}")
    print(f"Peak process memory: {peak_memory_mb:.2f} MB")

print("Program finished.")