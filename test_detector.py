
from detector import analyse_video

for frame, counts in analyse_video(
    "videos/BLK-HDPTZ12 Security Camera Parkng Lot Surveillance Video.mp4",
    ["People", "Cars"]
):
    print(counts)