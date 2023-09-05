import time
from pathlib import Path
import cv2
import argparse
import numpy as np
from threading import Thread
from djitellopy import Tello
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch


# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cache_directory = Path(__file__).parent.parent / "cache"
cache_directory.mkdir(parents=True, exist_ok=True)


COCO_LABELS = {
    0: 'N/A', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    # ... more labels can be added as needed
}

model = None      # TODO: don't do this
processor = None  # TODO: don't do this

# Initialize object detection model from Hugging Face
# processor = DetrImageProcessor.from_pretrained(args.model_name, cache_dir=cache_directory)
# model = DetrForObjectDetection.from_pretrained(args.model_name, cache_dir=cache_directory)
# model.to(device)


# def detect_objects(model, processor, frame) -> None:
#     """Detect objects in a frame."""
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     inputs = processor(frame, return_tensors="pt").to(device)
#     # model.to(device)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Extract results
#     target_sizes = torch.tensor([frame.shape[-2:]])
#     results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.4)[0]

#     # Extract bounding boxes
#     boxes = results["boxes"].cpu().detach().numpy()
#     labels = results["labels"].cpu().detach().numpy()
#     scores = results["scores"].cpu().detach().numpy()

#     # Draw bounding boxes and labels on the frame
#     for box, label, score in zip(boxes, labels, scores):
#         label_str = COCO_LABELS.get(label, "unknown")
#         cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
#         cv2.putText(frame, f"{label_str} {score:.2f}", (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     return frame


def video_recorder(drone_object, keep_recording: bool) -> None:
    """Record video from the drone and write to disk."""
    frame_read = drone_object.get_frame_read()

    height, width, _ = frame_read.frame.shape
    video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    while keep_recording:
        # current_time = time.time()
        # elapsed_time = current_time - start_time
        # if elapsed_time > recording_time:
        #     keep_recording = False
        #     break
        frame = frame_read.frame
        print(frame)
        # frame = detect_objects(model, processor, frame)

        video.write(frame)
        time.sleep(1 / 30)

    video.release()


def main():
    parser = argparse.ArgumentParser(description='Tello Drone Object Detection')
    parser.add_argument('--model_name', type=str, default="facebook/detr-resnet-50",
                        help='The name of the object detection model to use from Hugging Face')
    args = parser.parse_args()

    tello = Tello()
    tello.connect(wait_for_state=False)  # BUG: https://github.com/damiafuentes/DJITelloPy/issues/71#issuecomment-769790211

    tello.streamon()
    time.sleep(3)

    keep_recording = True
    recording_time = 15  # seconds
    start_time = time.time()  # Capture the start time
    recorder = Thread(target=video_recorder, args=(tello, keep_recording))
    recorder.start()

    # while keep_recording:
    #     # current_time = time.time()
    #     # elapsed_time = current_time - start_time
    #     # if elapsed_time > recording_time:
    #     #     keep_recording = False
    #     #     break

    tello.send_control_command("takeoff", timeout=Tello.TAKEOFF_TIMEOUT)
    # tello.takeoff()

    # tello.rotate_counter_clockwise(360)

    tello.land()
    time.sleep(10)

    tello.streamoff()
    recorder.join()


if __name__ == "__main__":
    main()
