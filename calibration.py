import cv2
from ultralytics import YOLO
import numpy as np

def refine_bounding_box(frame, box):
    """
    Refine bouding box using contours 

    :param frame: Input frame obtained from YOLO
    :param box: Current bounding box
    :return: Rotated bounding box
    """
    x1, y1, x2, y2 = map(int, box)

    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key = cv2.contourArea)

        rect = cv2.minAreaRect(largest_contour)
        box_points = cv2.boxPoints(rect)
        box_points = np.array(box_points, dtype=np.int32)

        box_points[:, 0] += x1
        box_points[:, 1] += y1

        return box_points
    return None

def visualize_rotated_bounding_box(frame, box_points):
    if box_points is not None: 
        cv2.drawContours(frame, [box_points], -1, (0,255,0), 2)

def calibrate_meters_per_pixel(video_path, conf_threshold):
    """
    Calibrate a known bus

    :param video path to laod and detect the bus
    :return: Meters per pixel
    """

    # Real world width and length of the bus
    bus_width_m = 2.6
    bus_length_m = 13

    frame_count = 0


    model = YOLO("best_1.pt") 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file {video_path}")
    
    for results in model.track(source=video_path, conf=conf_threshold, show=False, stream=True):
        frame_count += 1
        frame = results.orig_img.copy()

        for box in results.boxes:
            if box.id is None:
                continue  # Skip untracked boxes

            object_id = int(box.id[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Car, bus, truck
            if cls != 8:  
                continue


            # Box center coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            refined_box = refine_bounding_box(frame, [x1, y1, x2, y2])

            if refined_box is not None:
                visualize_rotated_bounding_box(frame, refined_box)


            # # Calculate the bus width and length in pixels
            # bus_width_px = x2 - x1
            bus_length_px = y2 - y1


            # if bus_length_px == 0 or bus_width_px == 0:
            #     continue

            # if bus_length_px < bus_width_px:
            #     bus_width_px, bus_length_px = bus_length_px, bus_width_px


            # # Calculate the meter per pixel for width and length
            # mpp_width = bus_width_m / bus_width_px
            mpp_length = bus_length_m / bus_length_px

            print(f"Mpp length.. {mpp_length}")

            # Draw bounding box and annotations
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(
            #     frame,
            #     f"ID:{object_id} {label}",
            #     # f"ID: {object_id}, Region: {region}",
            #     (x1, y1 - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 255, 0),
            #     2
            # )
        cv2.imshow("Vehicle Detection and Speed Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return mpp_length





def main():
    video_path = './Data/cropped_video.mp4'

    mpp = calibrate_meters_per_pixel(video_path, conf_threshold=0.6)

    print("Meter per pixel found...")



if __name__ == "__main__":
    main()
