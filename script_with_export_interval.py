from scipy.spatial import distance

import cv2
from ultralytics import YOLO
from ultralytics import solutions
import xml.etree.ElementTree as ET
import os
import math
import numpy as np
import json

from model_data import prepare_csv

LOGGING_STARTED=False
vehicle_present = False
skip_frame = False
output_folder = "frames"

# Function to generate nod.xml
def generate_nod_file(output_file):
    root = ET.Element("nodes")
    ET.SubElement(root, "node", id="center", x="0", y="0", type="traffic_light") # Center
    ET.SubElement(root, "node", id="n1", x="0", y="100", type="priority") # North
    ET.SubElement(root, "node", id="n2", x="100", y="0", type="priority") # East
    ET.SubElement(root, "node", id="n3", x="0", y="-100", type="priority") # South
    ET.SubElement(root, "node", id="n4", x="-100", y="0", type="priority") # West
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

# Function to generate edg.xml
def generate_edg_file(output_file):
    root = ET.Element("edges")
    
    # Define edges using a dictionary for attributes
    ET.SubElement(root, "edge", {
        "from": "n1",
        "to": "center",
        "id": "north_to_center",
        "type": "3L45"
    })  # North to Center
    
    ET.SubElement(root, "edge", {
        "from": "center",
        "to": "n1",
        "id": "center_to_north",
        "type": "2L45"
    })  # Center to North

    ET.SubElement(root, "edge", {
        "from": "n2",
        "to": "center",
        "id": "east_to_center",
        "type": "3L45"
    })  # East to Center
    
    ET.SubElement(root, "edge", {
        "from": "center",
        "to": "n2",
        "id": "center_to_east",
        "type": "2L45"
    })  # Center to East

    ET.SubElement(root, "edge", {
        "from": "n3",
        "to": "center",
        "id": "south_to_center",
        "type": "4L45"
    })  # South to Center
    
    ET.SubElement(root, "edge", {
        "from": "center",
        "to": "n3",
        "id": "center_to_south",
        "type": "2L45"
    })  # Center to South

    ET.SubElement(root, "edge", {
        "from": "n4",
        "to": "center",
        "id": "west_to_center",
        "type": "4L45"
    })  # West to Center
    
    ET.SubElement(root, "edge", {
        "from": "center",
        "to": "n4",
        "id": "center_to_west",
        "type": "2L45"
    })  # Center to West

    
    # Generate the XML tree
    tree = ET.ElementTree(root)
    
    # Write to an XML file
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    
# Function to generate type.xml
def generate_type_file(output_file):
    root = ET.Element("types")
    ET.SubElement(root, "type", id="3L45", priority="3", numLanes="3", speed="45")
    ET.SubElement(root, "type", id="2L45", priority="2", numLanes="2", speed="45")
    ET.SubElement(root, "type", id="4L45", priority="2", numLanes="4", speed="45")
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

# Function to generate rou.xml
def generate_route_file(vehicle_tracks, output_file, entry_exit_mapping, vTypes, vehicle_speeds, fps):
    """
    Generate Rou.XML file for SUMO using the vehicle tracking dara.

    :param valid_vehicle_tracks: Dict of detected valid vehicles
    :param output_file: Path for the output file
    :param route_mapping: Dict with the entry and exit points for route IDs.
    :param vTypes: Dict of generated vTypes based on it's speed
    :param vehicle_speeds: Dict with avg. vehicle speed
    """
    root = ET.Element("routes")

    # Vehicle Types
    for vType in vTypes.values():
        ET.SubElement(root, "vType", **vType)


    # ET.SubElement(root, "vType", id="car", accel="1.0", decel="5.0", sigma="0.0", length="5", maxSpeed="33.33")
    # ET.SubElement(root, "vType", id="bus", accel="1.0", decel="5.0", sigma="0.0", length="15", maxSpeed="3.33")
    # ET.SubElement(root, "vType", id="truck", accel="1.0", decel="5.0", sigma="0.0", length="10", maxSpeed="20")
    # route = ET.SubElement(root, "route", id="north_to_east", edges="north_to_center center_toeast")  # Route from North->Center->East

    for route_id, data in entry_exit_mapping.items():
        route_egdes = f"{data[0]}_to_center center_to_{data[1]}"
        ET.SubElement(root, "route", id=route_id, edges=route_egdes)  

    for vehicle_id, tracks in vehicle_tracks.items():
        # if len(tracks) < 2:
        #     continue


        speed = vehicle_speeds[vehicle_id]
        cls = tracks["cls"]
        # cls = vehicle_information[vehicle_id][0][4]
        vtype_id = f"vType_{cls}_{vehicle_speeds[vehicle_id]}"

        # Calculate departure time based on the first frame the vehicle appears
        # first_frame = tracks[0][0]
        first_frame = tracks["frame"]
        departure_time = first_frame / fps  #TODO: Actual FPS is 30.12 ....


        entry = tracks["entry"]
        exit = tracks["exit"]
        vehicle_route_id = f"route_{entry}_to_{exit}"

        ET.SubElement(
            root, "vehicle",
            id=f"veh{vehicle_id}",
            # type=tracks[0][4],
            type=vtype_id,
            route=vehicle_route_id,
            depart=f"{departure_time:.2f}"  
        )

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def define_vehicle_types(vehicle_speeds, vehicle_information):
    """
    Generate the vehicle types needed for SUMO based on the speed of the vehicles detected.

    :param vehicle_speeds Dict containing speed of the vehicle detected
    :return Dict of vehicletype with their respective config.
    """

    
    v_type_lengths = {
        "car":4.5,
        "bus": 13,
        "truck": 8,
    }

    v_types = {}

    # for spped in set(vehicle_speeds.values()):
    #     print("TYPE .... ", vehicle_information[veh])
    #     v_types[f"vType_{spped}"] = {
    #         "id":f"vType_{spped}", 
    #         "accel":"1.0", 
    #         "decel":"5.0", 
    #         "sigma":"0.0", 
    #         "length":"5", 
    #         "maxSpeed":str(spped / 3.6),
    #     }
    # return v_types

    for vehicle_id, spped in vehicle_speeds.items():
        cls = vehicle_information[vehicle_id][0][4]
        length = v_type_lengths[cls]

        v_types[f"vType_{cls}_{spped}"] = {
            "id":f"vType_{cls}_{spped}", 
            "accel":"1.0", 
            "decel":"5.0", 
            "sigma":"0.0", 
            "length":str(length), 
            "maxSpeed":str(spped / 3.6),
        }
    return v_types

# Function to generate sumo_config.sumocfg
def generate_config_file(output_file):
    root = ET.Element("configuration")
    input_el = ET.SubElement(root, "input")
    ET.SubElement(input_el, "net-file", value="simple_nw_se.net.xml")
    # ET.SubElement(input_el, "net-file", value="test_nw.net.xml")
    ET.SubElement(input_el, "route-files", value="route.rou.xml")
    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", value="0") 
    ET.SubElement(time_el, "end", value="1000") #TODO: Need to figure this out.
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)


def draw_polygonal_region(frame, regions, alpha=0.2):
    """
    Draw region of interest (polygons) for better understanding and analyzing.
    
    :param frame: Current frame to draw on. 
    :param regions: Dict of defined region
    :param alpha: Alpha value for transparency (0.0 fully transparent, 1.0 fully opaque)
    """

    for region, data in regions.items():
        points = np.array(data["points"], dtype=np.int32)
        color = data["color"]
        
        color = tuple(color)

        light_color = tuple(min(c + 50, 255) for c in color)

        temp_frame = frame.copy()

        cv2.fillPoly(temp_frame, [points], light_color)
        cv2.addWeighted(temp_frame, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)

        # Draw text in the center of the polygon
        cx, cy = np.mean(points, axis=0).astype(int)
        cv2.putText(
            frame,
            region.upper(),
            (cx - 50, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, 
            color, 
            2
        )

        
def draw_line_region(frame, regions):
    for region, data in regions.items():
        if len(data["points"]) >= 2:  
            points = np.array(data["points"], dtype=np.int32)
            color = data["color"]

            if data["Start"]:
                text = "START"
            else: 
                text = "END"
            
            cv2.line(frame, tuple(points[0]), tuple(points[1]), color, thickness=2)
            
            cx, cy = np.mean(points, axis=0).astype(int)
            cv2.putText(
                frame,
                text,
                (cx - 50, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, 
                color, 
                2
            )

def detect_region(cx,cy,regions):
    """
    Detect the region (north, south, etc.) based on the given point (cx,cy). 

    :param cx: center x-coordinate of the object. 
    :param cy: center y-coordinate of the object. 
    :param region: Defined regions:
    :return: Region
    """
    detected_region = None

    for region, data in regions.items():
        points = np.array(data["points"], dtype=np.int32)
        is_inside = cv2.pointPolygonTest(points, (cx, cy), False)

        if is_inside >= 0:
            detected_region = region
            break

    return detected_region
    

already_tracked = []
skipped_frames = []
# Function to keep track of the traffic light states, figure out red lights and green light based on the vehicle movement, position, and speed.
def track_traffic_light_states(frame_count, track_data, traffic_light_zones, light_durations, fps, detected_vehicles):
    global vehicle_present, LOGGING_STARTED, skip_frame
    
    current_time = frame_count / fps
    min_duration = 25
    lost_vehicle_threshold = 2000  

    for region, data in traffic_light_zones.items():
        tracked_vehicle_id = data.get("tracked_vehicle_id", None)
        
        if tracked_vehicle_id is None:
            # If no vehicle is being tracked, search for a new vehicle
            for vehicle_id, tracks in track_data.items():
                if vehicle_id not in detected_vehicles:
                    continue

                last_track = tracks[-1]
                cx, cy, speed = last_track[1], last_track[2], last_track[3]

                is_in_zone = detect_region(cx, cy, traffic_light_zones)
                
                if is_in_zone and speed <= 3:  # Vehicle is stopped in the zone
                    LOGGING_STARTED = True
                    vehicle_present = True
                    data["tracked_vehicle_id"] = vehicle_id
                    break
                else:
                    vehicle_present = False
        else:
            # A vehicle is already tracked
            for vehicle_id, tracks in track_data.items():
                if vehicle_id == tracked_vehicle_id:
                    last_track = tracks[-1]
                    cx, cy, speed = last_track[1], last_track[2], last_track[3]

                    is_in_zone = detect_region(cx, cy, traffic_light_zones)

                    # Check if vehicle is moving fast and not in the zone
                    if speed > 8 and not is_in_zone:
                        vehicle_present = False
                        data['tracked_vehicle_id'] = None
                    elif speed == 0 and not is_in_zone:
                        vehicle_present = False
                        data['tracked_vehicle_id'] = None
                    else:
                        if is_in_zone:
                            vehicle_present = True
                        else:
                            vehicle_present = False

                    break

            if tracked_vehicle_id not in detected_vehicles:
                vehicle_present = False
                data['tracked_vehicle_id'] = None
                data['lost_frames'] = data.get('lost_frames', 0) + 1
                continue
        
            if not vehicle_present:
                data['lost_frames'] = data.get('lost_frames', 0) + 1

                if data['lost_frames'] > lost_vehicle_threshold:
                    data['tracked_vehicle_id'] = None
                    vehicle_present = False

   
        if vehicle_present:
            current_state = "red"
            skip_frame = True
        else: 
            current_state = "green"
            skip_frame = False

        if not LOGGING_STARTED and current_state == "green":
            continue

        LOGGING_STARTED = True

        print(f"Vehicle {vehicle_id}, State: {current_state}")

        if len(light_durations) == 0:
            light_durations.append({"region": region, "state": current_state, "start": current_time, "end": None, "duration": None})
        elif light_durations[-1]["state"] != current_state:
            last_entry = light_durations[-1]
            last_entry_duration = current_time - last_entry["start"]

            if last_entry_duration < min_duration:
                continue

            last_entry["end"] = current_time
            last_entry["duration"] = current_time - last_entry["start"]

            light_durations.append({"region": region, "state": current_state, "start": current_time, "end": None, "duration": None})

        return skip_frame

# Function to process video and track vehicles
# km/hr
confusion_data = {}
def process_video(video_path, conf_threshold=0.7):
    down = {}
    up = {}
    global skipped_frames 
    is_skip_frame = False
    """
    Processes a video to detect vehicles, track them (id), and determine their speed.

    :param video_path: Path to the input video file.
    :param conf_threshold: Confidence threshold for detections (probability).
    :return: Dictionary containing track data for each vehicle.
    """

    model = YOLO("yolo.pt")
    model.verbose = False
    classes = model.names

    

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    regions = {
        "north": {"points": [(width // 3, height // 10 - 50), (width // 2 + 100, height // 10 - 50) , (width // 2 + 100, height // 5), (width // 2 - 200, height // 3)], "color": (255,0,0)}, # Blue
        "east": {"points": [(width // 2 + 350,  height // 5 - 50), (width // 2 + 450, height // 5 ) , (width // 2 + 450, height // 3 + 150) , (width // 2 + 100, height // 3 - 100)], "color": (210, 52, 235)}, # Green
        "south": {"points": [(width // 2 ,  height - 100 ), (width - 40 , height // 2 - 50) , (width , height) , (width // 2 - 100, height)], "color": (0,0,255)}, # Red
        "west": {"points": [(0,  height // 2 ), (width // 4 + 130,  height // 2 - 120) , (width // 2 + 30, height - 50) , (0, height)], "color": (255,255,0)}, # Cyan
    }

    traffic_light_zones = {
        # "west":{"points":[(width // 3 - 200 ,  height // 2 + 50 ), (width // 4 + 100,  height // 2 + 40) , (width // 4 + 100 , height - 170) , (width // 3 - 200, height - 140)], "color": (123,255,255)}, 
        # "north":{"points":[(width // 3  ,  height //  10  + 50), (width // 3 + 100,  height // 10 + 30) , (width // 2 - 50 , height // 5 + 50) , (width // 2  - 200, height // 3 )], "color": (123,255,255)}, 
        "east": {"points": [(width // 2 + 200,  height // 5 - 10), (width // 2 + 350, height // 5 ) , (width // 2 + 250, height // 3 - 50 ) , (width // 2 + 100, height // 3 - 100)], "color": (29, 12, 35)},
    }

    frame_count_zone = {
        "north": {"points": [(400, 100), (800, 100)], "color": (223, 162, 224), "Start":True},  
        "south": {"points": [(600, 550), (width, 350)], "color": (179, 133, 109), "Start": False},  
    }

    light_durations = []


    # Initialize tracking and speed estimation variables
    track_data = {}  # vehicle_id: [(frame, cx, cy, speed, label, entry, exit), ...]
    frame_count = 0
    past_positions={}

    time_count = 0

    time_for_crop_video = 0

    start_time_sec = 0
    export_interval = 120
    video_index = 1
    
    # Process video frames
    for results in model.track(source=video_path, conf=conf_threshold, show=False, stream=True, verbose=False):
        frame_count += 1
        frame = results.orig_img.copy()
        tracked_ids = set()

        time_count += 1

        time_for_crop_video = frame_count/fps

        if time_for_crop_video - start_time_sec >= export_interval:
            print(f"Saving XML Files at {time_for_crop_video} seconds")
            generate_xml_files(track_data,time_for_crop_video, fps, video_index)

            track_data.clear()
            past_positions.clear()
            down.clear()
            up.clear()
            frame_count = 0
            video_index += 1

            # Reset YOLO IDs by restarting tracking
            model = YOLO("best_1.pt")
            # start_time_sec = time_for_crop_video
            
        else:
            # Draw rectangular box around each region
            # draw_regions(frame, regions)
            draw_polygonal_region(frame,regions)

            draw_polygonal_region(frame,traffic_light_zones)

            draw_line_region(frame, frame_count_zone)


            for box in results.boxes:
                if box.id is None:
                    continue  # Skip untracked boxes

                object_id = int(box.id[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                tracked_ids.add(object_id)

                if cls not in [3, 5, 8]:  
                    continue

                # Box center coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                region = detect_region(cx,cy,regions)


                # Calculate speed
                if object_id not in track_data:
                    speed = 0.0
                    entry_point = region
                    region = None
                else:
                    last_frame, last_cx, last_cy, last_speed, *_ = track_data[object_id][-1]
                    frame_diff = frame_count - last_frame

                    if frame_diff > 0:
                        meters_per_pixel = 0.07   
                        
                        # Euclidean distance to calculate distance between 2 frames of the obj
                        distance_px = math.hypot(cx - last_cx, cy - last_cy)
                        
                        # Pixels to meter conversion
                        distance_m = distance_px * meters_per_pixel
                        
                        # Time difference in seconds
                        time_sec = frame_diff / fps
                        
                        # Convert speed to kilometers per hour
                        speed_m_per_s = distance_m / time_sec
                        speed = speed_m_per_s * 3.6
                    else:
                        speed = last_speed
                    
                    entry_point = None

                # Update tracking data
                # track_data.setdefault(object_id, []).append((frame_count, cx, cy, speed, label, entry_point, region))

                
                # if region is not None and speed >= 5:
                # if speed >= 3:
                #     if object_id in track_data and len(track_data[object_id]) > 1: 
                #         prev_data = track_data[object_id][-1] # Get previous position 
                #         # print(f"Prev Data: {prev_data} {object_id}") 
                #         prev_cx, prev_cy = prev_data[1], prev_data[2] 
                #         prev_region = prev_data[5] 
                #         if len(past_positions[object_id]) > 10:
                #             next_region = predict_next_region(cx, cy, prev_cx, prev_cy, region, regions, speed, past_positions[object_id], object_id) 
                #             # if region != next_region:
                #             print(f"Vehicle ID {object_id} is in {region}, likely moving to {next_region}") 

                #             if object_id not in confusion_data: 
                #                 confusion_data[object_id] = {} 
                #             confusion_data[object_id][frame_count] = {"actual": region, "next": next_region} 

                # if object_id not in past_positions:
                #     past_positions[object_id] = []
                
                # past_positions[object_id].append((cx, cy))

    
    

                # if object_id not in confusion_data: 
                #     confusion_data[object_id] = {} 
                # confusion_data[object_id][frame_count] = {"actual": region, "next_region": next_region}
                # 
        
                


                track_data.setdefault(object_id, []).append((frame_count, cx, cy, speed, label, entry_point, region)) 

                


                # Get the entry and exit frame count for the vehicle
                
                if cy + 7 > 100 and object_id not in down and region == "north": 
                    down[object_id] = {"start": time_count, "start_frame": frame_count}
                    # print(f"Vehicle id {object_id} entered the zone at frame {frame_count}.")
                if object_id in down:
                    # print(f"Vehicle id {object_id}. Value of cy: {cy}")
                    if 500 < cy + 8 :
                        down[object_id]["end_frame"] = frame_count
                        down[object_id]["end"] = time_count
                        # print(f"Vehicle id {object_id} exited the zone at frame {frame_count}.")


                # print(f"Vehicle id {object_id} is in region {region} with cx and cy: {cx}, {cy}))")
                if (cy + 7 > 500 and 600 <= cx <= width) and object_id not in up and region == "south":
                    up[object_id] = {"start": time_count, "start_frame": frame_count}
                    # print(f"Vehicle id {object_id} entered the zone at frame {frame_count}.")
                if object_id in up:
                    if  cy + 8 < 100:
                        up[object_id]["end_frame"] = frame_count
                        up[object_id]["end"] = time_count
                        # print(f"Vehicle id {object_id} exited the zone at frame {frame_count}.")
                
           

                # Draw tracking data# Draw bounding box and annotations
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID:{object_id} {label} {speed:.2f} km/hr",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # if region == "north":
                #     is_skip_frame = track_traffic_light_states(frame_count,track_data, traffic_light_zones, light_durations,fps, tracked_ids)
                
                # if is_skip_frame:
                #     # Remove the last entry from the track data
                #     skipped_frames.append(frame_count)
                #     remove_data_for_frame(track_data, frame_count)
                # else: 
                #     # Count the skipped frames and decrease the frame count by number of skipped frames
                #     no_of_skipped_frames = len(set(skipped_frames))
                #     if no_of_skipped_frames > 0:
                #         frame_count -= no_of_skipped_frames
                #         skipped_frames = []


                # if region == "east":
                #     is_skip_frame = track_traffic_light_states(frame_count,track_data, traffic_light_zones, light_durations,fps, tracked_ids)
                
                # if is_skip_frame:
                #     # Remove the last entry from the track data
                #     skipped_frames.append(frame_count)
                #     remove_data_for_frame(track_data, frame_count)
                # else: 
                #     # Count the skipped frames and decrease the frame count by number of skipped frames
                #     no_of_skipped_frames = len(set(skipped_frames))
                #     if no_of_skipped_frames > 0:
                #         frame_count -= no_of_skipped_frames
                #         skipped_frames = []

                

        
        data_to_save = {
            "skipped_frames": skipped_frames,
            "track_data": track_data
        }

        # Write in tracking log json file 
        # with open('tracking_log.json', 'w') as json_file:
        #     json.dump(data_to_save, json_file, indent=4)

        # File for confusion matrix
        # write_to_json(confusion_data, "data_for_confusion_matrix")


        # Save each frame 
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")

        cv2.imwrite(frame_filename, frame)
        cv2.imshow("Vehicle Detection and Speed Estimation", frame)
        # cv2.waitKey(100)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    calculate_average_time(up, down)

    return track_data, fps

def write_to_json(data_to_save, filename):
    with open(filename, 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)

def find_angle_distance(points, vehicle_id):
    """
    Find the angle of turn based on the movement trajectory of the vehicle.
    """
    # We consider the last 20 points and calculate the distance
    if len(points) < 4:  # If there aren't enough points, return 'stopped'
        print(f"Vehicle id {vehicle_id} stopped")
        return "stopped"


    # Calculate the covered distance
    d = calculate_covered_distance(points[-20:])
    if d > 30:
        points = points[-40:]
        size = len(points) // 4
        points = points[::size]
        p1, p2, p3, p4 = points[-4:]

        if calculate_covered_distance([p2, p4]) > 20:
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p4) - np.array(p3)
            unit_v1 = v1 / np.linalg.norm(v1)
            unit_v2 = v2 / np.linalg.norm(v2)
            angle = np.degrees(np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)))
            # Check for turns based on the angle
    
            if 0 <= angle <= 15:
                return "straight"
            elif 15 < angle < 90:
                A, B, C = p1, p2, p4
                diff = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
                if diff > 0:
                    return "right"
                elif diff < 0:
                    return "left"
                else:
                    return "straight"
    else:
        return "stopped"
    
def calculate_covered_distance(points):
    """
    Calculate the total distance covered by the vehicle based on the given points.
    """
    d = 0        
    for i in range(len(points) - 1):
        d += distance.euclidean(points[i], points[i + 1])
    return d

def apply_perspective_correction(point, H):
    corrected_point = cv2.perspectiveTransform(np.array([[point]], dtype="float32"), H)
    return corrected_point[0][0]

def predict_next_region(cx, cy, prev_cx, prev_cy, current_region, regions, speed, points, vehicle_id): 

    # Difference between previous and current distance
    dx = cx - prev_cx 
    dy = cy - prev_cy 
    
    # If speed is 0 (meaning red light), prediction is None
    if speed == 0: 
        return current_region 
    
   
    angle = find_angle_distance(points, vehicle_id)


    if angle == "stopped" and speed < 1:
        return current_region
    
    if angle == "straight":
        if abs(dy) > abs(dx):
            if current_region == "north" and dy > 0:
                return "south"
            elif current_region == "south" and dy < 0:
                return "north"
        elif abs(dx) > abs(dy):
            if current_region == "west" and dx > 0:
                return "east"
            elif current_region == "east" and dx < 0:
                return "west"
    elif angle == "right":
        return predict_turn_direction(current_region, "right")
    elif angle == "left":
        return predict_turn_direction(current_region, "left")
    else:
        return current_region
        
    if abs(dx) > abs(dy):  
        return "east" if dx > 0 else "west" 
    else:  
        return "south" if dy > 0 else "north" 

def predict_turn_direction(current_region, turn_direction):
                                                            
    if current_region == "north":
        if turn_direction == "left":
            return "east"
        elif turn_direction == "right":
            return "west"
    elif current_region == "south":
        if turn_direction == "left":
            return "west"
        elif turn_direction == "right":
            return "east"
    elif current_region == "east":
        if turn_direction == "left":
            return "south"
        elif turn_direction == "right":
            return "north"
    elif current_region == "west":
        if turn_direction == "left":
            return "north"
        elif turn_direction == "right":
            return "south"
    
    return current_region

def calculate_average_time(up_data, down_data):
    valid_times = []

    # Calculate valid times for up_data
    for vehicle_id, up_item in up_data.items():
        if "start" in up_item and "end" in up_item and up_item["start"] > 0 and up_item["end"] > 0:
            time_taken_frames = up_item["end"] - up_item["start"]
            time_in_sec = time_taken_frames / 30.12
            valid_times.append(time_in_sec)

    # Calculate valid times for down_data
    for vehicle_id, down_item in down_data.items():
        if "start" in down_item and "end" in down_item and down_item["start"] > 0 and down_item["end"] > 0:
            time_taken_frames = down_item["end"] - down_item["start"]
            time_in_sec = time_taken_frames / 30.12
            # print(f"Time taken for vehicle {vehicle_id} (Down) is {time_in_sec} seconds.")
            valid_times.append(time_in_sec)

    # print(f"Valid times: {valid_times}")

    if valid_times:
        average_time = sum(valid_times) / len(valid_times)
        # print(f"Average time: {average_time} seconds for {len(valid_times)} vehicles.")
        return average_time
    else:
        return 0.0


def remove_data_for_frame(track_data, frame_number):
    """
    Removes tracking data for all vehicles at a specific frame number.

    :param track_data: Dictionary containing vehicle tracking data.
    :param frame_number: The frame number for which to remove the tracking data.
    """

    key_to_delete = []

    for vehicle_id, data in track_data.items():
    # Check if the last tuple in the array matches the frame_number
        if data and data[-1][0] == frame_number:
            # Remove the last tuple from the list
            data.pop()
        
        # If the list is empty, remove the vehicle_id from the dictionary
        if len(data) == 0:
            key_to_delete.append(vehicle_id)
    
    if len(key_to_delete) > 0:
        for key in key_to_delete:
            del track_data[key]



def calculate_vehicle_speeds(vehicle_information):
    """
    Calaulate average speed for each vehicle

    :param vehicle_information: Dict information of the detected vehicle
    :return: Dictionary containing vehicle speed
    """

    vechicle_speeds = {}
    for vehicle_id, tracks in vehicle_information.items():
        speeds = [track[3] for track in tracks if track[3] > 0]
        if speeds: 
            avg_speed = sum(speeds) / len(speeds)
            vechicle_speeds[vehicle_id] = avg_speed
    return vechicle_speeds

def filter_valid_tracks(vehicle_information):
    """
    Filter vehicle tracks to ensure they have valid entry and exit points for the routes
    :param vehicle_information: Dict information of the detected vehicle
    :return: Dictionary containing valid tracks
    """

    valid_tracks = {}

    # print(f"Vehicle Information: {vehicle_information}")
    for vehicle_id, tracks in vehicle_information.items():

        entry = tracks[0][5]
        exit = tracks[-1][6]

        if (entry != None and exit != None) and (entry != exit):
            # if (entry == "north" or entry == "south") and (exit == "north" or exit == "south"):
            # if (entry == "east" or entry == "west") and (exit == "east" or exit == "west"):
                valid_tracks[vehicle_id] = {"entry": entry, "exit": exit, "frame":tracks[0][0],"cls":tracks[0][4]}
        # else: 
            # print(f"Vehicle {vehicle_id} disregarded: entry={entry} and exit={exit}")

    return valid_tracks


def main():
    global output_folder   

    # Path to the video
    video_path = "./Data/Bellevue_116th_NE12th__2017-09-11_08-08-50.mp4"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Function call to process the video, returns dict of detected vehicles
    vehicle_tracks, fps = process_video(video_path)

    # prepare_csv.prepare_lstm_dataset(vehicle_tracks)

    vehicle_speeds = calculate_vehicle_speeds(vehicle_tracks)
    valid_vehicle_tracks = filter_valid_tracks(vehicle_tracks)
    vTypes = define_vehicle_types(vehicle_speeds, vehicle_tracks)

    write_to_json(valid_vehicle_tracks, "valid_vehicle_data")

    # Generate SUMO input files
    os.makedirs("sumo_files", exist_ok=True)
    generate_nod_file("sumo_files/nod.xml") # Node file
    generate_edg_file("sumo_files/edg.xml") # Edges file
    generate_type_file("sumo_files/type.xml") # Types file


    entry_exit_mappings = {
        "route_north_to_east" : ["north","east"],
        "route_north_to_south" : ["north","south"],
        "route_north_to_west": ["north","west"],
        "route_east_to_west": ["east","west"],
        "route_east_to_south": ["east","south"],
        "route_east_to_north": ["east","north"],
        "route_south_to_west": ["south","west"],
        "route_south_to_north": ["south","north"],
        "route_south_to_east": ["south","east"],
        "route_west_to_south": ["west","south"],
        "route_west_to_north": ["west","north"],
        "route_west_to_east": ["west","east"],
        }

    generate_route_file(valid_vehicle_tracks, "sumo_files/route.rou.xml",entry_exit_mappings, vTypes,vehicle_speeds, fps) # Routes file
    generate_config_file("sumo_files/sumo_config.sumocfg") # Config File

    # Generate Network File
    os.system("netconvert --node-files sumo_files/nod.xml --edge-files sumo_files/edg.xml --type-files sumo_files/type.xml -o sumo_files/simple_nw_se.net.xml")

    print("SUMO input files generated successfully!")

def generate_xml_files(vehicle_tracks, time, fps, video_index):
    interval = 120
    vehicle_speeds = calculate_vehicle_speeds(vehicle_tracks)
    valid_vehicle_tracks = filter_valid_tracks(vehicle_tracks)
    vTypes = define_vehicle_types(vehicle_speeds, vehicle_tracks)

    
    print(f"Len of acutal vehicles detected... {len(vehicle_tracks)}")
    time = int(time/interval)

    write_to_json(valid_vehicle_tracks, "valid_vehicle_data")

    folder_path = f"sumo_files/Bellevue_116th_NE12th__2017-09-11_14-08-35/2Min/Video_{video_index}"
    # Generate SUMO input files
    os.makedirs(folder_path, exist_ok=True)
    generate_nod_file(f"{folder_path}/nod.xml") # Node file
    generate_edg_file(f"{folder_path}/edg.xml") # Edges file
    generate_type_file(f"{folder_path}/type.xml") # Types file


    entry_exit_mappings = {
        "route_north_to_east" : ["north","east"],
        "route_north_to_south" : ["north","south"],
        "route_north_to_west": ["north","west"],
        "route_east_to_west": ["east","west"],
        "route_east_to_south": ["east","south"],
        "route_east_to_north": ["east","north"],
        "route_south_to_west": ["south","west"],
        "route_south_to_north": ["south","north"],
        "route_south_to_east": ["south","east"],
        "route_west_to_south": ["west","south"],
        "route_west_to_north": ["west","north"],
        "route_west_to_east": ["west","east"],
        }

    generate_route_file(valid_vehicle_tracks, f"{folder_path}/route.rou.xml",entry_exit_mappings, vTypes,vehicle_speeds, fps) # Routes file
    generate_config_file(f"{folder_path}/sumo_config.sumocfg") # Config File

    # Generate Network File
    os.system(f"netconvert --node-files {folder_path}/nod.xml --edge-files {folder_path}/edg.xml --type-files {folder_path}/type.xml -o {folder_path}/simple_nw_se.net.xml")

    print("SUMO input files generated successfully!")

if __name__ == "__main__":
    main()
    # generate_config_file("sumo_files/sumo_config_updated.sumocfg")

