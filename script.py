import cv2
from ultralytics import YOLO
from ultralytics import solutions
import xml.etree.ElementTree as ET
import os
import math

# Function to generate nod.xml
def generate_nod_file(output_file):
    root = ET.Element("nodes")
    ET.SubElement(root, "node", id="center", x="0", y="0", type="priority") # Center
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
        "type": "3L45"
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
        "type": "3L45"
    })  # Center to East

    ET.SubElement(root, "edge", {
        "from": "n3",
        "to": "center",
        "id": "south_to_center",
        "type": "3L45"
    })  # South to Center
    
    ET.SubElement(root, "edge", {
        "from": "center",
        "to": "n3",
        "id": "center_to_south",
        "type": "3L45"
    })  # Center to South

    ET.SubElement(root, "edge", {
        "from": "n4",
        "to": "center",
        "id": "west_to_center",
        "type": "3L45"
    })  # West to Center
    
    ET.SubElement(root, "edge", {
        "from": "center",
        "to": "n4",
        "id": "center_to_west",
        "type": "3L45"
    })  # Center to West

    
    # Generate the XML tree
    tree = ET.ElementTree(root)
    
    # Write to an XML file
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    



# Function to generate type.xml
def generate_type_file(output_file):
    root = ET.Element("types")
    ET.SubElement(root, "type", id="3L45", priority="3", numLanes="3", speed="45")
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

# Function to generate rou.xml
def generate_route_file(vehicle_tracks, output_file, entry_exit_mapping):
    """
    Generate Rou.XML file for SUMO using the vehicle tracking dara.

    :param vehicle_tracks: Dict of detected vehicles
    :param output_file: Path for the output file
    :param route_mapping: Dict with the entry and exit points for route IDs.
    """
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="car", accel="1.0", decel="5.0", sigma="0.0", length="5", maxSpeed="33.33")
    ET.SubElement(root, "vType", id="bus", accel="1.0", decel="5.0", sigma="0.0", length="15", maxSpeed="3.33")
    ET.SubElement(root, "vType", id="truck", accel="1.0", decel="5.0", sigma="0.0", length="10", maxSpeed="20")
    # route = ET.SubElement(root, "route", id="north_to_east", edges="north_to_center center_toeast")  # Route from North->Center->East

    for route_id, data in entry_exit_mapping.items():
        route_egdes = f"{data[0]}_to_center center_to_{data[1]}"
        ET.SubElement(root, "route", id=route_id, edges=route_egdes)  

    for vehicle_id, tracks in vehicle_tracks.items():
        if len(tracks) < 2:
            continue

        # Calculate departure time based on the first frame the vehicle appears
        first_frame = tracks[0][0]
        departure_time = first_frame / 30.0  #TODO: Actual FPS is 30.12 ....

        ET.SubElement(
            root, "vehicle",
            id=f"veh{vehicle_id}",
            type=tracks[0][4],
            route="route0",
            depart=f"{departure_time:.2f}"  
        )

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

# Function to generate sumo_config.sumocfg
def generate_config_file(output_file):
    root = ET.Element("configuration")
    input_el = ET.SubElement(root, "input")
    ET.SubElement(input_el, "net-file", value="simple_nw_se.net.xml")
    ET.SubElement(input_el, "route-files", value="route.rou.xml")
    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", value="0") 
    ET.SubElement(time_el, "end", value="1000") #TODO: Need to figure this out.
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

def draw_regions(frame, regions):
    """
    Draw region of interest (rectangles) for better understanding and analyzing.

    :param frame: Current frame to draw on. 
    :param region: Dict of defined region
    """

    for region, bounds in regions.items():
        color = bounds["color"]
        cv2.rectangle(frame, (bounds["x_min"], bounds["x_max"]), (bounds["y_min"], bounds["y_max"]), color, 2)
        cv2.putText(
                frame,
                region.upper(),
                (bounds["x_min"] + 10, bounds["y_min"] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
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

    for region, bounds in regions.items():
        if bounds["x_min"] <= cx <= bounds["x_max"] and bounds["y_min"] <= cy <= bounds["y_max"]:
            return region
        return None
    
# Function to process video and track vehicles
# km/hr
def process_video(video_path, conf_threshold=0.3):
    """
    Processes a video to detect vehicles, track them (id), and determine their speed.

    :param video_path: Path to the input video file.
    :param conf_threshold: Confidence threshold for detections (probability).
    :return: Dictionary containing track data for each vehicle.
    """

    model = YOLO("yolo11n.pt") 

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    regions = {
        "north": {"x_min": 0, "x_max": width, "y_min": 0, "y_max": height // 4, "color": (255,0,0)}, # Blue
        "east": {"x_min": 3 * width // 4, "x_max": width, "y_min": 0, "y_max": height, "color": (0,255,0)}, # Green
        "south": {"x_min": 0, "x_max": width, "y_min": 3 * height // 4, "y_max": height // 4, "color": (0,0,255)}, # Red
        "west": {"x_min": 0,  "x_max": width // 4, "y_min": 0, "y_max": height, "color": (255,255,0)}, # Cyan
    }


    # Initialize tracking and speed estimation variables
    track_data = {}  # vehicle_id: [(frame, cx, cy, speed, label, entry, exit), ...]
    frame_count = 0

    # Process video frames
    for results in model.track(source=video_path, conf=conf_threshold, show=False, stream=True):
        frame_count += 1
        frame = results.orig_img.copy()


        # Draw rectangular box around each region
        draw_regions(frame, regions)

        for box in results.boxes:
            if box.id is None:
                continue  # Skip untracked boxes

            object_id = int(box.id[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Car, bus, truck
            if cls not in [2, 5, 7]:  
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
            else:
                # Get the data from prev frame of a object
                last_frame, last_cx, last_cy, last_speed, _ = track_data[object_id][-1]
                frame_diff = frame_count - last_frame

                if frame_diff > 0:
                    meters_per_pixel = 0.05   #TODO: How to initialize this?
                    
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
            track_data.setdefault(object_id, []).append((frame_count, cx, cy, speed, label, entry_point, region))

            # Draw bounding box and annotations
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID:{object_id} {label} {speed:.2f} km/hr",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        cv2.imshow("Vehicle Detection and Speed Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return track_data


def main():

    # Path to the video
    video_path = 'Bellevue_116th_NE12th__2017-09-11_12-08-33.mp4'

    # Function call to process the video, returns dict of detected vehicles
    vehicle_tracks = process_video(video_path)
    # print(vehicle_tracks)

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

    generate_route_file(vehicle_tracks, "sumo_files/route.rou.xml",entry_exit_mappings) # Routes file
    generate_config_file("sumo_files/sumo_config.sumocfg") # Config File

    # Generate Network File
    os.system("netconvert --node-files sumo_files/nod.xml --edge-files sumo_files/edg.xml --type-files sumo_files/type.xml -o sumo_files/simple_nw_se.net.xml")

    print("SUMO input files generated successfully!")

if __name__ == "__main__":
    main()

