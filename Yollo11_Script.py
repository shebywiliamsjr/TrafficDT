import cv2
from ultralytics import YOLO
import xml.etree.ElementTree as ET
import numpy as np
import time

def generate_sumo_route_file(vehicle_tracks, output_file):
    root = ET.Element("routes")
    vtype = ET.SubElement(root, "vType", id="car", accel="0.8", decel="4.5", sigma="0.5", length="5", minGap="2.5", maxSpeed="16.67", guiShape="passenger")

    for vehicle_id, tracks in vehicle_tracks.items():
        if len(tracks) < 2:
            continue
        
        start_time = tracks[0][0] / 30  # Assuming 30 fps
        vehicle = ET.SubElement(root, "vehicle", id=f"vehicle_{vehicle_id}", type="car", depart=f"{start_time:.2f}")
        
        # Simplified route generation
        route_edges = ["E0", "E1", "E2", "E3"]
        route = ET.SubElement(vehicle, "route", edges=" ".join(route_edges))

    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

def generate_sumo_network_file(output_file):
    root = ET.Element("net")
    
    # Define nodes
    nodes = [
        ("J0", "0", "0"),
        ("J1", "100", "0"),
        ("J2", "200", "0"),
        ("J3", "300", "0"),
        ("J4", "400", "0")
    ]
    
    for node_id, x, y in nodes:
        ET.SubElement(root, "junction", id=node_id, type="priority", x=x, y=y, incLanes="", intLanes="", shape=f"{x},1.60 {x},-1.60")
    
    # Define edges
    edges = [
        ("E0", "J0", "J1"),
        ("E1", "J1", "J2"),
        ("E2", "J2", "J3"),
        ("E3", "J3", "J4")
    ]
    
    for edge_id, from_node, to_node in edges:
        edge = ET.SubElement(root, "edge", id=edge_id, from_=from_node, to=to_node, priority="1")
        ET.SubElement(edge, "lane", id=f"{edge_id}_0", index="0", speed="13.89", length="100")
    
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

def generate_sumo_config_file(output_file):
    root = ET.Element("configuration")
    input_el = ET.SubElement(root, "input")
    ET.SubElement(input_el, "net-file", value="yolo11_road_network.net.xml")
    ET.SubElement(input_el, "route-files", value="yolo11_car_routes.rou.xml")
    
    time_el = ET.SubElement(root, "time")
    ET.SubElement(time_el, "begin", value="0")
    ET.SubElement(time_el, "end", value="1000")
    
    tree = ET.ElementTree(root)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

def process_video(video_path, conf_threshold=0.3):
    model = YOLO("yolo11n.pt")
    model.verbose = False
    cap = cv2.VideoCapture(video_path)
    
    vehicle_tracks = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, conf=conf_threshold, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls in [2, 5, 7]:  # car, bus, truck
                    vehicle_id = len(vehicle_tracks)
                    if vehicle_id not in vehicle_tracks:
                        vehicle_tracks[vehicle_id] = []
                    vehicle_tracks[vehicle_id].append((frame_count, (x1+x2)/2, (y1+y2)/2))

                label = f'{model.names[cls]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video processing completed.")
    
    return vehicle_tracks

# Main execution
video_path = '../Data/Bellevue_116th_NE12th__2017-09-11_12-08-33.mp4'
vehicle_tracks = process_video(video_path, conf_threshold=0.3)

generate_sumo_route_file(vehicle_tracks, 'yolo11_car_routes.rou.xml')
generate_sumo_network_file('yolo11_road_network.net.xml')
generate_sumo_config_file('yolo11_test.sumocfg')

print("SUMO input files generated successfully.")