import time
import cv2
from object_detection_api import ObjectDetection
import tensorflow as tf
import numpy as np
from object_detection.utils import visualization_utils as vis_util

obj = None

def compute_area_and_center(x1,y1,x2,y2):
	area_n_center = []
	area = (x2-x1)*(y2-y1)
	center_x = (x2+x1)/2
	center_y = (y2+y1)/2
	area_n_center = [area,center_x,center_y]
	return area_n_center

def compute_robot_action(x1,y1,x2,y2,image):
	moves = []
	#define the possible turning and moving action as strings
	turning = ""
	moving = ""

	area_n_center = compute_area_and_center(x1,y1,x2,y2)
	height, width = image.shape[:2]
	normalized_area = (1e6*area_n_center[0])/(height*width)
	# print('Normalized Area:',normalized_area)

	#obtain a x center between 0.0 and 1.0
	normalized_center_x = area_n_center[1] / width

	#obtain a y center between 0.0 and 1.0
	normalized_center_y = area_n_center[2] / height

	print('normalized center_x:', 1000*normalized_center_x)
	# print('normalized center_y:', normalized_center_y)

	#Right and left motion based upon the center
	if normalized_center_x > 0.6 :
		turning = "turn_right"
	elif normalized_center_x < 0.4 :
		turning = "turn_left"

	#if the area is too big move backwards
	if normalized_area >= 0.8000 :
		moving = "backwards"
	elif normalized_area < 0.7900 :
		moving = "ahead"

	moves = [turning,moving]
	return moves

def detection_object():
	cap = cv2.VideoCapture(0)
	global obj
	obj = ObjectDetection()
	with obj.detection_graph.as_default():
		with tf.Session(graph=obj.detection_graph) as sess:
			while True:
				ret, image_np = cap.read()
				# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np_expanded = np.expand_dims(image_np, axis=0)
				# Actual detection.
				start_time = time.time()
				(boxes, scores, classes, num_detections) = sess.run(
					[obj.boxes, obj.scores, obj.classes, obj.num_detections],
					feed_dict={obj.image_tensor: image_np_expanded})

				# Visualization of the results of a detection.
				stop_time = time.time()
				vis_util.visualize_boxes_and_labels_on_image_array(
					image_np,
					np.squeeze(boxes),
					np.squeeze(classes).astype(np.int32),
					np.squeeze(scores),
					obj.category_index,
					use_normalized_coordinates=True,
					line_thickness=8)

				# print(boxes[0][0][0], boxes[0][0][1], boxes[0][0][2], boxes[0][0][3])
				# print(obj.category_index[classes[0][0]]['name'])
				moves = compute_robot_action(boxes[0][0][0], boxes[0][0][1], boxes[0][0][2], boxes[0][0][3], image_np)
				print('Turning:',moves[0])
				print('Moving:',moves[1])
				# print('Detection Time: ', stop_time - start_time, 'secs.')

				cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
				if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break

if __name__ == '__main__':
	detection_object()