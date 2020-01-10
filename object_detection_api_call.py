import time
import cv2
from object_detection_api import ObjectDetection
import tensorflow as tf
import numpy as np
from object_detection.utils import visualization_utils as vis_util

obj = None

def detection_object():
	cap = cv2.VideoCapture(0)
	# cap = cv2.VideoCapture('media/sample_video.mp4')
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
				# print(boxes)

				# Visualization of the results of a detection.
				stop_time = time.time()
				vis_util.visualize_boxes_and_labels_on_image_array(
					image_np,
					np.squeeze(boxes),
					np.squeeze(classes).astype(np.int32),
					np.squeeze(scores),
					obj.category_index,
					use_normalized_coordinates=True,
					max_boxes_to_draw=10,
					line_thickness=8)

				print('Detection Time: ', stop_time - start_time, 'secs.')

				cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
				if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break

if __name__ == '__main__':
	detection_object()