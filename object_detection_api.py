import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class ObjectDetection:
	def __init__(self):
		# What model to download.
		current_path = os.path.dirname(os.path.realpath(__file__))
		# MODEL_NAME = 'inference_graph_doll_n_plane'

		MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
		MODEL_FILE = MODEL_NAME + '.tar.gz'
		DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

		# Path to frozen detection graph. This is the actual model that is used for the object detection.
		PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
		# PATH_TO_CKPT = os.path.join(current_path, MODEL_NAME, 'frozen_inference_graph.pb')
		# List of the strings that is used to add correct label for each box.
		# PATH_TO_LABELS = os.path.join(current_path,'data', 'object-detection.pbtxt')
		PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

		NUM_CLASSES = 90

		## Download Model
		opener = urllib.request.URLopener()
		opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
		tar_file = tarfile.open(MODEL_FILE)
		for file in tar_file.getmembers():
			file_name = os.path.basename(file.name)
			if 'frozen_inference_graph.pb' in file_name:
				tar_file.extract(file, os.getcwd())

		# ## Load a (frozen) Tensorflow model into memory.
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		# ## Loading label map
		label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
		                                                            use_display_name=True)
		self.category_index = label_map_util.create_category_index(categories)

		self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
		# Each box represents a part of the image where a particular object was detected.
		self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
		# Score is shown on the result image, together with the class label.
		self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
		self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
		self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

	def object_detection(self, cv_image):
		with self.detection_graph.as_default():
			with tf.Session(graph=self.detection_graph) as sess:
				# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np_expanded = np.expand_dims(cv_image, axis=0)

				# Actual detection.
				(boxes, scores, classes, num_detections) = sess.run(
					[self.boxes, self.scores, self.classes, self.num_detections],
					feed_dict={self.image_tensor: image_np_expanded})
				# Visualization of the results of a detection.

				vis_util.visualize_boxes_and_labels_on_image_array(
					cv_image,
					np.squeeze(boxes),
					np.squeeze(classes).astype(np.int32),
					np.squeeze(scores),
					self.category_index,
					use_normalized_coordinates=True,
					line_thickness=4)

		return cv_image

