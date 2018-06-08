# from styx_msgs.msg import TrafficLight

# class TLClassifier(object):
#     def __init__(self):
#         #TODO load classifier
#         pass

#     def get_classification(self, image):
#         """Determines the color of the traffic light in the image

#         Args:
#             image (cv::Mat): image containing the traffic light

#         Returns:
#             int: ID of traffic light color (specified in styx_msgs/TrafficLight)

#         """
#         #TODO implement light color prediction
#         return TrafficLight.UNKNOWN


from styx_msgs.msg import TrafficLight
from keras.models import model_from_yaml
import numpy as np
import tensorflow as tf


MAPPING = {0 : TrafficLight.GREEN,
           1 : TrafficLight.UNKNOWN,
           2 : TrafficLight.RED,
           3 : TrafficLight.YELLOW           
}

class TLClassifier(object):
    def __init__(self):
        # print('Load model')
        # Load model
        with open('../../../classifier/model.yaml', 'r') as f:
            self.model = model_from_yaml(f.read())
        self.model.load_weights('../../../classifier/weights.h5')   
        self.graph = tf.get_default_graph()
        pass     

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.graph.as_default():
            prediction = self.model.predict(image[np.newaxis, ...]).flatten()
            return MAPPING[np.argmax(prediction)]

        