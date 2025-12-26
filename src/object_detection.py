

class ObjectDetector:
    def __init__(self, model):
        self.model = model

    def detect(self, image):
        return self.model.detect(image)