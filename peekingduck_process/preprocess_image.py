from PIL import Image
from peekingduck.pipeline.nodes.model.jde import Node
import numpy as np

class Preprocessor:
    def __init__(self,**kwargs):
        self.tracker=Node(kwargs).model.tracker
    def __call__(self,image):
        image=image.convert('RGB')
        image_array=np.flip(np.asarray(image),2)
        predicts=self.tracker.track_objects_from_image(image_array)

        if predicts == ([],[],[]):
            return [],[]

        bboxes=np.clip(predicts[0],0,1)
        feats={track.track_id: track.features[-1] for track in self.tracker.tracked_stracks}
        x_bboxes=np.rint(image_array.shape[1]*bboxes[:,0::2]).astype(int)
        y_bboxes=np.rint(image_array.shape[0]*bboxes[:,1::2]).astype(int)
        images=[]
        for (left,right),(up,down) in zip(x_bboxes,y_bboxes):
            images.append(image.crop((left,up,right,down)))
        return images,[feats[f] for f in predicts[1]]
