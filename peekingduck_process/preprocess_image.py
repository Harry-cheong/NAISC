from peekingduck.pipeline.nodes.model.mtcnn import Node
import numpy as np

class Preprocessor:
    def __init__(self,**kwargs):
        self.detector=Node(kwargs).model.detector
    def __call__(self,image):
        image=image.convert('RGB')
        image_array=np.flip(np.asarray(image),2)
        predicts=self.detector.predict_object_bbox_from_image(image_array)

        if not predicts[0].any():
            return [],[]
        #print(predicts)
        bboxes=np.clip(predicts[0],0,1)
        landmarks=np.concatenate([(yx if ind == 1 else np.flip(yx,1)) for ind,yx in enumerate(predicts[2])],axis=1)
        x_bboxes=np.rint(image_array.shape[1]*bboxes[:,0::2]).astype(int)
        y_bboxes=np.rint(image_array.shape[0]*bboxes[:,1::2]).astype(int)
        images=[]
        for (left,right),(up,down) in zip(x_bboxes,y_bboxes):
            images.append(image.crop((left,up,right,down)))
        return images,landmarks

if __name__ == '__main__':
    from PIL import Image
    p=Preprocessor()
    print(p(Image.open(r'..\imgDataset\2.jpg')))