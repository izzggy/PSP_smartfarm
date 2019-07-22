from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import numpy as np
import json

def position():
    # define 81 classes that the coco model knowns about
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']
    
    # define the test configuration
    class TestConfig(Config):
        NAME = "test"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 80
    
    # define the model
    rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
    # load coco model weights
    rcnn.load_weights('mask_rcnn_coco.h5', by_name=True)
    # load photograph
    img = load_img('image1.jpg')
    img = img_to_array(img)
    # make prediction
    results = rcnn.detect([img], verbose=0)
    # get dictionary for first prediction
    r = results[0]
    class_id = class_names.index('potted plant')
    if class_names.index('potted plant') in r['class_ids']:
        k = list(np.where(r['class_ids'] == class_names.index('potted plant'))[0])
        r['scores'] = np.array([r['scores'][i] for i in k])
        r['rois'] = np.array([r['rois'][i] for i in k])
        print(r['rois'], type(r['rois']))
        rois_list = r['rois'].tolist()
        r['masks']=np.transpose(r['masks'])
        r['masks'] = np.array([r['masks'][i] for i in k])
        r['masks']=np.transpose(r['masks'])
        r['class_ids'] = np.array([r['class_ids'][i] for i in k])
        display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], figsize=(8, 8))
    return rois_list
