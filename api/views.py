from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import cv2, json
import numpy as np
import base64
import secrets
from django.core.files.base import ContentFile

# Create your views here.
@api_view(['GET'])
def index_page(request):
    return_data = {
        "error" : "0",
        "message" : "Successful",
    }
    return Response(return_data)


@api_view(["POST"])
def yolov4tiny_outcome(request):
    try:
        # classes(list sent as a string) where image's sign language word belongs to
        classes_str = request.data.get('classes', None)
        classes = eval(classes_str)

        # when inputting image via http: convert from dataURL to image
        image = request.data.get('image_serialized', None)
        _format, _image_url = image.split(';base64,')
        image_64 = base64.b64decode(_image_url)
        #from base64 to OpenCV Image
        image_arr = np.frombuffer(image_64, dtype="int16")  # im_arr is one-dim Numpy array
        nparr = cv2.imdecode(image_arr, flags=cv2.IMREAD_COLOR)
        print(nparr.shape)

        # # when inputting image locally: image decode
        # image = request.data.get('image_serialized', None)
        # image_64 = base64.b64decode(image)
        # nparr = np.frombuffer(image_64, dtype=np.uint8)
        # nparr = nparr.reshape(720, 1280, 3)
       
        #fields to check input data are valid
        fields = [classes_str, image]

        if not None in fields:
            img = nparr

            # image resize
            img = cv2.resize(img, (1280, 720))
            #img = cv2.resize(img, (640, 360))

            boxes = []
            confidences = []
            class_ids = []
            # 추후에 model을 처음 한번만 load하고 매번 load하지 않아도 되도록 수정
            config_path = './model/assets/yolov4-tiny.cfg'
            weights_path = './model/assets/yolov4-tiny_last.weights'
            net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
            net.setInput(blob)
            output_layers_name = net.getUnconnectedOutLayersNames()
            layerOutputs = net.forward(output_layers_name)

            height, width, _ = img.shape
            for output in layerOutputs:
                for detection in output:
                    score = detection[5:]
                    class_id = np.argmax(score)
                    confidence = score[class_id]

                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)

                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.4)
            print (indexes)
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i], 5))

            # save output of yolov4 tiny model as json type
            result = {
                'word_index': label,
                'box_coord': [x, y, w, h],
                'confidence': confidence
            }
            json_to_spring = json.dumps(result)
        else:
            json_to_spring = {
                'error': '1',
                'message': 'Issues with classes or image_deserialized'
            }
    except Exception as e:
        print(e)
        json_to_spring = {
            'error': '2',
            "message": str(e)
        }
    print(json_to_spring)
    return Response(json_to_spring)