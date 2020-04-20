python label_processor.py
cd YOLOv3/
python train.py --data data/visDrone.data --epoch 300 --batch 4 --weights '' --cfg cfg/yolov3_5l.cfg --img-size 512
