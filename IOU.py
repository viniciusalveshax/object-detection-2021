def inside(boxA, boxB):
  x1A, y1A, x2A, y2A = boxA
  x1B, y1B, x2B, y2B = boxB

  if (x1A <= x1B) and (y1A <= y1B) and (x2A >= x2B) and (y2A >= y2B):
    return True
  else:
    return False

#Código original dessa função por Coutinho
# https://github.com/lucas-coutinho/
def IOU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
  # boxA= (boxA[0],boxA[1], boxA[0] + width, boxA[1] + height)
  # boxB= (boxB[0],boxB[1], boxB[0] + width, boxB[1] + height)
  print("IOU: Comparando A: ", boxA, " com B: ", boxB)

  iou = 0

  if inside(boxA, boxB):
    print("B está dentro de A")
    #IOU é máximo então retorna 1
    iou=1.0
  else:
    if inside(boxB, boxA):
      print("A está dentro de B")
      iou=1.0

  if iou == 1.0:
    print("iou: ", iou)
    return iou

  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
  # compute the area of intersection rectangle
  interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = interArea / float(boxAArea + boxBArea - interArea)
  print("iou: ", iou)
  # return the intersection over union value
  return iou

