import glob
import json
import os
import shutil
import operator
import sys
import argparse

MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)

parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
# argparse receiving list of classes to be ignored
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
# argparse receiving list of classes with specific IoU
parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
args = parser.parse_args()

# if there are no classes to ignore then replace None by empty list
if args.ignore is None:
  args.ignore = []

specific_iou_flagged = False
if args.set_class_iou is not None:
  specific_iou_flagged = True

# if there are no images then no animation can be shown
img_path = 'images'
if os.path.exists(img_path): 
  for dirpath, dirnames, files in os.walk(img_path):
    if not files:
      # no image files found
      args.no_animation = True
else:
  args.no_animation = True

# try to import OpenCV if the user didn't choose the option --no-animation
show_animation = False
if not args.no_animation:
  try:
    import cv2
    show_animation = True
  except ImportError:
    print("\"opencv-python\" not found, please install to visualize the results.")
    args.no_animation = True

# try to import Matplotlib if the user didn't choose the option --no-plot
draw_plot = False
if not args.no_plot:
  try:
    import matplotlib.pyplot as plt
    draw_plot = True
  except ImportError:
    print("\"matplotlib\" not found, please install it to get the resulting plots.")
    args.no_plot = True

"""
 throw error and exit
"""
def error(msg):
  print(msg)
  sys.exit(0)

"""
 check if the number is a float between 0.0 and 1.0
"""
def is_float_between_0_and_1(value):
  try:
    val = float(value)
    if val > 0.0 and val < 1.0:
      return True
    else:
      return False
  except ValueError:
    return False

"""
 Calculate the AP given the recall and precision array
  1st) We compute a version of the measured precision/recall curve with
       precision monotonically decreasing
  2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
  """
  --- Official matlab code VOC2012---
  mrec=[0 ; rec ; 1];
  mpre=[0 ; prec ; 0];
  for i=numel(mpre)-1:-1:1
      mpre(i)=max(mpre(i),mpre(i+1));
  end
  i=find(mrec(2:end)~=mrec(1:end-1))+1;
  ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
  """
  rec.insert(0, 0.0) # insert 0.0 at begining of list
  rec.append(1.0) # insert 1.0 at end of list
  mrec = rec[:]
  prec.insert(0, 0.0) # insert 0.0 at begining of list
  prec.append(0.0) # insert 0.0 at end of list
  mpre = prec[:]
  
 
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])
 
  i_list = []
  for i in range(1, len(mrec)):
    if mrec[i] != mrec[i-1]:
      i_list.append(i) 
  ap = 0.0
  for i in i_list:
    ap += ((mrec[i]-mrec[i-1])*mpre[i])
  return ap, mrec, mpre


def file_lines_to_list(path):

  with open(path) as f:
    content = f.readlines()
  
  content = [x.strip() for x in content]
  return content


def draw_text_in_image(img, text, pos, color, line_width):
  font = cv2.FONT_HERSHEY_PLAIN
  fontScale = 1
  lineType = 1
  bottomLeftCornerOfText = pos
  cv2.putText(img, text,
      bottomLeftCornerOfText,
      font,
      fontScale,
      color,
      lineType)
  text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
  return img, (line_width + text_width)

"""
 Plot - adjust axes
"""
def adjust_axes(r, t, fig, axes):
  # get text width for re-scaling
  bb = t.get_window_extent(renderer=r)
  text_width_inches = bb.width / fig.dpi
  # get axis width in inches
  current_fig_width = fig.get_figwidth()
  new_fig_width = current_fig_width + text_width_inches
  propotion = new_fig_width / current_fig_width
  # get axis limit
  x_lim = axes.get_xlim()
  axes.set_xlim([x_lim[0], x_lim[1]*propotion])


"""
 Create a "tmp_files/" and "results/" directory
"""
tmp_files_path = "tmp_files"
if not os.path.exists(tmp_files_path): # if it doesn't exist already
  os.makedirs(tmp_files_path)
results_files_path = "results"
if os.path.exists(results_files_path): # if it exist already
  # reset the results directory
  shutil.rmtree(results_files_path)

os.makedirs(results_files_path)
if draw_plot:
  os.makedirs(results_files_path + "/classes")
if show_animation:
  os.makedirs(results_files_path + "/images")
  os.makedirs(results_files_path + "/images/single_predictions")

"""
 Ground-Truth
   Load each of the ground-truth files into a temporary ".json" file.
   Create a list of all the class names present in the ground-truth (gt_classes).
"""
# get a list with the ground-truth files
ground_truth_files_list = glob.glob('ground-truth/*.txt')
if len(ground_truth_files_list) == 0:
  error("Error: No ground-truth files found!")
ground_truth_files_list.sort()
# dictionary with counter per class
gt_counter_per_class = {}

for txt_file in ground_truth_files_list:
  #print(txt_file)
  file_id = txt_file.split(".txt",1)[0]
  file_id = os.path.basename(os.path.normpath(file_id))
  # check if there is a correspondent predicted objects file
  if not os.path.exists('predicted/' + file_id + ".txt"):
    error_msg = "Error. File not found: predicted/" +  file_id + ".txt\n"
    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
    error(error_msg)
  lines_list = file_lines_to_list(txt_file)
  # create ground-truth dictionary
  bounding_boxes = []
  is_difficult = False
  for line in lines_list:
    try:
      if "difficult" in line:
          class_name, left, top, right, bottom, _difficult = line.split()
          is_difficult = True
      else:
          class_name, left, top, right, bottom = line.split()
    except ValueError:
      error_msg = "Error: File " + txt_file + " in the wrong format.\n"
      error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
      error_msg += " Received: " + line
      error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
      error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
      error(error_msg)
    # check if class is in the ignore list, if yes skip
    if class_name in args.ignore:
      continue
    bbox = left + " " + top + " " + right + " " +bottom
    if is_difficult:
        bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
        is_difficult = False
    else:
        bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
        # count that object
        if class_name in gt_counter_per_class:
          gt_counter_per_class[class_name] += 1
        else:
          # if class didn't exist yet
          gt_counter_per_class[class_name] = 1
  # dump bounding_boxes into a ".json" file
  with open(tmp_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
    json.dump(bounding_boxes, outfile)

gt_classes = list(gt_counter_per_class.keys())
# let's sort the classes alphabetically
gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)
#print(gt_classes)
#print(gt_counter_per_class)

"""
 Check format of the flag --set-class-iou (if used)
  e.g. check if class exists
"""
if specific_iou_flagged:
  n_args = len(args.set_class_iou)
  error_msg = \
    '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
  if n_args % 2 != 0:
    error('Error, missing arguments. Flag usage:' + error_msg)
  # [class_1] [IoU_1] [class_2] [IoU_2]
  # specific_iou_classes = ['class_1', 'class_2']
  specific_iou_classes = args.set_class_iou[::2] # even
  # iou_list = ['IoU_1', 'IoU_2']
  iou_list = args.set_class_iou[1::2] # odd
  if len(specific_iou_classes) != len(iou_list):
    error('Error, missing arguments. Flag usage:' + error_msg)
  for tmp_class in specific_iou_classes:
    if tmp_class not in gt_classes:
          error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
  for num in iou_list:
    if not is_float_between_0_and_1(num):
      error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

"""
 Predicted
   Load each of the predicted files into a temporary ".json" file.
"""
# get a list with the predicted files
predicted_files_list = glob.glob('predicted/*.txt')
predicted_files_list.sort()

for class_index, class_name in enumerate(gt_classes):
  bounding_boxes = []
  for txt_file in predicted_files_list:
    #print(txt_file)
    # the first time it checks if all the corresponding ground-truth files exist
    file_id = txt_file.split(".txt",1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))
    if class_index == 0:
      if not os.path.exists('ground-truth/' + file_id + ".txt"):
        error_msg = "Error. File not found: ground-truth/" +  file_id + ".txt\n"
        error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
        error(error_msg)
    lines = file_lines_to_list(txt_file)
    for line in lines:
      try:
        tmp_class_name, confidence, left, top, right, bottom = line.split()
      except ValueError:
        error_msg = "Error: File " + txt_file + " in the wrong format.\n"
        error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
        error_msg += " Received: " + line
        error(error_msg)
      if tmp_class_name == class_name:
        #print("match")
        bbox = left + " " + top + " " + right + " " +bottom
        bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
        #print(bounding_boxes)
  # sort predictions by decreasing confidence
  bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
  with open(tmp_files_path + "/" + class_name + "_predictions.json", 'w') as outfile:
    json.dump(bounding_boxes, outfile)

"""
 Calculate the AP for each class
"""
sum_AP = 0.0
ap_dictionary = {}
# open file to store the results
with open(results_files_path + "/results.txt", 'w') as results_file:
  results_file.write("# AP and precision/recall per class\n")
  count_true_positives = {}
  for class_index, class_name in enumerate(gt_classes):
    count_true_positives[class_name] = 0
    """
     Load predictions of that class
    """
    predictions_file = tmp_files_path + "/" + class_name + "_predictions.json"
    predictions_data = json.load(open(predictions_file))

    """
     Assign predictions to ground truth objects
    """
    nd = len(predictions_data)
    tp = [0] * nd # creates an array of zeros of size nd
    fp = [0] * nd
    for idx, prediction in enumerate(predictions_data):
      file_id = prediction["file_id"]
      if show_animation:
        # find ground truth image
        ground_truth_img = glob.glob1(img_path, file_id + ".*")
        #tifCounter = len(glob.glob1(myPath,"*.tif"))
        if len(ground_truth_img) == 0:
          error("Error. Image not found with id: " + file_id)
        elif len(ground_truth_img) > 1:
          error("Error. Multiple image with id: " + file_id)
        else: # found image
          #print(img_path + "/" + ground_truth_img[0])
          # Load image
          img = cv2.imread(img_path + "/" + ground_truth_img[0])
          # load image with draws of multiple detections
          img_cumulative_path = results_files_path + "/images/" + ground_truth_img[0]
          if os.path.isfile(img_cumulative_path):
            img_cumulative = cv2.imread(img_cumulative_path)
          else:
            img_cumulative = img.copy()
          # Add bottom border to image
          bottom_border = 60
          BLACK = [0, 0, 0]
          img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
      # assign prediction to ground truth object if any
      #   open ground-truth with that file_id
      gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
      ground_truth_data = json.load(open(gt_file))
      ovmax = -1
      gt_match = -1
      # load prediction bounding-box
      bb = [ float(x) for x in prediction["bbox"].split() ]
      for obj in ground_truth_data:
        # look for a class_name match
        if obj["class_name"] == class_name:
          bbgt = [ float(x) for x in obj["bbox"].split() ]
          bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
          iw = bi[2] - bi[0] + 1
          ih = bi[3] - bi[1] + 1
          if iw > 0 and ih > 0:
            # compute overlap (IoU) = area of intersection / area of union
            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                    + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
            ov = iw * ih / ua
            if ov > ovmax:
              ovmax = ov
              gt_match = obj

      # assign prediction as true positive/don't care/false positive
      if show_animation:
        status = "NO MATCH FOUND!" # status is only used in the animation
      # set minimum overlap
      min_overlap = MINOVERLAP
      if specific_iou_flagged:
        if class_name in specific_iou_classes:
          index = specific_iou_classes.index(class_name)
          min_overlap = float(iou_list[index])
      if ovmax >= min_overlap:
        if "difficult" not in gt_match:
            if not bool(gt_match["used"]):
              # true positive
              tp[idx] = 1
              gt_match["used"] = True
              count_true_positives[class_name] += 1
              # update the ".json" file
              with open(gt_file, 'w') as f:
                  f.write(json.dumps(ground_truth_data))
              if show_animation:
                status = "MATCH!"
            else:
              # false positive (multiple detection)
              fp[idx] = 1
              if show_animation:
                status = "REPEATED MATCH!"
      else:
        # false positive
        fp[idx] = 1
        if ovmax > 0:
          status = "INSUFFICIENT OVERLAP"

      
    cumsum = 0
    for idx, val in enumerate(fp):
      fp[idx] += cumsum
      cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
      tp[idx] += cumsum
      cumsum += val
    #print(tp)
    rec = tp[:]
    for idx, val in enumerate(tp):
      rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
    #print(rec)
    prec = tp[:]
    for idx, val in enumerate(tp):
      prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    #print(prec)

    ap, mrec, mprec = voc_ap(rec, prec)
    sum_AP += ap
    text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)
    """
     Write to results.txt
    """
    rounded_prec = [ '%.2f' % elem for elem in prec ]
    rounded_rec = [ '%.2f' % elem for elem in rec ]
    results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")
    if not args.quiet:
      print(text)
    ap_dictionary[class_name] = ap

    

  if show_animation:
    cv2.destroyAllWindows()

  results_file.write("\n# mAP of all classes\n")
  mAP = sum_AP / n_classes
  text = "mAP = {0:.2f}%".format(mAP*100)
  results_file.write(text + "\n")
  print(text)

# remove the tmp_files directory
shutil.rmtree(tmp_files_path)

"""
 Count total of Predictions
"""
# iterate through all the files
pred_counter_per_class = {}
#all_classes_predicted_files = set([])
for txt_file in predicted_files_list:
  # get lines to list
  lines_list = file_lines_to_list(txt_file)
  for line in lines_list:
    class_name = line.split()[0]
    # check if class is in the ignore list, if yes skip
    if class_name in args.ignore:
      continue
    # count that object
    if class_name in pred_counter_per_class:
      pred_counter_per_class[class_name] += 1
    else:
      # if class didn't exist yet
      pred_counter_per_class[class_name] = 1
#print(pred_counter_per_class)
pred_classes = list(pred_counter_per_class.keys())



"""
 Write number of ground-truth objects per class to results.txt
"""
with open(results_files_path + "/results.txt", 'a') as results_file:
  results_file.write("\n# Number of ground-truth objects per class\n")
  for class_name in sorted(gt_counter_per_class):
    results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

"""
 Finish counting true positives
"""
for class_name in pred_classes:
  # if class exists in predictions but not in ground-truth then there are no true positives in that class
  if class_name not in gt_classes:
    count_true_positives[class_name] = 0
#print(count_true_positives)


"""
 Write number of predicted objects per class to results.txt
"""
with open(results_files_path + "/results.txt", 'a') as results_file:
  results_file.write("\n# Number of predicted objects per class\n")
  for class_name in sorted(pred_classes):
    n_pred = pred_counter_per_class[class_name]
    text = class_name + ": " + str(n_pred)
    text += " (tp:" + str(count_true_positives[class_name]) + ""
    text += ", fp:" + str(n_pred - count_true_positives[class_name]) + ")\n"
    results_file.write(text)

"""
 Draw mAP plot (Show AP's of all classes in decreasing order)
"""
if draw_plot:
  window_title = "mAP"
  plot_title = "mAP = {0:.2f}%".format(mAP*100)
  x_label = "Average Precision"
  output_path = results_files_path + "/mAP.png"
  to_show = True
  plot_color = 'royalblue'
