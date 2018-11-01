Quick-start

To start using the mAP you need to clone the repo:

git clone https://github.com/TianyouChen/mAP.git



Running the code

Step by step:

1.Create the ground-truth files
2.Move the ground-truth files into the folder ground-truth/
3.Create the predicted objects files
4.Move the predictions files into the folder predicted/
5.Run the code: python main.py

Optional (if you want to see the animation):

6.Insert the images into the folder images/






Create the ground-truth files

Create a separate ground-truth text file for each image.
Use matching names (e.g. image: "image_1.jpg", ground-truth: "image_1.txt"; "image_2.jpg", "image_2.txt"...).
In these files, each line should be in the following format:
<class_name> <left> <top> <right> <bottom> [<difficult>]
The difficult parameter is optional, use it if you want to ignore a specific prediction.
E.g. "image_1.txt":
tvmonitor 2 10 173 238
book 439 157 556 241
book 437 246 518 351 difficult
pottedplant 272 190 316 259
Create the predicted objects files

Create a separate predicted objects text file for each image.
Use matching names (e.g. image: "image_1.jpg", predicted: "image_1.txt"; "image_2.jpg", "image_2.txt"...).
In these files, each line should be in the following format:
<class_name> <confidence> <left> <top> <right> <bottom>
E.g. "image_1.txt":
tvmonitor 0.471781 0 13 174 244
cup 0.414941 274 226 301 265
book 0.460851 429 219 528 247
chair 0.292345 0 199 88 436
book 0.269833 433 260 506 336
