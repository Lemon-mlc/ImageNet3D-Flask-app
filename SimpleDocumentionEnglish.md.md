Instructions
Installation of Packages
Before starting, you need to install the following packages: Flask, numpy, matplotlib, pillow, opencv-python-headless, and flask-login. You can install them all at once with the command: pip install Flask numpy matplotlib pillow opencv-python-headless flask-login.

When the program runs, it will create an imagenet3dstorage folder in the current directory. There will be an empty sqlite file in the subfolder. You just need to replace its content with that of the database_default.sqlite file in the root directory of the software. If you don't know how to replace it, you can use the Navicat Premium software.

The file structure is as follows:

        
复制代码
ImageNet3D-Flask-app/
├── static/
│   ├── images/
│   ├── tmp/
├── templates/
│   ├── login.html
│   ├── account.html
│   ├── test.html
│   ├── index.html
│   ├── eval.html
│   ├── quality.html
│   ├── doc.html
├── app.py
├── config.py
├── render.py
├── database.sqlite
├── database_default.sqlite
├── README.md
├── README.zh.md
├── requirements.txt
└── logs/

    
Overview
In previous studies, synthetic images with pseudo-3D annotations were researched, and about 75% of them met the annotations, while the rest had issues with viewing angles or shapes. Although the dataset has some noise, NMMs can learn from it and improve performance. This offers new opportunities for 3D understanding, but an accurately 3D-annotated dataset is needed to evaluate the models. This project describes objects through some parameters and also pays attention to subtypes. Each real image has pre-trained annotations, and you need to improve the predictions and annotate the quality. Make sure to read the tutorial before starting.

v2 Update (March 2024)
New coarse adjustment buttons are added because the initial 3D direction setting is not ideal when dealing with difficult categories.
The annotation process may take longer, so be patient and ensure the visual accuracy.
The image quality may be low. For "bad" objects, annotate and ignore them, and you can skip adjusting other parameters.
Annotating visibility and scene density is still important.
Category-Specific Notes
Tandem Bicycle: Use the ordinary bicycle model and ensure the 2D center and 3D direction match.
Beaker: Match the 3D viewpoint by the direction of the beaker's notch, and other parameters are simple.
Accordion: Ignore the "azimuth" and match other parameters.
Crutches: Only annotate the underarm crutches type, and mark others as "poor quality/no object".
Sandbag: Only annotate the punching bags and hanging bags (cylindrical), and ignore the "azimuth".
User Interface
Log in with your ID to see the task list and track your progress. Click on a task to jump to the first unannotated issue. There are information and buttons at the top. You can switch between issues, undo, and save (pressing Enter also saves). At the bottom, there is a list of CAD models. Besides model parameters, you also need to answer questions about quality and scene density. After saving successfully, you will get a prompt and the status will light up.

Guide
Matching Grid Model: Choose the best matching model from the list. It's very important and should be done first. Click the buttons to change the model.
3D Rotation: Use three parameters to adjust so that the 3D rotation of the rendered object aligns with that of the image object, not just aligning the segmentation or boundaries.
2D Position: Determine the center position of the object.
Distance: When annotating, make the size of the rendered object similar to that of the image object.
Object Quality: Divided into "good", "partially visible", "almost invisible", and "poor quality/no object".
Dense Scene: Divided into "non-dense" and "dense", judged by the distance between objects of the same category in the 2D plane.
Example
See the tutorial.

References
[1] Adding 3D Geometric Control in Diffusion Models
[2] Robust Category-Level 6D Pose Estimation Based on Neural Feature Coarse-to-Fine Rendering

