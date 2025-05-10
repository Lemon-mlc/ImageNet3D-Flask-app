# ImageNet3D Flask app
## Guidelines
***Matching mesh model (with initializations provided by a pretrained model).***
The best matching model from a list of mesh models given. This is crucial to make accurate estimations of the following parameters so this should be the first thing to do. Click on "<- Model" and "Model ->" buttons to change the mesh model selection.

***3D rotation (with initializations provided by a pretrained model).***
Three parameters are used to specify the 3D rotation of an object: azimuth, elevation, and in-plane rotation (theta). Adjust the three parameters so the 3D rotation of the rendered object aligns with the 3D rotation of the object in the image. Make sure you are not simply aligning the segmentation or boundaries of the two objects. You should focus on aligning the 3D rotation of the objects so the rendered object is "pointing" to the same direction as the object in the image.

***2D location (with initializations provided by a pretrained model).*** 2D location specifies the location of the center of the object.

***Distance (with initializations provided by a pretrained model).***
Distance between the object and the camera. To annotate the distance, make sure the "size" of the rendered object is roughly the same as the "size" of the object in the image.

***Object quality.***
 Object quality specifies how clearly the object is visible from the image. A "good" object would be clearly visible and not occluded. A "bad" object may be barely visible (imaging drving in heavy fog when cars in front of you are barely visible) or occluded by other objects. Several choices are considered:
 -  **Good.** Most part (more than 90%) of the object is clearly visible in the image.
 -  **Partially visible.** A small part of the object is occluded by other objects or outside the image (truncated by image boundary).
 -  **Barely visible.** only a small part of the object is clearly visible -- the other parts are either occluded or outside the image, or barely visible due to other reasons (e.g., weather).
 -  **Bad quality / no object.** Most part of the object is occluded or outside the image; or the pose of the object is very hard to tell.

***Dense scene.*** 
This parameter tells if an object is very close to another object from the same category. Here "close" is defined in the 2D image plane -- two objects are close if the distance between them is small in the 2D image plane.

**Not dense scene.** The object is not close to another object from the same category. There can be multiple objects from the same category in one image but the objects are far away from each other.
**Dense scene.**The object is very close to another object from the same category. They may occlude each other or simply very close to each other.