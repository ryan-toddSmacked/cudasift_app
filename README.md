# CudaSift Modified
This software is modified from [CudaSift](https://github.com/Celebrandil/CudaSift).\
Original authors and their paper.\
M. Björkman, N. Bergström and D. Kragic, "Detecting, segmenting and tracking unknown objects using multi-label MRF inference", CVIU, 118, pp. 111-127, January 2014.

## Changes
This modification is designed to downsize the images in the event of a memory allocation failure.\
It just catches cuda errors, and if it is a **"out of memory"** identiifier, everything is cleaned up, and the input images are scaled by 1/sqrt(2) in each dimension.\
This, in theory, should reduce the memory usage of the images on the device by half.\
Another modification was creating a new class that is designed to allocate the temporary memory for the laplace buffers when extracting the features.\
This was done strictly to auto-free the memory through c++ destructor. The functionality remains the same.\
All of the functionality that was detected to be unused was removed. This includes the texture code from the CudaImage class. All of the "#if 0" blocks.\
The last modification was placing the output into a json file. The json file stores the following.\
```json
{
    "image1": "path to the first image",
    "image2": "path to the second image",
    "homography": [
		[1.000303745e+00, 2.957944889e-05, -3.326675781e+03],
		[2.541389222e-05, 1.000007987e+00, -2.913469727e+03],
		[3.503976842e-08, 1.379766079e-08, 1.000000000e+00]
	],
	"original_size1": [17911, 10078],
	"original_size2": [17911, 10078],
	"final_size1": [8956, 5039],
	"final_size2": [8956, 5039],
	"elapsed_time": 1.875954
}
```
The final sizes are always populated and they indicate if the images needed to be downsampled in the event of memory allocation issues.
    
