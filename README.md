# CudaSift
This software is modified from [CudaSift](https://github.com/Celebrandil/CudaSift).\
Original authors and their paper.\
M. Björkman, N. Bergström and D. Kragic, "Detecting, segmenting and tracking unknown objects using multi-label MRF inference", CVIU, 118, pp. 111-127, January 2014.

## Changes
This modification is designed to downsize the images in the event of a memory allocation failure.\
It just catches cuda errors, and if it is a **"out of memory"** identiifier, everything is cleaned up, and the input images are scaled by 1/sqrt(2) in each dimension.\
This, in theory, should reduce the memory usage of the images on the device by half.\
Another modification was creating a new class that is designed to allocate the temporary memory for the laplace buffers when extracting the features.\
This was done strictly to auto-free the memory through c++ destructor. The functionality remains the same.\
