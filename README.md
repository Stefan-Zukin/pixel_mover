# pixel_mover

The idea behind this project is to translate between one image and another by treating pixels as individual actors. This was motivated by free time and curiosity,
but I think it ended up making a pretty interesting effect.
You give an input image and a goal image and the program calculates the best way to represent the goal image using the pixels from the input image.
Currently this is just based on the grayscale values of the pixels.
Then, each pixel is assigned a target coordinate, and the program iterates over each pixel, at each step, the pixel swaps with another in a way that brings it closer to the goal.

This is currently very slow. I optimized it a little bit, but there's still more that could be done such as GPU support.

## Examples
![random_mover](/demo_movies/random_scan.mp4)
