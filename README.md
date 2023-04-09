# Recreating images with graphical objects using evolutionary computation

## Description

The goal of the project is to make program for recreating image provided by the user using 
graphic objects (e. g. letters, shapes, ect.) with proper parameters such as color, size,
or position, in such a way to  get the best recreation with the least number of used objects.

## Example

Given image:

<img src="https://user-images.githubusercontent.com/104419783/230797074-3cd978a3-ebb3-4613-b2bc-fe1e9f64f3da.png" width="300" />

Gif showing process of recreating given image (around 210 objects used):

<img src="https://user-images.githubusercontent.com/104419783/230797078-8b5922a6-d66e-4409-b0e8-c482c2181b63.gif" width="300" />


For more examples see `Examples` directory.

## Usage

To use it, change image from `picture` directory for your image, then run `generator.py` file and the results will start to save in `generated_im` directory.
In addition, at the bottom of `generator.py` file you can change parameters of evolutionary computation. 

## Inspiration

The inspiration for creating the program was [the video](https://youtu.be/6aXx6RA1IK4) by Spu7Nix,
showing how to recreate images and videos using objects in Geometry Dash level editor.

## Note

There are some things I would like to add to this project in the future like some small improvements, 
a lot of fun features, graphical interface or maybe CUDA optimization.
