# UNET
A package to generate and train a UNET deep convolutional network for 2D and 3D image segmentation

## Information

DTITools is developed for [Mathematica](https://www.wolfram.com/mathematica/).
It contains the following toolboxes:

- UnetCore
- UnetSupport

Documentation of all functions and their options is fully integrated in the Mathematica documentation.
The toolbox always works within the latest version of Mathematica and does not support any backward compatibility.

All code and documentation is maintained and uploaded to github using [Workbench](https://www.wolfram.com/workbench/).

## Install toolbox

Install the toolbox in the Mathematica UserBaseDirectory > Applications.

	FileNameJoin[{$UserBaseDirectory, "Applications"}]
  
## Using the toolbox

The toolbox can be loaded by using <<UNET`

The notbook ``UNET.nb`` shows examples of how to use the toolbox on artificially generated 2D data. 
There are also examples how to visualize the layer of your trained network and how to visualize the training itself. 

## Functionality

The network supports multi channel inputs and multi class segmentation.

* UNET generates a UNET convolutional network.
* TrainUNET trains the network.
* SplitTrainData splits the data and labels into training, validation and test data.

Training the data is done using two loss layers: a SoftDiceLossLayer and a CrossEntropyLossLayer.

## License
https://opensource.org/licenses/MIT

Some code was based on https://github.com/alihashmiii/UNet-Segmentation-Wolfram
