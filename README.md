# UNET
[![DOI](https://zenodo.org/badge/137186334.svg)](https://zenodo.org/badge/latestdoi/137186334)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
[![HitCount](http://hits.dwyl.io/mfroeling/UNET.svg)](http://hits.dwyl.io/mfroeling/UNET)

A package to generate and train a UNET deep convolutional network for 2D and 3D image segmentation

* [Information](#information)
* [Install toolbox](#install-toolbox)
* [Using the toolbox](#using-the-toolbox)
* [Functionality](#functionality)

## Information

UNET is developed for [Mathematica](https://www.wolfram.com/mathematica/).
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
	* 2D UNET
![UNET 2D](https://github.com/mfroeling/UNET/blob/master/images/UNET2D.PNG)
	* 3D UNET
![UNET 3D](https://github.com/mfroeling/UNET/blob/master/images/UNET3D.PNG)
	* Loss Layers: Training the data is done using two loss layers: a SoftDiceLossLayer, BrierLossLayer and a CrossEntropyLossLayer.
![SoftDiceLossLayer, BrierLossLayer and a CrossEntropyLossLayer](https://github.com/mfroeling/UNET/blob/master/images/Loss.PNG)

* SplitTrainData splits the data and labels into training, validation and test data.
![split data](https://github.com/mfroeling/UNET/blob/master/images/Split.PNG)

* TrainUNET trains the network.
![Train Unet](https://github.com/mfroeling/UNET/blob/master/images/Train.PNG)

* Visualize the network and results
	* Visualize the layers
![Visualize the layers](https://github.com/mfroeling/UNET/blob/master/images/Visualize1.PNG)
	* Results
![Visualize hidden layers](https://github.com/mfroeling/UNET/blob/master/images/Visualize2.PNG)


Example: 3D segmentation of lower legg muscles using MRI data.

![Automated 3D muscle segmentation using UNET / RESNET using DIXON MRI data](https://github.com/mfroeling/UNET/blob/master/images/Muscle_Segmentation.jpg)


## License
https://opensource.org/licenses/MIT

Some code was based on https://github.com/alihashmiii/UNet-Segmentation-Wolfram
