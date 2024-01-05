# UNET

[![DOI](https://zenodo.org/badge/137186334.svg)](https://zenodo.org/badge/latestdoi/137186334)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmfroeling%2FUNET&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

## Notice

This package will no longer be developed!

Its main functionality generating UNETs has been integrated in [QMRITools](https://www.qmritools.com) in the [SegmentationTools](https://github.com/mfroeling/QMRITools/blob/master/QMRITools/Kernel/SegmentationTools.wl) package.
The last update included a more flexible down sampling schedule for and some other customizations for UNET generation.
It has been a playground for development learning and testing.

A package to generate and train a UNET deep convolutional network for 2D and 3D image segmentation

* [Information](#information)
* [Install toolbox](#install-toolbox)
* [Using the toolbox](#using-the-toolbox)
* [Functionality](#functionality)
* [Visualization](#visualization)
* [Example](#example)

## Information

UNET is developed for [Mathematica](https://www.wolfram.com/mathematica/).
It contains the following toolboxes:

* UnetCore
* UnetSupport

Documentation of all functions and their options is fully integrated in the Mathematica documentation.
The toolbox always works within the latest version of Mathematica and does not support any backward compatibility.

## Install toolbox

Install the toolbox in the Mathematica UserBaseDirectory > Applications.

 FileNameJoin[{$UserBaseDirectory, "Applications"}]
  
## Using the toolbox

The toolbox can be loaded by using <<UNET`

The notebook ``UNET.nb`` shows examples of how to use the toolbox on artificially generated 2D data.
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

* Convolution blocks: The toolbox contains five different convolution blocks that build up the network: [UNET](https://arxiv.org/abs/1505.04597), UResNet, [RestNet](https://arxiv.org/abs/1512.03385), UDenseNet, [DensNet](https://arxiv.org/abs/1608.06993).  
![split data](https://github.com/mfroeling/UNET/blob/master/images/convblocks.PNG)

* Network complexity for each of the blocks and for 2D and 3D UNET.
![UNET complexity](https://github.com/mfroeling/UNET/blob/master/images/networks.png)

* SplitTrainData splits the data and labels into training, validation and test data.  
![split data](https://github.com/mfroeling/UNET/blob/master/images/Split.PNG)

* TrainUNET trains the network.  
![Train Unet](https://github.com/mfroeling/UNET/blob/master/images/Train.PNG)

* Training is done with random batch selection that allows for on the fly data augmentation.  
![Train Unet](https://github.com/mfroeling/UNET/blob/master/images/batch.png)

## Visualization

* Visualize the network and results
  * Visualize the layers  
![Visualize the net layers](https://github.com/mfroeling/UNET/blob/master/images/Visualize1.PNG)
  * Results  
![Visualize training results](https://github.com/mfroeling/UNET/blob/master/images/Visualize2.PNG)
  * Visualize the training  
![animate unet training process](https://github.com/mfroeling/UNET/blob/master/images/amin0-v2.gif)  
![animate unet training process muscle](https://github.com/mfroeling/UNET/blob/master/images/anim.gif)

## Example

*Example: 3D segmentation of lower leg muscles using MRI data.  
![Automated 3D muscle segmentation using UNET / RESNET using DIXON MRI data](https://github.com/mfroeling/UNET/blob/master/images/Muscle_Segmentation.jpg)

## License

[MIT](https://opensource.org/licenses/MIT)

Some code was based on [this blog post](https://github.com/alihashmiii/UNet-Segmentation-Wolfram)
