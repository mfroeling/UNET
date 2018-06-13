---
title: 'A UNET implementation for image segmentation'
tags:
  - UNET
  - Mathematica
  - Image segmentation
  - deep convolutional network
authors:
 - name: M Froeling
   orcid: 0000-0003-3841-0497
   affiliation: "1"
affiliations:
 - name:  affiliation: Department of Radiology, University Medical Center Utrecht, Utrecht, The Netherlands
   index: 1
date: 02 June 2018
bibliography: paper.bib
---

# Summary

The is an implementation of UNET [DBLP:journals/corr/RonnebergerFB15] in Mathematica which can be used for image segmentation. The implementation automatically detect the number of channels and classes that are used in the network. The data dimensions can be 2D or 3D with size (N x N x N) where N should be a member of {16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 
256, 272, 288, 304, 320}. The number of trainable parameters of the first layer can be set and the rest of the network is generated accordingly. 

The data for training should be 2D or 3D array for each channel and the labels should be integers (1: background, 2...: the classes). The toolbox has functions to automatically split you data into train, validation and test data and formats it such that is can be used as an input for the net. 

The notbook ``UNET.nb`` shows examples of how to use the toolbox on artificially generated 2D data. There are also examples how to visualize the layer of your trained network and how to visualize the training itself. All functions have documentation integerated in the mathematica help browser.

# Acknoledgements 

# References

