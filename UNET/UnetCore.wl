(* ::Package:: *)

(* ::Title:: *)
(*UNET*)


(* ::Subtitle:: *)
(*Written by: Martijn Froeling, PhD*)
(*m.froeling@gmail.com*)


(* ::Section:: *)
(*start Package*)


BeginPackage["UNET`UnetCore`"]

$ContextPath =  Union[$ContextPath, System`$UNETContextPath];


(* ::Section:: *)
(*Usage Notes*)


(* ::Subsection:: *)
(*Functions*)


DiceSimilarityClass::usage = 
"DiceSimilarityClass[prediction, groundTruth, nclasses] gives the Dice Similarity between of each of Nclasses between prediction and groundTruth. 
nClasses can also be a list of class number for which the Dice needs to be calculated."

DiceSimilarity::usage = 
"DiceSimilarity[x, y] gives the Dice Similarity between 1 and 0 of vectors x and y for class 1.
DiceSimilarity[x, y, class] gives the Dice Similarity for vectors x and y for Integer Class."

MakeUNET::usage = 
"MakeUNET[nchan, nclass, dep, dimIn] Generates a UNET with nchan as input and nclass as output. The number of parameter of the first convolution layer can be set with dep.
The data dimensions can be 2D or 3D and each of the dimensions should be 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240 or 256."

AddLossLayer::usage = 
"AddLossLayer[net] adds two loss layers to a NetGraph, a SoftDiceLossLayer and a CrossEntropyLossLayer."

SoftDiceLossLayer::usage = 
"SoftDiceLossLayer[dim] represents a net layer that computes the SoftDice loss by comparing input class probability vectors with the target class vector."

BrierLossLayer::usage = 
"BrierLossLayer[dim] represents a net layer that computes the Brier loss by comparing input class probability vectors with the target class vector."

TrainUNET::usage = 
"TrainUNET[trainData, validationData] Trains a UNET for the given data.
TrainUNET[trainData, validationData, {testData, testLabels}] Trains a UNET for the given data and also gives similarity results for the testData.
The inputs trainData, validationData, testData and testLabels can be generated using SplitTrainData."

SplitTrainData::usage = 
"SplitTrainData[data, label] splits the data and label in trainData, validationData, testData and testLabels that can be used in TrainUNET.
The data and label should be in the form {N, Nchan, x, y} or {N, Nchan, z, x, y}. The label sould be Integers with 1 for the background class and should go from 1 to Nclass."

ClassEncoder::usage = 
"ClassEncoder[label, nclass] encodes Integer label data of 1 to Ncalss into a Nclass vector of 1 and 0."

ClassDecoder::usage = 
"ClassDecoder[probability, nclass] decodes a probability vector of 1 and 0 into Integers of 1 to Nclass."

RotateFlip::usage = 
"RotateFlip[data] transforms one dataset into 8 by generating a mirrored version and rotation both 4x90 degree."

MakeClassImage::usage = 
"MakeClassImage[label] makes a images of the labels automatically scaled betweern the min and max label.
MakeClassImage[label, ratio] makes a images of the labels with aspectratio ratio.
MakeClassImage[label, {min, max}] makes a images of the labels automatically scaled betweern the min and max.
MakeClassImage[label, {min, max}, ratio] makes a images of the labels automatically scaled betweern the min and max with aspectratio ratio."

MakeChannelImage::usage = 
"MakeChannelImage[image] creates a row of the channels. The Input should be a list of 2D arrays.
MakeChannelImage[image, ratio] creates a row of the channels with aspectratio ratio." 

VisualizeUNET2D::usage = 
"VisualizeUNET2D[testData, trainedNet] visualises the hidden layers of a trained 2D UNET."

MakeDiffLabel::usage = 
"MakeDiffLabel[label, result] makes a label datasets with 1 = false positive, 2 = false negative, 3 = true positive."

ShowChannelClassData::usage =
"ShowChannelClassData[data, label] makes a grid of the data and label in 2D.
ShowChannelClassData[data, label, result] makes a grid of the data, label and result in 2D."

MakeNetPlots::usage = 
"MakeNetPlots[trainedNet]
MakeNetPlots[trainedNet, size]"


(* ::Subsection::Closed:: *)
(*Options*)


NetParameters::usage = 
"NetParameters is an option for TrainUNET. It Specifies the number of trainable parameters of the first layer of the UNET"

BlockType::usage = 
"BlockType is an option for TrainUNET and UNET. It specifies which block are used to build the network. Values can be \"UNET\" or \"ResNet\"."

DropOutRate::usage = 
"DropOutRate is an option for TrainUNET and UNET. It specifies how musch dropout is used after each block. It is a value between 0 and 1, default is .2."

RandomizeSplit::usage = 
"RandomizeSplit is an option for SplitTrainData. If True the data is randomized"

AugmentTrainData::usage =
"AugmentTrainData is an option for SplitTrainData. If True the train and validation data is augmented using RotateFlip. 
This increases the data by a factor 8 by generating a mirrored version and rotation both 4x90 degree."

SplitRatios::usage = 
"SplitRatios is an optino for SplitTrainData. Defines the ratios of the train validation and test data."

ClassScale::usage = 
"ClassScale is an options for ShowChannelClassData. Allows to scale the calss collors just as in MakeClassImage."

NumberRowItems::usage = 
"NumberRowItems is an options for ShowChannelClassData. Specifies how many images are on each row."

MakeDifferenceImage::usage = 
"MakeDifferenceImage is an options for ShowChannelClassData. If a result is provided this allos to show the diffrence between the label and result.
1 = false positive, 2 = false negative, 3 = true positive."

StepSize::usage = 
"StepSize is an options for ShowChannelClassData. It defines how many images are displayed by stepping through the data with stepsize."

NetLossLayers::usage = 
"NetLossLayers is an option for TrainUNET. It defines which loss layers to use default is ALL. Values are 1 - SoftDice, 2 - CrossEntropy, 3 - Brier. Can also be a combination, i.e. {1,2}."


(* ::Subsection::Closed:: *)
(*Error messages*)


TrainUNET::dim = 
"`1` is not a valid data dimension. Allowed dimension values are 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240 or 256."


(* ::Section:: *)
(*Unet Core Functionality*)


Begin["`Private`"]


verb = False;


(* ::Subsection:: *)
(*DICE*)


(* ::Subsubsection::Closed:: *)
(*DiceSimilarityClass*)


SyntaxInformation[DiceSimilarityClass] = {"ArgumentsPattern" -> {_, _, _}};

DiceSimilarityClass[pred_,gt_,nClasses_]:=Block[{predf,gtf},
	predf=Flatten[pred];
	gtf=Flatten[gt];
	Table[DiceSimilarity[predf,gtf,c],{c,nClasses}]
]


(* ::Subsubsection::Closed:: *)
(*DiceSimilarity*)


(*DiceSimilarity of two vetors*)
DiceSimilarity[v1_, v2_] := DiceSimilarityC[v1, v2, 1]


(*DiceSimiilartiy of a given class label*)
DiceSimilarity[v1_, v2_, c_] := DiceSimilarityC[v1, v2, c]


DiceSimilarityC = Compile[{{predi, _Integer, 1}, {gti, _Integer, 1}, {class, _Integer, 0}}, Block[{predv, gtv, denom},
    predv = Flatten[1 - Unitize[predi - class]];
    gtv = Flatten[1 - Unitize[gti - class]];
    denom = (Total[predv] + Total[gtv]);
    If[denom === 0., 1., N[2 Total[predv gtv]/denom]]
    ], RuntimeOptions -> "Speed"];


(* ::Subsection:: *)
(*LossLayers*)


(* ::Subsubsection::Closed:: *)
(*SoftDiceLossLayer*)


SyntaxInformation[SoftDiceLossLayer] = {"ArgumentsPattern" -> {_}};

SoftDiceLossLayer[dim_]:=NetGraph[<|
  "times" -> ThreadingLayer[Times],
  "flattot1" -> FlatTotLayer[dim - 1],
  "flattot2" -> FlatTotLayer[dim - 1],
  "flattot3" -> FlatTotLayer[dim - 1],
  "total" -> TotalLayer[],
  "devide" -> {ThreadingLayer[Divide], AggregationLayer[Mean, 1],ElementwiseLayer[1 - 2 # &]},
  "weight" -> ElementwiseLayer[1/(# + 1) &],
  "times1" -> ThreadingLayer[Times],
  "times2" -> ThreadingLayer[Times]
  |>, {
  {NetPort["Input"], NetPort["Target"]} -> "times" -> "flattot1",
  NetPort["Input"] -> "flattot2",
  NetPort["Target"] -> "flattot3",
  {"flattot2", "flattot3"} -> "total",
  "flattot3" -> "weight",
  {"flattot1", "weight"} -> "times1",
  {"total", "weight"} -> "times2",
  {"times1", "times2"} -> "devide" -> NetPort["Loss"]
  }, "Loss" -> "Real"]


FlatTotLayer[lev_]:=NetChain[{FlattenLayer[lev],AggregationLayer[Total,1]}];


(* ::Subsubsection::Closed:: *)
(*BrierLossLayer*)


SyntaxInformation[BrierLossLayer] = {"ArgumentsPattern" -> {_}};
   
BrierLossLayer[dim_] := NetGraph[<|
   "sub" -> ThreadingLayer[Subtract],
   "SqMn" -> {ElementwiseLayer[#^2 &], FlattenLayer[dim - 1], TransposeLayer[], AggregationLayer[Mean]},
   "tot1" -> AggregationLayer[Total, 1],
   "weigth" -> {FlatTotLayer[dim - 1], ElementwiseLayer[1/(# + 1) &]},
   "times" -> ThreadingLayer[Times],
   "tot1" -> AggregationLayer[Total, 1],
   "tot2" -> AggregationLayer[Total, 1],
   "devide" -> ThreadingLayer[Divide]
   |>, {
   {NetPort["Input"], NetPort["Target"]} -> "sub" -> "SqMn",
   NetPort["Target"] -> "weigth",
   {"weigth", "SqMn"} -> "times" -> "tot1",
   "weigth" -> "tot2",
   {"tot1", "tot2"} -> "devide" -> NetPort["Loss"]
   }, "Loss" -> "Real"]



(* ::Subsection::Closed:: *)
(*MakeNetPlots*)


MakeNetPlots[trained_, size_: 400] := Block[{n, pl1, pl2},
  n = Round[trained["TotalBatches"]/trained["TotalRounds"]];
  pl1 = ListLogPlot[{trained["ValidationLossList"][[All, 2]], 
     trained["BatchLossList"][[;; ;; n]]}, Joined -> True, 
    PlotLegends -> {"Validation", "Training"}, Frame -> True, 
    GridLines -> Automatic, FrameLabel -> {"Epochs", "Loss"}, 
    LabelStyle -> Directive[Black, Bold, 12], 
    FrameStyle -> Directive[Black, Thick], PlotStyle -> Thick, 
    ImageSize -> size];
  pl2 = ListLogPlot[
    100 {trained["ValidationErrorRateList"][[All, 2]], 
      trained["BatchErrorRateList"][[;; ;; n]]}, Joined -> True, 
    PlotLegends -> {"Validation", "Training"}, Frame -> True, 
    GridLines -> Automatic, 
    FrameLabel -> {"Epochs", "Error Rate [%]"}, 
    LabelStyle -> Directive[Black, Bold, 12], 
    FrameStyle -> Directive[Black, Thick], PlotStyle -> Thick, 
    ImageSize -> size];
  {pl1, pl2}
  ]


(* ::Subsection:: *)
(*UNET*)


(* ::Subsubsection::Closed:: *)
(*UNET*)


Options[MakeUNET] = {BlockType->"ResNet", DropOutRate->0.2}

SyntaxInformation[MakeUNET] = {"ArgumentsPattern" -> {_, _, _, _, OptionsPattern[]}};

MakeUNET[Nchan_,Nclass_,dep_,dimIn_,OptionsPattern[]]:=Switch[Length[dimIn],2,UNET2D,3,UNET3D][Nchan,Nclass,Floor[dep,2],dimIn,OptionValue[BlockType],OptionValue[DropOutRate]]


(* ::Subsubsection::Closed:: *)
(*General*)


layName[rep_] := "layer_" <> ToString[rep]

connect[dep_] := Flatten@Table[Switch[rep,
     1, {NetPort["Input"] -> layName[rep]}, dep + 1, Flatten[{NetPort["Input"], Table[layName[rr], {rr, 1, rep - 1}]}] -> "trans",
     _, Flatten[{NetPort["Input"], Table[layName[rr], {rr, 1, rep - 1}]}] -> layName[rep]
     ], {rep, 1, dep + 1}];


(* ::Subsubsection::Closed:: *)
(*UNet2D*)


UNET2D[NChan_:1,Nclass_:1,depI_:64,dimIn_:{128,128}, res_:"ResNet", drop_:0.2] := Block[{dep, depi},
 {dep, depi} = Switch[res,
 	"DenseNet", {Table[{depI[[1]], depI[[2]] i}, {i, {1, 2, 2, 3, 3}}], 4 depI[[1]]},
 	"UDenseNet", {Table[{depI[[1]], depI[[2]] i}, {i, {1, 2, 2, 3, 3}}], 4 depI[[1]]},
 	_, {Table[depI i, {i, {{1, 1}, {1, 2}, {2, 4}, {4, 8}, {8, 16}}}], depI}
   ];
 
 If[verb, Print["Layer parameters:" , {dep, depi}]];
 
 NetGraph[<|
   "start" -> convBN2[depi, 1],
   "enc_1" -> conv2[dep[[1]], dimIn, res, drop],
   "enc_2" -> {PoolingLayer[{2, 2}, 2], conv2[dep[[2]], dimIn/2, res, drop]},
   "enc_3" -> {PoolingLayer[{2, 2}, 2], conv2[dep[[3]], dimIn/4, res, drop]},
   "enc_4" -> {PoolingLayer[{2, 2}, 2], conv2[dep[[4]], dimIn/8, res, drop]},
   "enc_5" -> {PoolingLayer[{2, 2}, 2], conv2[dep[[5]], dimIn/16, res, drop]},
   "dec_1" -> dec2[dep[[5]], dimIn/8, res, drop],
   "dec_2" -> dec2[dep[[4]], dimIn/4, res, drop],
   "dec_3" -> dec2[dep[[3]], dimIn/2, res, drop],
   "dec_4" -> dec2[dep[[2]], dimIn, res, drop],
   "map" -> ConvolutionLayer[Nclass, {1, 1}],
   "prob" -> If[Nclass > 1, {TransposeLayer[{1 <-> 3, 1 <-> 2}], SoftmaxLayer[]}, {LogisticSigmoid, FlattenLayer[1]}]
   |>, {
   NetPort["Input"] -> "start" -> "enc_1" -> "enc_2" -> "enc_3" -> "enc_4" -> "enc_5",
   {"enc_4", "enc_5"} -> "dec_1",
   {"enc_3", "dec_1"} -> "dec_2",
   {"enc_2", "dec_2"} -> "dec_3",
   {"enc_1", "dec_3"} -> "dec_4",
   "dec_4" -> "map" -> "prob"
   }, "Input" -> Prepend[dimIn, NChan]]
 ]


(* ::Subsubsection::Closed:: *)
(*ConvBN2*)


convBN2[dep_, k_, r_: True] := Block[{p = (k - 1)/2, ch},
  ch = {ConvolutionLayer[dep, {k, k}, "PaddingSize" -> {p, p}], BatchNormalizationLayer[]};
  ch = If[r, Append[ch, ElementwiseLayer["ELU"]], ch];
  NetChain[ch]]


(* ::Subsubsection::Closed:: *)
(*Conv2*)


conv2[n_, dimIn_, res_, drop_] := Block[{k, dep, ni, no},
   Switch[res,
    "ResNet",
    {ni, no} = n;
    If[verb, Print["conv - dimensions", Prepend[dimIn, n]]];
    NetGraph[<|
      "con1" -> convBN2[no/2, 3], 
      "con2" -> convBN2[no, 3, False], 
      "skip" -> convBN2[no, 1, False], "tot" -> TotalLayer[],
      "elu" -> {ElementwiseLayer["ELU"], DropoutLayer[drop]}
      |>, {
      NetPort["Input"] -> "con1" -> "con2",
      NetPort["Input"] -> "skip",
      {"skip", "con2"} -> "tot" -> "elu" -> NetPort["Output"]},
     "Input" -> Prepend[dimIn, ni]
     ],
     
     "UResNet",(*same as resnet but without skip layer*)
    {ni, no} = n;
    If[verb, Print["conv - dimensions", Prepend[dimIn, n]]];
    NetChain[{convBN2[no/2, 3], convBN2[no, 3], DropoutLayer[drop]},"Input" -> Prepend[dimIn, ni]],
    
    "DenseNet",
    {k, dep} = n;
    If[verb, Print["conv - dimensions", Prepend[dimIn, 4 k]]];
    NetGraph[Association[Join[
       {"trans" -> {CatenateLayer["Inputs" -> Prepend[ConstantArray[Prepend[dimIn, k], dep], Prepend[dimIn, 4 k]]], convBN2[4 k, 1], DropoutLayer[drop]}},
       convLayers2[k, dep, dimIn]
       ]],
     connect[dep],
     "Input" -> Prepend[dimIn, 4 k]
     ],
     
    "UDenseNet",
    {k, dep} = n;
    If[verb, Print["conv - dimensions", Prepend[dimIn, 4 k]]];
    NetChain[Append[ConstantArray[convBN2[4 k, 3], dep], DropoutLayer[drop]],"Input" -> Prepend[dimIn, 4 k]],
    
    _,
    {ni, no} = n;
    If[verb, Print["conv - dimensions", Prepend[dimIn, n]]];
    NetChain[{convBN2[no, 3], convBN2[no, 3], DropoutLayer[drop]},"Input" -> Prepend[dimIn, ni]]
    ]
   ];


(* ::Subsubsection::Closed:: *)
(*ConvLayers2*)


convLayers2[k_, dep_, dimIn_] := Table[layName[rep] -> Switch[rep,
     1, convBN2[k, 3],
     _, {CatenateLayer["Inputs" -> Prepend[ConstantArray[Prepend[dimIn, k], rep - 1], Prepend[dimIn, 4 k]]], convBN2[4 k, 1], convBN2[k, 3]}
     ], {rep, 1, dep}];


(* ::Subsubsection::Closed:: *)
(*Dec2*)


dec2[ni_, dimIn_, res_, drop_] := Block[{n, n1, n2},
   (*determine in and output channels*)
   (*n is (ni,no) or (k,dep) for conv2, n1 is n input2 , n2 is n input2*)
   {n, n1, n2} = Switch[res,
   	 "DenseNet", {ni, 4 ni[[1]], 4 ni[[1]]}, 
   	 "UDenseNet", {ni, 4 ni[[1]], 4 ni[[1]]},
   	 _, {{Total[ni], ni[[1]]}, ni[[1]], ni[[2]]}];
   If[verb, Print["dec - dimensions and in/out par", {Prepend[dimIn, n1], n1, n2}]];
   (*the deconv graph*)
   NetGraph[
    <|
     "deconv" -> ResizeLayer[{Scaled[2], Scaled[2]}],
     "cat" -> CatenateLayer["Inputs" -> {Prepend[dimIn, n1], Prepend[dimIn, n2]}],
     "conv" -> Switch[res, 
     	"DenseNet", {convBN2[n2, 1], conv2[n, dimIn, res, drop]},
     	"UDenseNet", {convBN2[n2, 1], conv2[n, dimIn, res, drop]}, 
     	_, conv2[n, dimIn, res, drop]]
     |>, {
     NetPort["Input2"] -> "deconv",
     {NetPort["Input1"], "deconv"} -> "cat" -> "conv"},
    "Input1" -> Prepend[dimIn, n1], "Input2" -> Prepend[dimIn/2, n2]
    ]
   ];


(* ::Subsubsection::Closed:: *)
(*UNet3D*)


UNET3D[NChan_: 1, Nclass_: 1, depI_: 32, dimIn_: {32, 128, 128}, res_:"ResNet", drop_:0.2] := Block[{dep, depi},
 {dep, depi} = Switch[res,
   "DenseNet", {Table[{depI[[1]], depI[[2]] i}, {i, {1, 2, 2, 3, 3}}], 4 depI[[1]]},
 	"UDenseNet", {Table[{depI[[1]], depI[[2]] i}, {i, {1, 2, 2, 3, 3}}], 4 depI[[1]]},
   _, {Table[depI i, {i, {{1, 1}, {1, 2}, {2, 4}, {4, 8}, {8, 16}}}], depI}
   ];
 
 If[verb, Print["Layer parameters:" , {dep, depi}]]; 
 NetGraph[<|
 	"start" -> convBN3[depi, 1],
   "enc_1" -> conv3[dep[[1]], dimIn, res, drop],
   "enc_2" -> {PoolingLayer[{2, 2, 2}, 2], conv3[dep[[2]], dimIn/2, res, drop]},
   "enc_3" -> {PoolingLayer[{2, 2, 2}, 2], conv3[dep[[3]], dimIn/4, res, drop]},
   "enc_4" -> {PoolingLayer[{2, 2, 2}, 2], conv3[dep[[4]], dimIn/8, res, drop]},
   "enc_5" -> {PoolingLayer[{2, 2, 2}, 2], conv3[dep[[5]], dimIn/16, res, drop]},
   "dec_1" -> dec3[dep[[5]], dimIn/8, res, drop],
   "dec_2" -> dec3[dep[[4]], dimIn/4, res, drop],
   "dec_3" -> dec3[dep[[3]], dimIn/2, res, drop],
   "dec_4" -> dec3[dep[[2]], dimIn, res, drop],
   "map" -> ConvolutionLayer[Nclass, {1, 1, 1}],
   "prob" -> If[Nclass > 1, {TransposeLayer[{1 <-> 4, 1 <-> 3, 1 <-> 2}], SoftmaxLayer[]}, {LogisticSigmoid, FlattenLayer[1]}]
   |>, {
   NetPort["Input"] -> "start" -> "enc_1" -> "enc_2" -> "enc_3" -> "enc_4" -> "enc_5",
   {"enc_4", "enc_5"} -> "dec_1",
   {"enc_3", "dec_1"} -> "dec_2",
   {"enc_2", "dec_2"} -> "dec_3",
   {"enc_1", "dec_3"} -> "dec_4",
   "dec_4" -> "map" -> "prob"
   }, "Input" -> Prepend[dimIn, NChan]]
 ]


(* ::Subsubsection::Closed:: *)
(*ConvBN3*)


convBN3[dep_, k_, r_: True] := Block[{p = (k - 1)/2, ch},
  ch = {ConvolutionLayer[dep, {k, k, k}, "PaddingSize" -> {p, p, p}], BatchNormalizationLayer[]};
  ch = If[r, Append[ch, ElementwiseLayer["ELU"]], ch];
  NetChain[ch]
  ]


(* ::Subsubsection::Closed:: *)
(*Conv3*)


conv3[n_, dimIn_, res_, drop_] := Block[{k, dep, ni, no},
   Switch[res,
    "ResNet",
    {ni, no} = n;
    If[verb, Print["conv - dimensions", Prepend[dimIn, n]]];
    NetGraph[<|
      "con1" -> convBN3[no/2, 1], 
      "con2" -> convBN3[no, 3], 
      "skip" -> convBN3[no, 1, False], "tot" -> TotalLayer[],
      "elu" -> {ElementwiseLayer["ELU"], DropoutLayer[drop]}
      |>, {
      NetPort["Input"] -> "con1" -> "con2",
      NetPort["Input"] -> "skip",
      {"skip", "con2"} -> "tot" -> "elu" -> NetPort["Output"]},
     "Input" -> Prepend[dimIn, ni]
     ],

     "UResNet",(*same as resnet but without skip layer*)
    {ni, no} = n;
    If[verb, Print["conv - dimensions", Prepend[dimIn, n]]];
    NetChain[{convBN3[no/2, 3], convBN3[no, 3], DropoutLayer[drop]},"Input" -> Prepend[dimIn, ni]],
    
    "DenseNet",
    {k, dep} = n;
    If[verb, Print["conv - dimensions", Prepend[dimIn, 4 k]]];
    NetGraph[Association[Join[
       {"trans" -> {CatenateLayer["Inputs" -> Prepend[ConstantArray[Prepend[dimIn, k], dep], Prepend[dimIn, 4 k]]], convBN3[4 k, 1], DropoutLayer[drop]}},
       convLayers3[k, dep, dimIn]]
      ],
     connect[dep],
     "Input" -> Prepend[dimIn, 4 k]
     ],

    "UDenseNet",
    {k, dep} = n;
    If[verb, Print["conv - dimensions", Prepend[dimIn, 4 k]]];
    NetChain[Append[ConstantArray[convBN3[4 k, 3], dep], DropoutLayer[drop]],"Input" -> Prepend[dimIn, 4 k]],
    
    _,
    {ni, no} = n;
    If[verb, Print["conv - dimensions", Prepend[dimIn, n]]];
    NetChain[{convBN3[no, 3], convBN3[no, 3], DropoutLayer[drop]}, "Input" -> Prepend[dimIn, ni]]
    ]
   ];


(* ::Subsubsection::Closed:: *)
(*ConvLayers3*)


convLayers3[k_, dep_, dimIn_] := Table[layName[rep] -> Switch[rep,
     1, convBN3[k, 3],
     _, {CatenateLayer["Inputs" -> Prepend[ConstantArray[Prepend[dimIn, k], rep - 1], Prepend[dimIn, 4 k]]], convBN3[4 k, 1], convBN3[k, 3]}
     ], {rep, 1, dep}];


(* ::Subsubsection::Closed:: *)
(*Dec3*)


dec3[ni_, dimIn_, res_, drop_] := Block[{n, n1, n2},
   (*determine in and output channels*)
   (*n is (ni,no) or (k,dep) for conv2, n1 is n input2 , n2 is n input2*)
   {n, n1, n2} = Switch[res, 
   	"DenseNet", {ni, 4 ni[[1]], 4 ni[[1]]}, 
   	"UDenseNet", {ni, 4 ni[[1]], 4 ni[[1]]},
   	_, {{Total[ni], ni[[1]]}, ni[[1]], ni[[2]]}
   	];
   If[verb, Print["dec - dimensions and in/out par", {Prepend[dimIn, n1], n1, n2}]];
   NetGraph[<|
     "deconv" -> ResizeLayer3D[n2, dimIn/2],
     "cat" -> CatenateLayer["Inputs" -> {Prepend[dimIn, n1], Prepend[dimIn, n2]}],
     "conv" -> Switch[res, 
     	"DenseNet", {convBN3[n2, 1], conv3[n, dimIn, res, drop]}, 
     	"UDenseNet", {convBN3[n2, 1], conv3[n, dimIn, res, drop]}, 
     	_, conv3[n, dimIn, res, drop]]
     |>, {
     NetPort["Input2"] -> "deconv",
     {NetPort["Input1"], "deconv"} -> "cat" -> "conv"},
    "Input1" -> Prepend[dimIn, n1], "Input2" -> Prepend[dimIn/2, n2]
    ]
   ];


(* ::Subsubsection::Closed:: *)
(*ResizeLayer3D*)


ResizeLayer3D[n_, {dimInx_, dimIny_, dimInz_}] := Block[{sc = 2},
  NetChain[{
    FlattenLayer[1, "Input" -> {n, dimInx, dimIny, dimInz}],
    ResizeLayer[{Scaled[sc], Scaled[sc]}],
    ReshapeLayer[{n, dimInx, sc dimIny, sc dimInz}],
    TransposeLayer[2 <-> 3],
    FlattenLayer[1],
    ResizeLayer[{Scaled[sc], Scaled[1]}],
    ReshapeLayer[{n, sc dimIny, sc dimInx, sc dimInz}],
    TransposeLayer[2 <-> 3]
    }]
  ]


(* ::Subsection:: *)
(*Train Net*)


(* ::Subsubsection::Closed:: *)
(*Train UNET*)


Options[TrainUNET]=Join[{NetParameters->32, BlockType ->"ResNet", NetLossLayers->All, DropOutRate->0.2},Options[NetTrain]];

SyntaxInformation[TrainUNET] = {"ArgumentsPattern" -> {_, _, _., OptionsPattern[]}};

TrainUNET[train_, valid_, opt:OptionsPattern[]]:=TrainUNET[train, valid, {None, None}, opt]

TrainUNET[train_, valid_, {testData_, testLabel_}, opt:OptionsPattern[]]:=Block[{
	Nchan,Nclass,net,device,trained,netTrained,result,plots,iou, block, loss, drop,
	netDim,datDim,netPar,trainopt,lossNet,lossFunction},
	
	(*get the data dime*)
	datDim=Dimensions[train[[1,1]]][[2;;]];
	netDim=Length[datDim];
	(*Get the data Channels and classes*)
	Nchan=Length[train[[1,1]]];
	Nclass=Dimensions[train[[1, 2]]];
	Nclass=If[Length[Nclass]===netDim,1,Nclass[[-1]]];
	
	Print["channels: ",Nchan," - classes: ",Nclass, " - Dimensions :", datDim];
	
	(*get the function options*)
	netPar = OptionValue[NetParameters];
	device = OptionValue[TargetDevice];
	trainopt = Sequence@FilterRules[{opt},Options[NetTrain]];
	block = OptionValue[BlockType];
	drop = OptionValue[DropOutRate];
	
	loss = {"Loss1","Loss2","Loss3"}[[OptionValue[NetLossLayers]]];
	
	(*chekc if data dimensions are valid for net*)
	If[(!AllTrue[MemberQ[Range[20]2^4,#]&/@datDim,TrueQ])&&(MemberQ[{2,3},netDim]),
		(*not a valid dimension*)
		Return[Message[TrainUNET::dim, datDim]]
		,
		(*initialize and train net*)
		net=MakeUNET[Nchan,Nclass,netPar,datDim, BlockType -> block, DropOutRate -> drop];
		(*Attatch the loss funtion if needed*)
		{lossNet,lossFunction}=If[Nclass>1,{AddLossLayer[net,netDim],loss},{net,Automatic}];
		(*train the net*)
		trained = NetTrain[lossNet,train,All,TargetDevice->device,ValidationSet->valid,LossFunction->lossFunction,trainopt];
		
		(*extract the trained net*)
		{netTrained, plots} = If[Nclass>1,
			{
				NetExtract[trained["TrainedNet"],"net"],
				Row[MakeNetPlots[trained]]
				},
			{
				trained["TrainedNet"],
				trained["LossEvolutionPlot"]
				}
		];
		
		Print["Evaluating test data"];
		
		If[testData=!=None,
			(*test data provided*)
			result = netTrained[testData,TargetDevice->device];
			(*decode the data*)
			result = If[Nclass>1,ClassDecoder[result,Nclass],Round[result]];
			(*get the Dice or test data*)
			iou = DiceSimilarityClass[result,testLabel,Nclass];
			Print["DICE per class: ",Thread[Range[Nclass]->Round[iou,0.001]]];
			(*give Output*)
			{{lossNet, trained, netTrained}, {plots, result, iou}}
			,
			(*no test data*)
			{{lossNet, trained, netTrained}, plots}
		]
	]
]


(* ::Subsubsection::Closed:: *)
(*AddLossLayer*)


AddLossLayer[net_,dim_]:=NetGraph[<|
		"net"->net,
		"SoftDice"->SoftDiceLossLayer[dim],
		"CrossEntr"->CrossEntropyLossLayer["Probabilities"],
		"Brier"->BrierLossLayer[dim]
	|>,{
		NetPort["Input"]->"net"->NetPort["Output"],
		{"net",NetPort["Target"]}->"SoftDice"->NetPort["Loss1"],
		{"net",NetPort["Target"]}->"CrossEntr"->NetPort["Loss2"],
		{"net",NetPort["Target"]}->"Brier"->NetPort["Loss3"]
}]


(* ::Subsubsection::Closed:: *)
(*ClassEndocer*)


SyntaxInformation[ClassEncoder] = {"ArgumentsPattern" -> {_, _}};

ClassEncoder[data_,NClass_]:=Map[NetEncoder[{"Class",Range[NClass],"UnitVector"}],data,{ArrayDepth[data]-1}]


(* ::Subsubsection::Closed:: *)
(*ClassDecoder*)


SyntaxInformation[ClassDecoder] = {"ArgumentsPattern" -> {_, _}};

ClassDecoder[data_,NClass_]:=NetDecoder[{"Class","Labels"->Range[1,NClass],"InputDepth"->ArrayDepth[data]}][data]


(* ::Subsection:: *)
(*Prepare Data*)


(* ::Subsubsection::Closed:: *)
(*SplitTestData*)


Options[SplitTrainData]={RandomizeSplit->True,SplitRatios->{0.7,.2,.1}, AugmentTrainData->False};

SyntaxInformation[SplitTrainData] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}};

SplitTrainData[data_,label_,OptionsPattern[]]:=Block[{
	allData, train, valid, test, testData, testLabel, datas1, datas2, labels1, labels2, 
	Nclass, dim, s1, s2, s3, order, ratio, rand, aug, augi},
	
	Print["Dimensions data: ", Dimensions[data]," - Dimensions label: ", Dimensions[label]];
	
	(*get the options*)
	rand=OptionValue[RandomizeSplit];
	ratio=OptionValue[SplitRatios];
	
	(*split data*)
	order=Range[Length[data]];
	order=If[rand,RandomSample[order],order];
	
	(*get the ratios*)
	{s1,s2,s3}=Accumulate@Round[ratio Length[data]];
	{s1,s2,s3}={order[[1;;s1]],order[[s1+1;;s2]],order[[s2+1;;]]};
	Print["Nuber of Samples in each set: ",Length/@{s1,s2,s3}];
	
	(*Encode Data*)
	Nclass=Max[label];
	dim=ArrayDepth[label]-1;
	
	aug = OptionValue[AugmentTrainData];
	aug = If[aug === True || aug === 1 || aug === 2, True, False];
	augi = aug /. True -> 1;
	
	(*data augmentation*)
	{datas1, datas2} = If[aug,
		{RotateFlip[data[[s1]], augi],RotateFlip[data[[s2]], augi]},
		{data[[s1]],data[[s2]]}];
		
	{labels1, labels2} = If[aug,
		{RotateFlip[label[[s1]], augi],RotateFlip[label[[s2]], augi]},
		{label[[s1]],label[[s2]]}];
		
	If[aug, Print["Nuber of Samples in each set afeter augmentation: ",{8,8,1} Length/@{s1,s2,s3}]];
	
	(*make training validation and test data*)
	If[Nclass>1,
		train = Thread[datas1->ClassEncoder[labels1,Nclass]];
		valid = Thread[datas2->ClassEncoder[labels2,Nclass]];
		,
		train = Thread[datas1->labels1];
		valid = Thread[datas2->labels2];
	];
	testData=data[[s3]];
	testLabel=label[[s3]];
	
	(*define the output*)
	If[rand,
		{train,valid,testData,testLabel,{s1,s2,s3}},
		{train,valid,testData,testLabel}
	]
]


(* ::Subsubsection::Closed:: *)
(*Rotate Flip data*)


SyntaxInformation[RotateFlip] = {"ArgumentsPattern" -> {_}};

RotateFlip[images_] := RotateFlip[images, 1]

RotateFlip[images_, met_] := Block[{dep = ArrayDepth[images] - 2}, Flatten[Map[RotateFlip[#,met]&, images, {dep}], {1, dep + 1}]]

RotateFlip[image_?MatrixQ] := RotateFlip[image, 1]

RotateFlip[image_?MatrixQ, met_] := Block[{ima,imb},
	ima=image;
	imb=Reverse[ima,{1}];
	Switch[met,
		1,
		{ima,Reverse[ima,{1,2}],Reverse[Transpose@ima,{2}],Reverse[Transpose@ima,{1}],imb,Reverse[imb,{1,2}],Reverse[Transpose@imb,{1}],Reverse[Transpose@imb,{2}]},
		2,
		{ima,Reverse[ima,{1,2}],Reverse[Transpose@ima,{2}],Reverse[Transpose@ima,{1}]}
	]
]


(* ::Subsection:: *)
(*Visualisation*)


(* ::Subsubsection::Closed:: *)
(*Make Class Images*)


SyntaxInformation[MakeClassImage] = {"ArgumentsPattern" -> {_, _., _.}};

MakeClassImage[label_] := MakeClassImage[label, MinMax[label], 1]

MakeClassImage[label_, {off_,max_}] := MakeClassImage[label, {off, max}, 1]

MakeClassImage[label_, ratio_] := MakeClassImage[label, MinMax[label], ratio]

MakeClassImage[label_, {off_,max_}, ratio_] := ImageResize[
	Colorize[Image[((label-off)/(max-off))], ColorFunction->"Rainbow", ColorFunctionScaling->False],
	{Max[Dimensions[label]],ratio Max[Dimensions[label]]}
]


(* ::Subsubsection::Closed:: *)
(*Make Diff Images*)


SyntaxInformation[MakeChannelImage] = {"ArgumentsPattern" -> {_, _., _.}};

MakeChannelImage[data_] := MakeChannelImage[data, 1]

MakeChannelImage[data_, ratio_]:=Block[{dat},
	(dat=#;
	ImageResize[Image[#/Max[#]],{Max[Dimensions[dat]],ratio Max[Dimensions[dat]]}]
	)&/@data]


(* ::Subsubsection::Closed:: *)
(*Make Diff Label*)


SyntaxInformation[MakeDiffLabel] = {"ArgumentsPattern" -> {_, _}};

MakeDiffLabel[label_, result_] := If[ArrayDepth[label] == ArrayDepth[result] == 2,
	MakeDiffLabel2D[label, result],
	If[ArrayDepth[label] == ArrayDepth[result] == 3,
		MakeDiffLabel3D[label, result],
		$Failed
	]
]


(* ::Subsubsection::Closed:: *)
(*MakeDiffLabel2D*)


MakeDiffLabel2D = Compile[{{lab, _Integer, 2}, {res, _Integer, 2}}, Block[{resU, labU, tp, fp, fn},
    resU = Unitize[res - Min[lab]];
    labU = Unitize[lab - Min[lab]];
    
    tp = labU (1 - Unitize[lab - res]);
    fp = Clip[resU - labU, {0, 1}];
    fn = labU - tp;
    
    1 fp + 2 tp + 3 fn]
   , RuntimeAttributes -> {Listable}, RuntimeOptions -> "Speed"
];


(* ::Subsubsection::Closed:: *)
(*MakeDiffLabel3D*)


MakeDiffLabel3D = Compile[{{lab, _Integer, 3}, {res, _Integer, 3}}, Block[{resU, labU, tp, fp, fn},
    resU = Unitize[res - Min[lab]];
    labU = Unitize[lab - Min[lab]];
    
    tp = labU (1 - Unitize[lab - res]);
    fp = Clip[resU - labU, {0, 1}];
    fn = labU - tp;
    
    1 fp + 2 tp + 3 fn]
   , RuntimeAttributes -> {Listable}, RuntimeOptions -> "Speed"
];



(* ::Subsubsection::Closed:: *)
(*ShowChannelClassData*)


Options[ShowChannelClassData]={ImageSize->500, ClassScale->Automatic, NumberRowItems->3, MakeDifferenceImage->False, StepSize->1, AspectRatio->1};

ShowChannelClassData[data_, label_, opts:OptionsPattern[]] := ShowChannelClassData[data, label, None, opts]

ShowChannelClassData[data_, label_, result_, OptionsPattern[]]:=Block[{minmax,diff,step,lab,dat,res, ratio},
	(*get the label scaling*)
	minmax = If[OptionValue[ClassScale]===Automatic,MinMax[label],OptionValue[ClassScale]];
	ratio = OptionValue[AspectRatio];
	
	(*part the data*)
	step=OptionValue[StepSize];
	lab=label[[;;;;step]];
	dat=data[[;;;;step]];
	
	(*make result and diff images if needed*)
	If[result=!=None,
		res=result[[;;;;step]];
		diff=If[OptionValue[MakeDifferenceImage],
			MakeClassImage[#,{0,3},ratio]&/@MakeDiffLabel[lab,res],
			Nothing
		];
		res=MakeClassImage[#,minmax,ratio]&/@res;
		,
		diff=res=Nothing;
	];
	
	(*make the image*)
	lab=MakeClassImage[#, minmax, ratio]&/@lab;
	dat=MakeChannelImage[#, ratio]&/@dat;

	(*make the output*)
	Grid[Partition[GraphicsRow[Flatten[#],ImageSize->OptionValue[ImageSize]]&/@Thread[{dat,lab,res,diff}],OptionValue[NumberRowItems]],Frame->All]
]


(* ::Subsubsection::Closed:: *)
(*Visualize UNET 2D*)


SyntaxInformation[VisualizeUNET2D] = {"ArgumentsPattern" -> {_, _}};

VisualizeUNET2D[dataI_,net_]:=Manipulate[
	(*get the data slice*)
	data=dataI[[m]];
	
	(*size=Max[Round[Sqrt[Length[data]]],2];*)
	size=Round[Sqrt[Length[data]]];
	datIm=ImageAssemble[Partition[Image[#/Max[#]]&/@data,size,size,1,Image[0 data[[1]]]]];
	
	Row[{Image[datIm,ImageSize->{500,500}],Colorize[images[[n]],ColorFunction->col,ImageSize->{500,500}]}]

	,
	(*fixed parameters*)
	{{nodes,{"enc_1","enc_2","enc_3","enc_4","enc_5","dec_1","dec_2","dec_3","dec_4","map","prob"}}, ControlType->None},
	{data,ControlType->None},
	{datIm,ControlType->None},
	{images,ControlType->None},
	
	(*dynamic parameters*)
	{{m,1,"image"},1,Length[dataI],1},
	Button["visualize net",	images=VisualizeNetIm[data,net,nodes]],
	Delimiter,
	{{n,1,"layer"},Thread[Range[11]->nodes],ControlType->SetterBar},
	{{col,"TemperatureMap","color"},{"Rainbow","TemperatureMap",(Blend[{RGBColor[1,1,1],RGBColor[0,0,0]},1-#]&)->"Grayscale"}},
	(*initialize net image*)
	Initialization :> (images = ConstantArray[Image[ConstantArray[0, {2, 2}]],11])	
]


(* ::Subsubsection::Closed:: *)
(*VisualizeNetIm*)


VisualizeNetIm[data_,net_,nodes_]:=Block[{out,size,port},(
		out=Take[net,{NetPort["Input"],#}][data];
		out=If[#==="prob", If[Length@Dimensions@out>2,TransData[out,"r"],{out}]	,out];
		size=Round[Sqrt[Length[out]]];
		ImageAssemble[Partition[Image[#]&/@out,size,size,1,Image[0 out[[1]]]]]
	)&/@nodes
]


TransData[data_,dir_]:=Block[{ran,dep,fun},
	ran=Range[dep=ArrayDepth[data]];
	fun=Switch[dir,"r",RotateLeft[ran],"l",RotateRight[ran]];
	Transpose[data,fun]
]


(* ::Section:: *)
(*End Package*)


End[]

EndPackage[]
