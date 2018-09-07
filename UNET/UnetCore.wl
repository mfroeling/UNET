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


DiceSimilarityClass::usage = "
DiceSimilarityClass[prediction, groundTruth, nclasses] gives the Dice Similarity between of each of Nclasses between prediction and groundTruth."

DiceSimilarity::usage = 
"DiceSimilarity[x, y] gives the Dice Similarity between 1 and 0 vectors x and y.
DiceSimilarity[x, y, class] gives the Dice Similarity Integer vectors x and y for Integer Class."

UNET::usage = 
"UNET[nchan, nclass, dep, dimIn] Generates a UNET with nchan as input and nclass as output. The number of parameter of the first convolution layer can be set with dep.
The data dimensions can be 2D or 3D and each of the dimensions should be 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240 or 256."

AddLossLayer::usage = 
"AddLossLayer[net] adds two loss layers to a NetGraph, a SoftDiceLossLayer and a CrossEntropyLossLayer."

SoftDiceLossLayer::usage = 
"SoftDiceLossLayer[dim] represents a net layer that computes the SoftDice loss by comparing input class probability vectors with the target class vector."

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


(* ::Subsection::Closed:: *)
(*Options*)


NetParameters::usage = 
"NetParameters is an option for TrainUNET. It Specifies the number of trainable parameters of the first layer of the UNET"

BlockType::usage = 
"BlockType is an option for TrainUNET and UNET. It specifies which block are used to build the network. Values can be \"UNET\" or \"ResNet\"."

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


(* ::Subsection::Closed:: *)
(*Error messages*)


TrainUNET::dim = 
"`1` is not a valid data dimension. Allowed dimension values are 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240 or 256."


(* ::Section:: *)
(*Unet Core Functionality*)


Begin["`Private`"]


(* ::Subsection:: *)
(*DICE*)


(* ::Subsubsection::Closed:: *)
(*DICE*)


SyntaxInformation[DiceSimilarityClass] = {"ArgumentsPattern" -> {_, _, _}};

DiceSimilarityClass[pred_,gt_,nClasses_]:=Block[{predf,gtf},
	predf=Flatten[pred];
	gtf=Flatten[gt];
	Table[DiceSimilarity[predf,gtf,c],{c,nClasses}]
]


(*DiceSimilarity of two vetors*)
DiceSimilarity[v1_ , v2_] := DiceSimilarity1[v1, v2]

DiceSimilarity1 = Compile[{{predi, _Integer, 1}, {gti, _Integer, 1}}, Block[
	{predv, gtv, denom},
    denom = (Total[predi] + Total[gti]);
    If[denom === 0., 1., N[2 Total[predi gti]/denom]]
	], RuntimeOptions -> "Speed"
];


(*DiceSimiilartiy of a given class label*)
DiceSimilarity[v1_, v2_, c_] := DiceSimilarity2[v1, v2, c]

DiceSimilarity2 = Compile[{{predi, _Integer, 1}, {gti, _Integer, 1}, {class, _Integer, 0}}, Block[
	{predv, gtv, denom}, 
    predv = 1 - Unitize[predi - class];
    gtv = 1 - Unitize[gti - class];
    denom = (Total[predv] + Total[gtv]);
    If[denom === 0., 1., N[2 Total[predv gtv]/denom]]
	], RuntimeOptions -> "Speed"
	];


(* ::Subsubsection::Closed:: *)
(*SoftDiceLossLayer*)


SyntaxInformation[SoftDiceLossLayer] = {"ArgumentsPattern" -> {_}};

SoftDiceLossLayer[dim_]:=NetGraph[<|
		"times"->ThreadingLayer[Times],
		"flattot1"->{ElementwiseLayer[2*#&],FlatTotLayer[dim-1]},
		"flattot2"->FlatTotLayer[dim-1],
		"flattot3"->FlatTotLayer[dim-1],
		"total1"->TotalLayer[],
		"devide"->{ThreadingLayer[Divide],AggregationLayer[Mean,1],ElementwiseLayer[1-#&]}
	|>,
	{
		{NetPort["Input"],NetPort["Target"]}->"times"->"flattot1",
		NetPort["Input"]->"flattot2",
		NetPort["Target"]->"flattot3",
		{"flattot2","flattot3"}->"total1",
		{"flattot1","total1"}->"devide"->NetPort["Loss"]
	},"Loss"->"Real"
]

FlatTotLayer[lev_]:=NetChain[{FlattenLayer[lev],AggregationLayer[Total,1]}];


(* ::Subsection:: *)
(*UNET*)


(* ::Subsubsection::Closed:: *)
(*UNET*)


Options[UNET] = {BlockType->"ResNet"}

SyntaxInformation[UNET] = {"ArgumentsPattern" -> {_, _, _, _, OptionsPattern[]}};

UNET[Nchan_,Nclass_,dep_,dimIn_,OptionsPattern[]]:=Switch[Length[dimIn],2,UNET2D,3,UNET3D][Nchan,Nclass,Floor[dep,2],dimIn,OptionValue[BlockType]]


(* ::Subsubsection::Closed:: *)
(*Unet2D*)


UNET2D[NChan_:1,Nclass_:1,dep_:64,dimIn_:{128,128}, res_:"ResNet"] := NetGraph[<|
		"start" -> convBN2[dep, 1],
		"enc_1"->conv2[dep, res],
		"enc_2"->{PoolingLayer[{2, 2}, 2], conv2[2 dep, res]},
		"enc_3"->{PoolingLayer[{2, 2}, 2], conv2[4 dep, res]},
		"enc_4"->{PoolingLayer[{2, 2}, 2], conv2[8 dep, res]},
		"enc_5"->{PoolingLayer[{2, 2}, 2], conv2[16 dep, res]},
		"dec_1"->dec2[8 dep, res],
		"dec_2"->dec2[4 dep, res],
		"dec_3"->dec2[2 dep, res],
		"dec_4"->dec2[dep, res],
		"map"->ConvolutionLayer[Nclass,{1,1}],
		"prob"->If[Nclass>1,{TransposeLayer[{1<->3,1<->2}],SoftmaxLayer[]},{LogisticSigmoid,FlattenLayer[1]}]
	|>,{
		NetPort["Input"]->"start"->"enc_1"->"enc_2"->"enc_3"->"enc_4"->"enc_5",
		{"enc_4","enc_5"}->"dec_1", {"enc_3","dec_1"}->"dec_2",
		{"enc_2","dec_2"}->"dec_3",	{"enc_1","dec_3"}->"dec_4",
		"dec_4"->"map"->"prob"
	},"Input"->Prepend[dimIn,NChan]
]


convBN2[dep_, k_, r_: True] := Block[{p = (k - 1)/2, ch},
  ch = {ConvolutionLayer[dep, {k, k}, "PaddingSize" -> {p, p}], BatchNormalizationLayer[]};
  ch = If[r, Append[ch, ElementwiseLayer["ELU"]], ch];
  NetChain[ch]]


conv2[n_, res_] := Switch[res,
   "ResNet",
   NetGraph[<|
	     "con1" -> convBN2[n/2, 1], "con2" -> convBN2[n/2, 3], "con3" -> convBN2[n, 1, False],
	     "skip" -> convBN2[n, 1, False], "tot" -> TotalLayer[], "elu" -> {ElementwiseLayer["ELU"], DropoutLayer[0.2]}
     |>, {
    	NetPort["Input"] -> "con1" -> "con2" -> "con3", NetPort["Input"] -> "skip", 
    	{"skip", "con3"} -> "tot" -> "elu" -> NetPort["Output"]
     }],
   _,
   NetChain[{convBN2[n, 3], convBN2[n, 3], DropoutLayer[0.2]}]
];


dec2[n_, res_] := NetGraph[
   <|"deconv" -> ResizeLayer[{Scaled[2], Scaled[2]}], "cat" -> CatenateLayer[], "conv" -> conv2[n, res]|>,
   {NetPort["Input1"] -> "cat", NetPort["Input2"] -> "deconv" -> "cat" -> "conv"}
   ];


(* ::Subsubsection::Closed:: *)
(*UNET3D*)


UNET3D[NChan_: 1, Nclass_: 1, dep_: 32, dimIn_: {32, 128, 128}, res_:"ResNet"] := NetGraph[<|
  "start" -> convBN3[dep, 1],
  "enc_1" -> conv3[dep, res],
  "enc_2" -> {PoolingLayer[{2, 2, 2}, 2], conv3[2 dep, res]},
  "enc_3" -> {PoolingLayer[{2, 2, 2}, 2], conv3[4 dep, res]},
  "enc_4" -> {PoolingLayer[{2, 2, 2}, 2], conv3[8 dep, res]},
  "enc_5" -> {PoolingLayer[{2, 2, 2}, 2], conv3[16 dep, res]},
  "dec_1" -> dec3[8 dep, dimIn/16, res],
  "dec_2" -> dec3[4 dep, dimIn/8, res],
  "dec_3" -> dec3[2 dep, dimIn/4, res],
  "dec_4" -> dec3[dep, dimIn/2, res],
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


convBN3[dep_, k_, r_: True] := Block[{p = (k - 1)/2, ch},
  ch = {ConvolutionLayer[dep, {k, k, k}, "PaddingSize" -> {p, p, p}], BatchNormalizationLayer[]};
  ch = If[r, Append[ch, ElementwiseLayer["ELU"]], ch];
  NetChain[ch]
  ]

conv3[n_, res_] := Switch[res,
  "ResNet",
  NetGraph[<|
    	"con1" -> convBN3[n/2, 1], "con2" -> convBN3[n/2, 3], "con3" -> convBN3[n, 1, False],
    	"skip" -> convBN3[n, 1, False], "tot" -> TotalLayer[], "elu" -> {ElementwiseLayer["ELU"], DropoutLayer[0.2]}
    |>, {
    	NetPort["Input"] -> "con1" -> "con2" -> "con3", NetPort["Input"] -> "skip", 
   		{"skip", "con3"} -> "tot" -> "elu" -> NetPort["Output"]
    }],
  _,
  NetChain[{convBN3[n, 3], convBN3[n, 3], DropoutLayer[0.2]}]
  ];


dec3[n_, dimIn_, res_] := NetGraph[
   <|"deconv" -> ResizeLayer3D[n, dimIn], "cat" -> CatenateLayer[], "conv" -> conv3[n, res]|>,
   {NetPort["Input1"] -> "cat", NetPort["Input2"] -> "deconv" -> "cat" -> "conv"}
   ];


ResizeLayer3D[n_, {dimInx_, dimIny_, dimInz_}] := Block[{sc = 2},
  NetChain[{
    FlattenLayer[1, "Input" -> {n sc, dimInx, dimIny, dimInz}],
    ResizeLayer[{Scaled[sc], Scaled[sc]}],
    ReshapeLayer[{n sc, dimInx, sc dimIny, sc dimInz}],
    TransposeLayer[2 <-> 3],
    FlattenLayer[1],
    ResizeLayer[{Scaled[sc], Scaled[1]}],
    ReshapeLayer[{n sc, sc dimIny, sc dimInx, sc dimInz}],
    TransposeLayer[2 <-> 3]
    }]
  ]


(* ::Subsection:: *)
(*Train Net*)


(* ::Subsubsection::Closed:: *)
(*Train UNET*)


Options[TrainUNET]=Join[{NetParameters->32, BlockType ->"ResNet"},Options[NetTrain]];

SyntaxInformation[TrainUNET] = {"ArgumentsPattern" -> {_, _, _., OptionsPattern[]}};

TrainUNET[train_, valid_, opt:OptionsPattern[]]:=TrainUNET[train, valid, {None, None}, opt]

TrainUNET[train_, valid_, {testData_, testLabel_}, opt:OptionsPattern[]]:=Block[{
	Nchan,Nclass,net,device,trained,netTrained,result,plots,iou, block,
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
	
	(*chekc if data dimensions are valid for net*)
	If[(!AllTrue[MemberQ[Range[20]2^4,#]&/@datDim,TrueQ])&&(MemberQ[{2,3},netDim]),
		(*not a valid dimension*)
		Return[Message[TrainUNET::dim, datDim]]
		,
		(*initialize and train net*)
		net=UNET[Nchan,Nclass,netPar,datDim, BlockType -> block];
		(*Attatch the loss funtion if needed*)
		{lossNet,lossFunction}=If[Nclass>1,{AddLossLayer[net,netDim],{"Loss1","Loss2"}},{net,Automatic}];
		(*train the net*)
		trained = NetTrain[lossNet,train,All,TargetDevice->device,ValidationSet->valid,LossFunction->lossFunction,trainopt];
		
		(*extract the trained net*)
		{netTrained, plots} = If[Nclass>1,
			{
				NetExtract[trained["TrainedNet"],"net"],
				Row[{trained["LossEvolutionPlot"],trained["ErrorRateEvolutionPlot"]}]
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
		"loss1"->SoftDiceLossLayer[dim],
		"loss2"->CrossEntropyLossLayer["Probabilities"]
	|>,{
		NetPort["Input"]->"net"->NetPort["Output"],
		{"net",NetPort["Target"]}->"loss1"->NetPort["Loss1"],
		{"net",NetPort["Target"]}->"loss2"->NetPort["Loss2"]
}]


(* ::Subsubsection::Closed:: *)
(*Decoders and Encoders*)


SyntaxInformation[ClassEncoder] = {"ArgumentsPattern" -> {_, _}};

ClassEncoder[data_,NClass_]:=Map[NetEncoder[{"Class",Range[NClass],"UnitVector"}],data,{ArrayDepth[data]-1}]


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
	{train,valid,testData,testLabel}
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


(* ::Subsubsection:: *)
(*Make Diff Label*)


SyntaxInformation[MakeDiffLabel] = {"ArgumentsPattern" -> {_, _}};

MakeDiffLabel[label_, result_] := If[ArrayDepth[label] == ArrayDepth[result] == 2,
	MakeDiffLabel2D[label, result],
	If[ArrayDepth[label] == ArrayDepth[result] == 3,
		MakeDiffLabel3D[label, result],
		$Failed
	]
]


MakeDiffLabel2D = Compile[{{lab, _Integer, 2}, {res, _Integer, 2}}, Block[{resU, labU, tp, fp, fn},
    resU = Unitize[res - Min[lab]];
    labU = Unitize[lab - Min[lab]];
    
    tp = labU (1 - Unitize[lab - res]);
    fp = Clip[resU - labU, {0, 1}];
    fn = labU - tp;
    
    1 fp + 2 tp + 3 fn]
   , RuntimeAttributes -> {Listable}, RuntimeOptions -> "Speed"
];


MakeDiffLabel3D = Compile[{{lab, _Integer, 3}, {res, _Integer, 3}}, Block[{resU, labU, tp, fp, fn},
    resU = Unitize[res - Min[lab]];
    labU = Unitize[lab - Min[lab]];
    
    tp = labU (1 - Unitize[lab - res]);
    fp = Clip[resU - labU, {0, 1}];
    fn = labU - tp;
    
    1 fp + 2 tp + 3 fn]
   , RuntimeAttributes -> {Listable}, RuntimeOptions -> "Speed"
];

(* ::Subsubsection:: *)
(*Make Diff Images*)


Options[ShowChannelClassData]={ImageSize->500,ClassScale->Automatic, NumberRowItems->3,MakeDifferenceImage->False,StepSize->1, AspectRatio->1};

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
