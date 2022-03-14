(* ::Package:: *)

(* ::Title:: *)
(*UNET*)


(* ::Subtitle:: *)
(*Written by: Martijn Froeling, PhD*)
(*m.froeling@gmail.com*)


(* ::Section:: *)
(*start Package*)


BeginPackage["UNET`UnetCore`", Join[{"Developer`"}, Complement[UNET`$Contexts, {"UNET`UnetCore`"}]]]


(* ::Section:: *)
(*Usage Notes*)


(* ::Subsection:: *)
(*Functions*)


DiceSimilarityClass::usage = 
"DiceSimilarityClass[prediction, groundTruth, nclasses] gives the Dice Similarity between of each of Nclasses between prediction and groundTruth. 
nClasses can also be a list of class number for which the Dice needs to be calculated."

DiceSimilarity::usage = "DiceSimilarity[x, y] gives the Dice Similarity between 1 and 0 of vectors x and y for class 1.
DiceSimilarity[x, y, class] gives the Dice Similarity for vectors x and y for Integer Class."

MakeUNET::usage = 
"MakeUNET[nchan, nclass, dep, dimIn] Generates a UNET with nchan as input and nclass as output. The number of parameter of the first convolution layer can be set with dep.\n
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
"MakeNetPlots[trainedNet] makes the loss and error plots after training.
MakeNetPlots[trainedNet, size] same but with specifying image size."

TrainFunction::usage=
"TrainFunction[train, b, aug ,n] Generates a random training batch from training data train with batchsize b and number of classes n. 
It allows for autmenting traing data by specifying aug for which default is {False, False, False}. First input is rotation which is \"90\" or a angel which rotates randomly between -angle and +angle.
Second input is scaling, for 20 procent up and downscaling input is {0.8, 1.2}. Last input is flipping which can be True or False."


(* ::Subsection:: *)
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
	{"flattot2", "flattot3"} -> "total", "flattot3" -> "weight",
	{"flattot1", "weight"} -> "times1",
	{"total", "weight"} -> "times2",
	{"times1", "times2"} -> "devide" -> NetPort["Loss"]
}, "Loss" -> "Real"]


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


(* ::Subsubsection::Closed:: *)
(*FlatTotLayer*)


FlatTotLayer[lev_]:=NetChain[{FlattenLayer[lev],AggregationLayer[Total,1]}];


(* ::Subsection::Closed:: *)
(*MakeNetPlots*)


MakeNetPlots[trained_, size_: 400] := Block[{n, pl1, pl2, d1, d2},
	d1 = {trained["ValidationLossList"], trained["RoundLossList"]};
	d2 = {trained["ValidationMeasurementsLists"]["ErrorRate"], trained["RoundMeasurementsLists"]["ErrorRate"]};
	
	pl1 = ListLogPlot[d1, Joined -> True,
		PlotLegends -> Placed[{"Validation", "Training"},Below], Frame -> True,
		GridLines -> Automatic, FrameLabel -> {"Epochs", "Loss"},
		LabelStyle -> Directive[Black, Bold, 12],
		FrameStyle -> Directive[Black, Thick], PlotStyle -> Thick,
		ImageSize -> size];
	pl2 = ListLogPlot[100 d2, Joined -> True,
		PlotLegends -> Placed[{"Validation", "Training"},Below], Frame -> True,
		GridLines -> Automatic, FrameLabel -> {"Epochs", "Error Rate [%]"},
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

MakeUNET[Nchan_,Nclass_,dep_,dimIn_,OptionsPattern[]]:=UNET[Nchan, Nclass, dep, dimIn, OptionValue[BlockType], OptionValue[DropOutRate]]


(* ::Subsubsection:: *)
(*UNet*)


UNET[NChan_, Nclass_, depI_, dimIn_, res_:"ResNet", drop_] := Block[{dep, pool, map},
	
	(*make the layer feature list*)
	dep = If[IntegerQ[depI]&&(res==="DenseNet" || res==="UDenseNet"),{depI,1}, depI];
	dep = Switch[res,
		"DenseNet" | "UDenseNet", Table[{dep[[1]], dep[[2]] + i -1}, {i, {1, 2, 4, 6, 8}}],
		_, Table[dep i, {i, {1, 2, 4, 8, 16}}]];
	If[verb, Print["Layer parameters: ", dep]];
	
	(*define dimention dependant *)
	map = Prepend[If[Nclass > 1, 
		{TransposeLayer[{{1 <-> 3, 1 <-> 2},{1 <-> 4, 1 <-> 3, 1 <-> 2}}[[Length[dimIn]-1]]], SoftmaxLayer[]}, 
		{LogisticSigmoid, FlattenLayer[1]}], ConvolutionLayer[Nclass, 1]];
	
	(*make the network*)	
	NetGraph[<|
		"start" -> CBE[dep[[1]], 1],
		"enc_1" -> enCode[dep[[1]], dimIn, res, drop, False],
		"enc_2" -> enCode[dep[[2]], dimIn/2, res, drop, True],
		"enc_3" -> enCode[dep[[3]], dimIn/4, res, drop, True],
		"enc_4" -> enCode[dep[[4]], dimIn/8, res, drop, True],
		"enc_5" -> enCode[dep[[5]], dimIn/16, res, drop, True],
		"dec_1" -> deCode[dep[[4]], dimIn/8, res, drop],
		"dec_2" -> deCode[dep[[3]], dimIn/4, res, drop],
		"dec_3" -> deCode[dep[[2]], dimIn/2, res, drop],
		"dec_4" -> deCode[dep[[1]], dimIn, res, drop],
		"map" -> map
	|>,{
		NetPort["Input"] -> "start" -> "enc_1" -> "enc_2" -> "enc_3" -> "enc_4" -> "enc_5",
		{"enc_5", "enc_4"} -> "dec_1",
		{"dec_1", "enc_3"} -> "dec_2",
		{"dec_2", "enc_2"} -> "dec_3",
		{"dec_3", "enc_1"} -> "dec_4",
		"dec_4" -> "map"
	}, "Input" -> Prepend[dimIn, NChan]]
]


(* ::Subsubsection::Closed:: *)
(*CBE*)


CBE[dep_, k_, r_: True] := Block[{d = Round[dep]},
	{ConvolutionLayer[If[IntegerQ[d],d,First[d]], k, "PaddingSize" -> (k - 1)/2], BatchNormalizationLayer[], ElementwiseLayer["ELU"]}[[;;If[r, -1, -2]]]
]


(* ::Subsubsection::Closed:: *)
(*enCode*)


enCode[n_, dimIn_, res_, drop_, resc_]:= Flatten@{
	If[resc, Resize["down", dimIn], Nothing],
	conv[n, dimIn, res, "encode"],
	DropoutLayer[drop]}


(* ::Subsubsection::Closed:: *)
(*deCode*)


deCode[n_, dimIn_, res_, drop_] := NetGraph[<|
	"upconv" -> Flatten[{Resize["up", dimIn], CBE[n ,1, False]}],
	"conv" -> Flatten@{CatenateLayer[], conv[n, dimIn, res, "decode"], DropoutLayer[drop]}
|>, {
	{NetPort["Input2"] -> "upconv", NetPort["Input1"]} -> "conv"
}];



(* ::Subsubsection::Closed:: *)
(*Resize*)


Resize[dir_, dim_] := Block[{d, sc},
	d=Length[dim];
	sc = If[Ceiling[dim[[1]]]===1, Prepend[ConstantArray[2, d-1], 1], ConstantArray[2, d]];
	Switch[dir,
		"up", ResizeLayer[Scaled/@sc, Resampling -> "Nearest"],
		"down", PoolingLayer[sc, 2, "Function" -> Max]
	]
];



(* ::Subsubsection::Closed:: *)
(*conv*)


conv[n_, dimIn_, res_, lab_] := Block[{},
	If[verb, Print[lab<>" - dimensions: ", Prepend[dimIn, n]]];
	Switch[res,
		"ResNet",
		NetGraph[<|
			"con" -> Join[CBE[n/2, 3], CBE[n, 3, False]], 
			"skip" -> CBE[n, 1, False], 
			"tot" -> {TotalLayer[], ElementwiseLayer["ELU"]}
		|>, {
			{"con", "skip"} -> "tot"
		}],
		
		"UResNet",(*same as resnet but without skip layer*)
		Flatten[{CBE[n/2, 3], CBE[n, 3]}],
		
		"DenseNet",
		NetGraph[Association[Join[{
				"trans" -> Flatten[{CatenateLayer[], CBE[n, 1][[All]]}]
				}, convLayers[n]
			]],
			connect[NetPort["Input"], n[[2]]]
		],
		
		"UDenseNet",(*same as dnesenete but without skip layer*)
		Flatten[ConstantArray[CBE[n, 3], n[[2]]]],
		
		_,(*default which is UNET*)
		Flatten[{CBE[n, 3], CBE[n, 3]}]
	]
]


(* ::Subsubsection::Closed:: *)
(*connect*)


connect[inp_, dep_] := Flatten[Table[Switch[rep,
	1, {inp -> layName[rep]},
	dep + 1, Flatten[{inp, Table[layName[rr], {rr, 1, rep - 1}]},1] -> "trans",
	_, Flatten[{inp, Table[layName[rr], {rr, 1, rep - 1}]},1] -> layName[rep]
], {rep, 1, dep + 1}],1]


(* ::Subsubsection::Closed:: *)
(*ConvLayers*)


convLayers[{k_, dep_}] :=Table[layName[rep] -> Flatten[{CatenateLayer[], CBE[k, 1], CBE[k, 3]}[[If[rep===1,-2,1];;]]], {rep, 1, dep}]


(* ::Subsubsection::Closed:: *)
(*layName*)


layName[rep_] := "layer_" <> ToString[rep]


(* ::Subsection:: *)
(*Train Net*)


(* ::Subsubsection::Closed:: *)
(*Train UNET*)


Options[TrainUNET]=Join[{
	NetParameters->8, 
	BlockType ->"ResNet", 
	NetLossLayers->All, 
	DropOutRate->0.2,
	AugmentTrainData->{False,False,False}
	},Options[NetTrain]];

SyntaxInformation[TrainUNET] = {"ArgumentsPattern" -> {_, _, _., OptionsPattern[]}};

TrainUNET[train_, valid_, opt:OptionsPattern[]]:=TrainUNET[train, valid, {None, None}, opt]

TrainUNET[train_, valid_, {testData_, testLabel_}, opt:OptionsPattern[]]:=Block[{
	Nchan,Nclass,net,device,trained,netTrained,result,plots,iou, block, loss, drop,
	netDim,datDim,netPar,trainopt,lossNet,lossFunction,aug},
	
	(*get the data dime*)
	datDim=Dimensions[train[[1,1]]][[2;;]];
	netDim=Length[datDim];
	
	(*Get the data Channels and classes*)
	Nchan = Length[train[[1, 1]]];
	Nclass = Max[train[[All, 2]]];
	
	Print["channels: ",Nchan," - classes: ",Nclass, " - Dimensions :", datDim];
	
	(*get the function options*)
	netPar = OptionValue[NetParameters];
	block = OptionValue[BlockType];
	drop = OptionValue[DropOutRate];
	aug = OptionValue[AugmentTrainData];
	device = OptionValue[TargetDevice];
	trainopt = Sequence@FilterRules[{opt},Options[NetTrain]];
	
	(*check if net is complex enough*)
	(*If[netPar<Nclass, Print["Number of NetParameters should be higher than number of classes!"]; netPar = Nclass+1];*)
	
	loss = {"Loss1","Loss2","Loss3"}[[OptionValue[NetLossLayers]]];
	
	(*chekc if data dimensions are valid for net*)
	If[(!AllTrue[MemberQ[Range[20]2^4,#]&/@datDim,TrueQ])&&(MemberQ[{2,3},netDim]),
		(*not a valid dimension*)
		Return[Message[TrainUNET::dim, datDim]]
		,
		(*initialize and train net*)
		net = MakeUNET[Nchan, Nclass, netPar, datDim, BlockType -> block, DropOutRate -> drop];
		(*Attatch the loss funtion if needed*)
		{lossNet, lossFunction} = If[Nclass>1, {AddLossLayer[net,netDim], loss}, {net, Automatic}];
		(*train the net*)
		trained = NetTrain[lossNet,
			{TrainFunction[train, #BatchSize, aug, Nclass]&, "RoundLength" -> Length[train]}, 
			All,
			ValidationSet -> TrainFunction[valid, Length[valid], {False,False,False}, Nclass],
			TargetDevice -> device, LossFunction->lossFunction,
			WorkingPrecision -> "Mixed", PerformanceGoal -> {"TrainingMemory", "TrainingSpeed"},
			trainopt
		];
		
		(*extract the trained net*)
		netTrained = If[Nclass>1, NetExtract[trained["TrainedNet"],"net"], trained["TrainedNet"]];
		
		(*evlaueate the test data*)
		Print["Evaluating test data"];
		If[testData=!=None,
			(*test data provided*)
			result = netTrained[testData, TargetDevice->device];
			(*decode the data*)
			result = If[Nclass>1, ClassDecoder[result, Nclass], Round[result]];
			(*get the Dice or test data*)
			iou = DiceSimilarityClass[result,testLabel, Nclass];
			Print["DICE per class: ",Thread[Range[Nclass]->Round[iou,0.001]]];
			(*give Output*)
			{{lossNet, trained, netTrained}, {result, iou}}
			,
			(*no test data*)
			{lossNet, trained, netTrained}
		]
	]
]


(* ::Subsubsection::Closed:: *)
(*Train UNET*)


TrainFunction[train_, b_, aug_ ,n_] := AugmentData[RandomBatch[train[[All, 1]], train[[All, 2]], b], aug, n]


RandomBatch[data_, label_, b_] := Block[{s, l},
	l = Length[data];
	s = If[b>=l, Range[l], RandomSample[Range[l], b]];
	{data[[s]], label[[s]]}
]


AugmentData[{data_, label_}, set_, n_] := Block[{dat, lab, fun, f},
	{dat, lab} = If[AllTrue[set, # === False &],
		{data,  label},
		fun = Switch[ArrayDepth[data], 4, AugmentFunction2D, 5, AugmentFunction3D];
		Transpose[MapThread[(f = fun[set]; {f /@ #1, Round[f[#2]]}) &, {data, label}]]
	];
	Thread[dat -> ClassEncoder[lab,n]]
]


AugmentFunction2D[{rot_, scale_, flip_}] := Block[{fl, r, sc, rr, scs, flf},
	r = RotationTransform[If[rot === "90", RandomChoice[{0., 90., 180., -90.} Degree], If[ListQ[rot], RandomReal[rot], 0.]], {0.5, 0.5}];
	sc = ScalingTransform[If[ListQ[scale], RandomReal[scale, 2], {1, 1}]];
	
	fl = If[flip, RandomChoice[{ImageReflect[#, Left] &, ImageReflect[#, Top] &, ImageReflect[#, All] &, # &}], # &];
	
	ImageData[flf@ImageTransformation[Image[#, "Real32"], rr . scs, Full, Masking -> Full, Resampling -> "Nearest", Padding -> "Reversed"]] & /. {rr -> r, scs -> sc, flf -> fl}
]


AugmentFunction3D[{rot_, scale_, flip_}] := Block[{fl, r, sc, rr, scs, flf},
	r = RotationTransform[
		If[rot === "90", RandomChoice[{0., 90., 180., -90.} Degree], If[ListQ[rot], RandomReal[rot], 0.]],
		If[rot === "90", RandomChoice[{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}], If[ListQ[rot], Normalize@RandomVariate[NormalDistribution[], 3], {1, 0, 0}]],
	{0.5, 0.5, 0.5}];
	sc = ScalingTransform[If[ListQ[scale], RandomReal[scale, 3], {1, 1, 1}]];
	
	fl = If[flip, RandomChoice[{ImageReflect[#, Left] &, ImageReflect[#, Top] &, ImageReflect[#, All] &, # &}], # &];

	ImageData[flf@ImageTransformation[Image3D[#, "Real32"], rr . scs, All, Masking -> Full, Resampling -> "Nearest", Padding -> "Reversed"]] & /. {rr -> r, scs -> sc, flf -> fl}
]


(* ::Subsubsection:: *)
(*AddLossLayer*)


AddLossLayer[net_,dim_]:=NetGraph[<|
		"net"->net,
		"SoftDice" -> SoftDiceLossLayer[dim],
		"CrossEntr" -> CrossEntropyLossLayer["Probabilities"],
		"Brier" -> {BrierLossLayer[dim], FunctionLayer[10 #&]}
	|>,{
		NetPort["Input"]->"net"->NetPort["Output"],
		{"net",NetPort["Target"]}->"SoftDice"->NetPort["Loss1"],
		{"net",NetPort["Target"]}->"CrossEntr"->NetPort["Loss2"],
		{"net",NetPort["Target"]}->"Brier"->NetPort["Loss3"]
}]


(* ::Subsubsection::Closed:: *)
(*ClassEndocer*)


SyntaxInformation[ClassEncoder] = {"ArgumentsPattern" -> {_, _}};

ClassEncoder[data_,NClass_]:=ClassEncoderC[data,NClass]

ClassEncoderC = Compile[{{data, _Integer, 2}, {n, _Integer, 0}},
	Transpose[1 - Unitize[ConstantArray[data, n] - Range[n]], {3, 1, 2}],
	RuntimeAttributes -> {Listable}, RuntimeOptions -> "Speed"
 ]


(* ::Subsubsection::Closed:: *)
(*ClassDecoder*)


SyntaxInformation[ClassDecoder] = {"ArgumentsPattern" -> {_, _}};

ClassDecoder[data_,NClass_]:=ClassDecoderC[data,NClass]

ClassDecoderC = Compile[{{data, _Real, 1}, {n, _Integer, 0}},  Block[{r = Range[n]}, Total[r (1 - Unitize[Chop[(data/Max[data]) - 1]])]], RuntimeAttributes -> {Listable}, RuntimeOptions -> "Speed"]


(* ::Subsection:: *)
(*Prepare Data*)


(* ::Subsubsection::Closed:: *)
(*SplitTestData*)


Options[SplitTrainData]={RandomizeSplit->True,SplitRatios->{0.7,.2,.1}};

SyntaxInformation[SplitTrainData] = {"ArgumentsPattern" -> {_, _, OptionsPattern[]}};

SplitTrainData[data_,label_,OptionsPattern[]]:=Block[{
	allData, train, valid, test, testData, testLabel, datas1, datas2, labels1, labels2, 
	Nclass, dim, s1, s2, s3, order, ratio, rand},
	
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
	Print["Number of Samples in each set: ",Length/@{s1,s2,s3}];
	
	(*Encode Data*)
	Nclass=Max[label];
	dim=ArrayDepth[label]-1;
	
	(*data augmentation*)
	{datas1, datas2} = {data[[s1]],data[[s2]]};
	{labels1, labels2} = {label[[s1]],label[[s2]]};

	(*make training validation and test data*)
	If[Nclass>1,
		train = Thread[datas1->labels1];
		valid = Thread[datas2->labels2];
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

SyntaxInformation[ShowChannelClassData] = {"ArgumentsPattern" -> {_, _, _., OptionsPattern[]}};

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
	data = dataI[[m]];
	
	size=Round[Sqrt[Length[data]]];
	datIm=ImageAssemble[Partition[Image[#/Max[#]]&/@data,size,size,1,Image[0 data[[1]]]]];
	
	Row[{Image[datIm,ImageSize->{500,500}],Colorize[images[[n]],ColorFunction->col,ImageSize->{500,500}]}]

	,
	(*fixed parameters*)
	{{nodes,{"start","enc_1","enc_2","enc_3","enc_4","enc_5","dec_1","dec_2","dec_3","dec_4","map"}}, ControlType->None},
	{data,ControlType->None},
	{size,ControlType->None},
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


(* ::Subsubsection:: *)
(*VisualizeNetIm*)


VisualizeNetIm[data_,net_,nodes_]:=Block[{out,size,port},(
		out=Take[net,{NetPort["Input"],#}][data];
		out=If[#==="map", If[Length@Dimensions@out>2,TransData[out,"r"],{out}]	,out];
		size=Round[Sqrt[Length[out]]];
		ImageAssemble[Partition[Image[#]&/@out,size,size,1,Image[0 out[[1]]]]]
	)&/@nodes
]


TransData[data_,dir_]:=Block[{ran,dep,fun},
	ran=Range[dep=ArrayDepth[data]];
	fun=Switch[dir,"r",RotateLeft[ran],"l",RotateRight[ran]];
	Transpose[data,fun]
]


MeanSurfaceDistance[res_, label_] := 
 Block[{resS, labS, posR, posL, m, pR, pL, mat, msd, lpR, lpL},
  (*Split the segmentations*)
  resS = SplitSegmentations[res][[1]];
  labS = SplitSegmentations[label][[1]];
  (*Get the posisitons of the edge voxels*)
  posR = aa = 
    Map[Position[m = ArrayPad[#, 1]; m1 = m - Erosion[m, 1], 1] &, 
     resS, {2}];
  posL = bb = 
    Map[Position[m = ArrayPad[#, 1]; m2 = m - Erosion[m, 1], 1] &, 
     labS, {2}];
  (*calculate the MSD for each point for ech ROI*)
  msd = MapThread[(
      pR = #; pL = #2;
      lpR = Length[pR]; lpL = Length[pL];
      If[lpR == 0 || lpL == 0,
       Missing,
       msd = ParallelMap[Norm[Nearest[pL, #, 1][[1]] - #] &, pR];
       Mean[N[msd]]
       ]
      ) &, {posR, posL}, 2];
  
  Mean[DeleteCases[#, Missing]] & /@ Transpose[msd]
  ]


(* ::Section:: *)
(*End Package*)


End[]

EndPackage[]
