(* ::Package:: *)

(* ::Title:: *)
(*UNET*)


(* ::Subtitle:: *)
(*Written by: Martijn Froeling, PhD*)
(*m.froeling@gmail.com*)


(* ::Section:: *)
(*start Package*)


BeginPackage["UNET`UnetSupport`",  Join[{"Developer`"}, Complement[UNET`$Contexts, {"UNET`UnetSupport`"}]]]


(* ::Section:: *)
(*Usage Notes*)


MakeTestImages::usage = "MakeTestImages[n,case] generates n 2D test images for segmentation testing. There are four cases.
case1: One channel one class.
case2: One channel two classes.
case3: One channel four classes.
case4: Three channels four classes.
"  

CreateImage1::usage = "CreateImage1[] creates a test images with label with one channel and one class."

CreateImage2::usage = "CreateImage2[] creates a test images with label with one channel and two classes."

CreateImage3::usage = "CreateImage3[] creates a test images with label with one channel and four classes."

CreateImage4::usage = "CreateImage4[] creates a test images with label with three channels and four classes."

(* ::Subsection:: *)
(*Functions*)


(* ::Subsection:: *)
(*Options*)


(* ::Subsection:: *)
(*Error messages*)


(* ::Section:: *)
(*Create training Data*)


Begin["`Private`"]


(* ::Subsection:: *)
(*Create Data Functions*)


(* ::Subsubsection::Closed:: *)
(*Signal*)


(*MRI signal equation*)
SignalI[{T2_,T1_},{TE_,TR_}]:=N[(1-Exp[-TR/T1])Exp[-TE/T2]]

(*T2 series for muscle and fat relaxation parameters*)
par={{20,2000},{40,2000},{60,2000}};
sig={SignalI[{30,1400},#],SignalI[{200,300},#]}&/@par;

(*normalize the signal and give the lowest signal an SNR of 5*)
sig=sig/Max[sig];
noise=Min[sig]/5;

(*how big and where to place images*)
pos={10,90};
size={10,30};


(* ::Subsubsection::Closed:: *)
(*AddNoise*)


AddNoiseI=Compile[{{Mu,_Real,0},{Sigma,_Real,0}},Sqrt[RandomReal[NormalDistribution[Mu,Sigma]]^2+RandomReal[NormalDistribution[0,Sigma]]^2],RuntimeAttributes->{Listable},RuntimeOptions->"Speed"];


(* ::Subsubsection::Closed:: *)
(*Single Channel single class*)


CreateImage1[]:=CreateImage1i[]

CreateImage1i[]:=Block[{an,l1,r1,square,d1,data},
	(*create box*)
	an=RandomReal[{0,90}]Degree;
	{l1,r1}={RandomReal[pos,2],RandomReal[size]};
	square=1-Round@ImageData[ColorConvert[ImageResize[Graphics[Rotate[Rectangle[l1-r1,l1+r1],an,l1],PlotRange->{{0,100},{0,100}}],{128,128}],"Grayscale"]];
	
	(*generate data*)
	d1=N[square];
	data=AddNoiseI[d1,0.2];
	{{data},square}
]


(* ::Subsubsection::Closed:: *)
(*Single Channel single class as two*)


CreateImage2[]:=CreateImage2i[]

CreateImage2i[]:=Block[{an,l1,r1,square,d1,data},
(*create box*)
	an=RandomReal[{0,90}]Degree;
	{l1,r1}={RandomReal[pos,2],RandomReal[size]};
	square=1-Round@ImageData[ColorConvert[ImageResize[Graphics[Rotate[Rectangle[l1-r1,l1+r1],an,l1],PlotRange->{{0,100},{0,100}}],{128,128}],"Grayscale"]];
	
	(*generate data*)
	d1=N[square];
	data=AddNoiseI[d1,0.3];
	{{data},square+1}
]


(* ::Subsubsection::Closed:: *)
(*Single Channel multi class*)

CreateImage3[]:=CreateImage3i[]

CreateImage3i[]:=Block[{an,l1,r1,square,d1,data, l2, r2, sphere, l3, l0, labels},
	(*create box*)
	an=RandomReal[{0,90}]Degree;
	{l1,r1}={RandomReal[pos,2],RandomReal[size]};
	square=1-Round@ImageData[ColorConvert[ImageResize[Graphics[Rotate[Rectangle[l1-r1,l1+r1],an,l1],PlotRange->{{0,100},{0,100}}],{128,128}],"Grayscale"]];
	
	(*create disk*)
	{l2,r2}={RandomReal[pos,2],RandomReal[size]};
	sphere=1-Round@ImageData[ColorConvert[ImageResize[Graphics[Disk[l2,r2],PlotRange->{{0,100},{0,100}}],{128,128}],"Grayscale"]];
	
	(*generate label*)
	l1=sphere(1-square);
	l2=square(1-sphere);
	l3=square sphere;
	l0=1-(l1+l2+l3);
	labels=l0+2l1+3l2+4l3;
	
	(*generate data*)
	d1=N[Total[{1,1}{square ,sphere}]];
	data=AddNoiseI[d1,0.2];
	
	{{data},labels}
]


(* ::Subsubsection::Closed:: *)
(*Multi Channel multi class*)


CreateImage4[]:=CreateImage4i[]

CreateImage4i[]:=Block[{an,l1,r1,square,d1,data, l2, r2, sphere, l3, l0, labels},
	(*create box*)
	an=RandomReal[{0,90}]Degree;
	{l1,r1}={RandomReal[pos,2],RandomReal[size]};
	square=1-Round@ImageData[ColorConvert[ImageResize[Graphics[Rotate[Rectangle[l1-r1,l1+r1],an,l1],PlotRange->{{0,100},{0,100}}],{128,128}],"Grayscale"]];
	
	(*create disk*)
	{l2,r2}={RandomReal[pos,2],RandomReal[size]};
	sphere=1-Round@ImageData[ColorConvert[ImageResize[Graphics[Disk[l2,r2],PlotRange->{{0,100},{0,100}}],{128,128}],"Grayscale"]];
	
	(*generate label*)
	l1=sphere(1-square);
	l2=square(1-sphere);
	l3=square sphere;
	l0=1-(l1+l2+l3);
	labels=l0+2l1+3l2+4l3;
	
	(*generate data*)
	d1=N[Total[# {square, sphere}]&/@sig];
	d1={0,0.1l0,0.8l0}+d1;
	data=AddNoiseI[d1,0.1];
	
	{data,labels}
]


(* ::Subsubsection::Closed:: *)
(*Make Test Images*)


MakeTestImages[n_,case_]:=Block[{ii},
	ii=0;
	(*DistributeDefinitions[CreateImage1i,CreateImage2i,CreateImage3i,CreateImage4i];*)
	(*SetSharedVariable[ii];*)
	PrintTemporary[Dynamic[ii]];
	Transpose[Switch[case,
		1,Table[ii++;CreateImage1i[],{i,1,n}],
		2,Table[ii++;CreateImage2i[],{i,1,n}],
		3,Table[ii++;CreateImage3i[],{i,1,n}],
		4,Table[ii++;CreateImage4i[],{i,1,n}]
	]]
];


(* ::Section:: *)
(*End Package*)


End[]

EndPackage[]

