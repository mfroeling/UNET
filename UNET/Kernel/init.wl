(* ::Package:: *)

(* ::Title:: *)
(*UNET init File*)


(* ::Subtitle:: *)
(*Written by: Martijn Froeling, PhD*)
(*m.froeling@gmail.com*)

BeginPackage["UNET`"];


$Package::usage = "Name of the package.";
$SubPackages::usage = "List of the subpackages.";
$Contexts::usage = "The package contexts.";
$Verbose::usage = "When set True, verbose loading is used.";


(*package naem*)
$Package = "UNET`";
$SubPackages = {
	(*core packages that contain functions for other toolboxes*)
	"UnetCore`", "UnetSupport`"
};


(*define context and verbose*)
$Contexts = ($Package <> # & /@ $SubPackages);
$Verbose = If[$Verbose===True, True, False];


(*print the contexts*)
If[$Verbose,
	Print["--------------------------------------"];
	Print["All defined packages to be loaded are: "];
	Print[$Contexts];
];


(*load all the packages without error reporting such we can find the names*)
If[$Verbose, Print["--------------------------------------"]];
Quiet[Get/@$Contexts];


(*Destroy all functions defined in the subpackages*)
(
	If[$Verbose, 
		Print["Removing all definitions of "<>#];
		Print["- Package functions: \n", Names[# <> "*"]];
		Print["- Package functions in global:\n", Intersection[Names["Global`*"], "Global`" <> # & /@ Names[# <> "*"]]];
	];
	
	Unprotect @@ Names[# <> "*"];
	ClearAll @@ Names[# <> "*"];
	
	Unprotect @@ Intersection[Names["Global`*"], "Global`" <> # & /@ Names[# <> "*"]];
	ClearAll @@ Intersection[Names["Global`*"], "Global`" <> # & /@ Names[# <> "*"]];
	Remove @@ Intersection[Names["Global`*"], "Global`" <> # & /@ Names[# <> "*"]];
) &/@ $Contexts


(*reload all the sub packages with error reporting*)
If[$Verbose,Print["--------------------------------------"]];
(
	If[UNET`$Verbose, Print["Loading all definitions of "<>#]];
	Get[#];
)&/@$Contexts;	


(*protect all functions*)
If[$Verbose,Print["--------------------------------------"]];
(
	If[$Verbose,
		Print["protecting all definitions of "<>#];
		Print[Names[# <> "*"]];
		Print["--------------------------------------"]
	];
	
	SetAttributes[#,{Protected, ReadProtected}]&/@ Names[# <> "*"];
)& /@ $Contexts;

Begin["`Private`"]

End[]

EndPackage[]