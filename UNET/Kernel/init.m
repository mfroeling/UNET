(* ::Package:: *)

(* ::Title:: *)
(*UNET init File*)


(* ::Subtitle:: *)
(*Written by: Martijn Froeling, PhD*)
(*m.froeling@gmail.com*)


(*package naem*)
UNET`$Package = "UNET`";
UNET`$SubPackages = {
	(*core packages that contain functions for other toolboxes*)
	"UnetCore`", "UnetSupport`"
};


(*define context and verbose*)
UNET`$Contexts = (UNET`$Package <> # & /@ UNET`$SubPackages);
UNET`$Verbose = False;


(*print the contexts*)
If[UNET`$Verbose,
	Print["--------------------------------------"];
	Print["All defined packages to be loaded are: "];
	Print[UNET`$Contexts];
];


(*load all the packages without error reporting such we can find the names*)
If[UNET`$Verbose, Print["--------------------------------------"]];
Quiet[Get/@UNET`$Contexts];


(*Destroy all functions defined in the subpackages*)
(
	If[UNET`$Verbose, 
		Print["Removing all definitions of "<>#];
		Print["- Package functions: \n", Names[# <> "*"]];
		Print["- Package functions in global:\n", Intersection[Names["Global`*"], "Global`" <> # & /@ Names[# <> "*"]]];
	];
	
	Unprotect @@ Names[# <> "*"];
	ClearAll @@ Names[# <> "*"];
	
	Unprotect @@ Intersection[Names["Global`*"], "Global`" <> # & /@ Names[# <> "*"]];
	ClearAll @@ Intersection[Names["Global`*"], "Global`" <> # & /@ Names[# <> "*"]];
	Remove @@ Intersection[Names["Global`*"], "Global`" <> # & /@ Names[# <> "*"]];
) &/@ UNET`$Contexts


(*reload all the sub packages with error reporting*)
If[UNET`$Verbose,Print["--------------------------------------"]];
(
	If[UNET`$Verbose, Print["Loading all definitions of "<>#]];
	Get[#];
)&/@UNET`$Contexts;	


(*protect all functions*)
If[UNET`$Verbose,Print["--------------------------------------"]];
(
	If[UNET`$Verbose,
		Print["protecting all definitions of "<>#];
		Print[Names[# <> "*"]];
		Print["--------------------------------------"]
	];
	
	SetAttributes[#,{Protected, ReadProtected}]&/@ Names[# <> "*"];
)& /@ UNET`$Contexts;