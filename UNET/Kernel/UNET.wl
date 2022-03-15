(* ::Package:: *)

(* ::Title:: *)
(*UNET init File*)


(* ::Subtitle:: *)
(*Written by: Martijn Froeling, PhD*)
(*m.froeling@gmail.com*)


BeginPackage["UNET`"];

(*usage notes*)
UNET`$SubPackagesUNET::usage = "List of the subpackages.";
UNET`$ContextsUNET::usage = "The package contexts.";
UNET`$ContextsFunctionsUNET::usage = "The package contexts with the list of functions.";
UNET`$VerboseUNET::usage = "When set True, verbose loading is used.";
UNET`$InstalledVersionUNET::usage = "The version number of the installed package.";

(*subpackages names*)
UNET`$SubPackagesUNET = {"UnetCore`", "UnetSupport`"};

(*define context and verbose*)
UNET`$ContextsUNET = (Context[] <> # & /@ UNET`$SubPackagesUNET);
UNET`$VerboseUNET = If[UNET`$VerboseUNET===True, True, False];
UNET`$InstalledVersionUNET = First[PacletFind[StringDrop[Context[],-1]]]["Version"];

(*load all the packages without error reporting such we can find the names of all the functions and options*)
Quiet[Get/@UNET`$ContextsUNET];
UNET`$ContextsFunctionsUNET = {#, Names[# <> "*"]}& /@ UNET`$ContextsUNET;

(*print the Toolbox content and version*)
If[UNET`$VerboseUNET,
	Print["--------------------------------------"];
	Print["Loading ", Context[]," with version number ", UNET`$InstalledVersionUNET];
	Print["--------------------------------------"];
	Print["Defined packages and functions to be loaded are: "];
	(
		Print["   - ", First@#, " with functions:"];
		Print[Last@#];
	)&/@ UNET`$ContextsFunctionsUNET
];

Begin["`Private`"];

End[];

EndPackage[];


(*Destroy all functions defined in the subpackages*)
If[UNET`$VerboseUNET, 
	Print["--------------------------------------"];
	Print["Removing all local and global definitions of:"];
];

With[{
		global = Intersection[Names["Global`*"], "Global`" <> # & /@ Last[#]]
	},
		
	If[UNET`$VerboseUNET, 
		Print["   - ", First@#];
		If[global=!={}, Print[global]]
	];

	Unprotect @@ Join[Last@#,global];
	ClearAll @@ Join[Last@#,global];
	Remove @@ global;
] &/@ UNET`$ContextsFunctionsUNET

(*Reload and protect all the sub packages with error reporting*)
If[UNET`$VerboseUNET, 
	Print["--------------------------------------"];
	Print["Loading and protecting all definitions of:"];
];

(
	If[UNET`$VerboseUNET, Print["   - ", First@#]];
	Get[First@#];
	SetAttributes[#, {Protected, ReadProtected}]& /@ Last[#]
)& /@ UNET`$ContextsFunctionsUNET;
