(* ::Package:: *)

(* ::Title:: *)
(*UNET init File*)


(* ::Subtitle:: *)
(*Written by: Martijn Froeling, PhD*)
(*m.froeling@gmail.com*)


BeginPackage["UNET`"];

(*usage notes*)
UNET`$SubPackages::usage = "List of the subpackages.";
UNET`$Contexts::usage = "The package contexts.";
UNET`$ContextsFunctions::usage = "The package contexts with the list of functions.";
UNET`$Verbose::usage = "When set True, verbose loading is used.";
UNET`$InstalledVersion::usage = "The version number of the installed package.";

(*subpackages names*)
UNET`$SubPackages = {"UnetCore`", "UnetSupport`"};

(*define context and verbose*)
UNET`$Contexts = (Context[] <> # & /@ UNET`$SubPackages);
UNET`$Verbose = If[UNET`$Verbose===True, True, False];
UNET`$InstalledVersion = First[PacletFind[StringDrop[Context[],-1]]]["Version"];

(*load all the packages without error reporting such we can find the names of all the functions and options*)
Quiet[Get/@UNET`$Contexts];
UNET`$ContextsFunctions = {#, Names[# <> "*"]}& /@ UNET`$Contexts;

(*print the Toolbox content and version*)
If[UNET`$Verbose,
	Print["--------------------------------------"];
	Print["Loading ", Context[]," with version number ", UNET`$InstalledVersion];
	Print["--------------------------------------"];
	Print["Defined packages and functions to be loaded are: "];
	(
		Print["   - ", First@#, " with functions:"];
		Print[Last@#];
	)&/@ UNET`$ContextsFunctions
];

Begin["`Private`"];

End[];

EndPackage[];


(*Destroy all functions defined in the subpackages*)
If[UNET`$Verbose, 
	Print["--------------------------------------"];
	Print["Removing all local and global definitions of:"];
];

With[{
		global = Intersection[Names["Global`*"], "Global`" <> # & /@ Last[#]]
	},
		
	If[UNET`$Verbose, 
		Print["   - ", First@#];
		If[global=!={}, Print[global]]
	];

	Unprotect @@ Join[Last@#,global];
	ClearAll @@ Join[Last@#,global];
	Remove @@ global;
] &/@ UNET`$ContextsFunctions

(*Reload and protect all the sub packages with error reporting*)
If[UNET`$Verbose, 
	Print["--------------------------------------"];
	Print["Loading and protecting all definitions of:"];
];

(
	If[UNET`$Verbose, Print["   - ", First@#]];
	Get[First@#];
	SetAttributes[#, {Protected, ReadProtected}]& /@ Last[#]
)& /@ UNET`$ContextsFunctions;
