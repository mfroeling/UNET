(* ::Package:: *)

(* ::Title:: *)
(*UNET init File*)


(* ::Subtitle:: *)
(*Written by: Martijn Froeling, PhD*)
(*m.froeling@gmail.com*)


(* ::Section:: *)
(*Functions*)


(*check for latest version*)
UpdateWarning[]:=If[$VersionNumber != 11.3,
	CreateDialog[Column[{Style["
	Current Mathematica version is "<>ToString[$VersionNumber]<>"
	The toolbox is tested developed in 11.3.
	You need to update! (Or I am behind...)
	Some functions wont work in older versions.
	", TextAlignment -> Center], DefaultButton[], ""}, 
	Alignment -> Center], WindowTitle -> "Update!"];
];
 

(*Fucntions to clear, load and protect package functions*)
ClearFunctions[pack_,subpack_,print_:False]:=Module[{packageName,packageSymbols,packageSymbolsG},
	If[print,Print["--------------------------------------"]];
	Quiet[
		If[print,Print["Removing all definitions of "<>#]];
		packageName = pack <> #;
		Get[packageName];
		packageSymbols =Names[packageName <> "*"];
		packageSymbolsG="Global`"<>#&/@packageSymbols;
		
		(*remove all global and private definitions*)
		Unprotect @@ packageSymbols;
		ClearAll @@ packageSymbols;
		Remove @@ packageSymbols;
		Unprotect @@ packageSymbolsG;
		ClearAll @@ packageSymbolsG;
		Remove @@ packageSymbolsG;
		
		]& /@ subpack;
];


LoadPackages[pack_,subpack_,print_:False]:=Module[{},
	If[print,Print["--------------------------------------"]];
	(
		If[print,
			Print["Loading all definitions of "<>#]];
		Get[pack<>#];
	)&/@subpack;
]


ProtectFunctions[pack_,subpack_,print_:False]:=Module[{},
	If[print,Print["--------------------------------------"]];
	(
		If[print,Print["protecting all definitions of "<>#]];
		SetAttributes[#,{Protected, ReadProtected}]&/@ Names[pack <> # <> "*"];
		If[print,Print[Names[pack <> # <> "*"]]];
		If[print,Print["--------------------------------------"]];
	)& /@ subpack;
]


(* ::Section:: *)
(*Settings*)


(*List Main package and sub packages*)
package = "UNET`";

subPackages = {"Unet`","UnetSupport`"};


(*define all the toolbox contexts*)
System`$UNETContextPath = (package <> # & /@ subPackages);

$ContextPath = Union[$ContextPath, System`$UNETContextPath]

System`$UNETContextPath::usage = "$UNETContextPath lists all the packages";


(* ::Section:: *)
(*Initialize all packages*)


(*state if verbose is true to monitor initialization*)
UNET`verbose = False;

If[UNET`verbose,
Print["--------------------------------------"];
Print[System`$UNETContextPath];
];

(*clear all definitions from the subPacakges*)
ClearFunctions[package, subPackages, UNET`verbose];
(*load all packages*)
LoadPackages[package, subPackages, UNET`verbose];
(*Protect functions*)
ProtectFunctions[package, subPackages, UNET`verbose];
