(* ::Package:: *)

(* ::Title:: *)
(*UNET init File*)


(* ::Subtitle:: *)
(*Written by: Martijn Froeling, PhD*)
(*m.froeling@gmail.com*)


(* ::Section:: *)
(*Functions*)


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
	(If[print,Print["Loading all definitions of "<>#]];Get[pack<>#];)&/@subpack;
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

subPackages = {
	"UnetCore`","UnetSupport`"
	};


(*define all the toolbox contexts*)
System`$UNETContextPaths = (package <> # & /@ subPackages);
Print[System`$UNETContextPaths];

$ContextPath = Union[$ContextPath, System`$UNETContextPaths];

(*state if verbose is true to monitor initialization*)
UNET`verbose = False;


(* ::Section:: *)
(*Initialize all packages*)


If[UNET`verbose,
Print["--------------------------------------"];
Print[System`$UNETContextPaths];
];

(*clear all definitions from the subPacakges*)
ClearFunctions[package, subPackages, UNET`verbose];
(*load all packages*)
LoadPackages[package, subPackages, UNET`verbose];
(*Protect functions*)
ProtectFunctions[package, subPackages, UNET`verbose];
