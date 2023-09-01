(* ::Package:: *)

Paclet[
    Name -> "UNET",
    Version -> "3.0",
    WolframVersion -> "13.0+",
    Description -> "A package to generate and train a UNET deep convolutional network for 2D and 3D image segmentation",
    Creator -> "Martijn Froeling <m.froeling@gmail.com>",
    Support -> "https://github.com/mfroeling/UNET",
    Icon -> "Resources/thumb.gif",
    Extensions -> 
        {
            {"Kernel", Root -> "Kernel", Context -> "UNET`"}, 
            {"Documentation", Language -> "English", MainPage -> "Guides/UNET"},
            {"Resource", Root -> "Resources", Resources -> {{"Logo", "thumb.gif"}}}
        }
]
