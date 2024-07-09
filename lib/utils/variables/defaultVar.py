# System
TextColors = {
    "end_colour": "\033[0m",
    "bold": "\033[1m", # 033 is the escape code and 1 is the color code 
    "error": "\033[31m", # red
    "light_green": "\033[32m",
    "light_yellow": "\033[33m",
    "light_blue": "\033[34m",
    "light_cyan": "\033[36m",
    "warning": "\033[37m", # white
}


# Metric
LogMetric = ["accuracy", "miou", "map"]
WeightMetric = {
    "classificaiton": "maxacc", 
    "segmentation": "maxiou", 
    "detection": "maxap",
}


# Method
ClrMethod = [
    "simclr", "mixclr",
    "simsiam", "mixsiam",
    "moco", "barlow",
    "triplet",
]

ClrNumClasses = {
    **dict.fromkeys(["simclr", "mixclr"], 128), 
    **dict.fromkeys(["simsiam", "mixsiam"], 2048), 
    "moco": 128, "barlow": 8192, 
    "triplet": 256,
}


# Dataset
RESDICT = {
    "default": {
        "classification" : 224, "segmentation": 512, 
        "detection": 320, "ins_segmentation": 512,
        "autoencoder": 224,
    }, # "regression"
    
    # segmentation
    **dict.fromkeys(
        [
            "semspermna", "SpermSeg", 
        ], 
        512
    ),
}

# True if using the default image size of dataset, otherwise False
PRE_RESIZE = {
    "default": False,
    **dict.fromkeys(
        [
            "semspermna", "SpermSeg",
        ], 
    False
    ),
}

CLASS_NAMES = {
    # classification, autoencoder
    "isic2018t1": ["symptoms"],
    **dict.fromkeys(["spermseg"], 
                    ["normal", "abnormal"]),
    "spermtrack": ["sperm"], # detection
    "atlas": ["liver", "tumour"],
    "fives": ["amd", "dr", "glaucoma", "health"], # amd: age-related macular degeneration, dr: diabetic retinopathy
}