import imgviz


COLOUR_CODES = {
    'twoclasses': [
        [0, 0, 0], # background
        [255, 255, 255]
        ],
    **dict.fromkeys(
        [
            'default',
            'spermseg', "spermtrack",
        ], 
        list(map(list, imgviz.label_colormap(256)))
    ),
    }