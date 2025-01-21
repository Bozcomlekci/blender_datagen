from utils.numeric import *

# clothing labels
CLOSED_CLOTHING_LABELS = {
    0: 'Hat',
    1: 'Body',
    2: 'Shirt',
    3: 'TShirt',
    4: 'Vest',
    5: 'Coat',
    6: 'Dress',
    7: 'Skirt',
    8: 'Pants',
    9: 'ShortPants',
    10: 'Shoes',
    11: 'Hoodies',
    12: 'Hair',
    13: 'Swimwear',
    14: 'Underwear',
    15: 'Scarf',
    16: 'Jumpsuits',
    17: 'Jacket'
}

# when the label is identified as noise or so, this proximity list is used to find the nearest label
CLOSED_CLOTHING_LABELS_NEAREST = {
    0: [12, 1, 11, 15, 17, 5, 2, 3, 4, 6, 7, 8, 9, 10, 13, 14, 16],  # Hat
    1: [12, 17, 5, 15, 6, 16, 11, 2, 3, 4, 7, 8, 9, 10, 13, 14, 0],  # Body
    2: [11, 1, 17, 5, 16, 4, 3, 12, 15, 0, 6, 7, 8, 9, 10, 13, 14],  # Shirt
    3: [2, 1, 11, 4, 5, 17, 16, 12, 15, 0, 6, 7, 8, 9, 10, 13, 14],  # TShirt
    4: [2, 1, 17, 5, 16, 3, 6, 11, 12, 15, 0, 7, 8, 9, 10, 13, 14],  # Vest
    5: [17, 1, 11, 16, 4, 6, 2, 12, 15, 0, 7, 8, 9, 10, 3, 13, 14],  # Coat
    6: [7, 17, 5, 1, 11, 16, 2, 12, 15, 0, 4, 8, 9, 10, 3, 13, 14],  # Dress
    7: [6, 9, 8, 1, 16, 3, 2, 12, 15, 0, 4, 5, 11, 10, 13, 14, 17],  # Skirt
    8: [9, 16, 17, 1, 5, 7, 11, 12, 15, 0, 2, 3, 4, 10, 13, 14, 6],  # Pants
    9: [8, 7, 16, 17, 3, 1, 2, 12, 15, 0, 4, 5, 10, 13, 14, 11, 6],  # ShortPants
    10: [1, 8, 9, 16, 5, 17, 11, 12, 15, 0, 2, 3, 4, 13, 14, 7, 6],  # Shoes
    11: [17, 5, 1, 2, 3, 16, 4, 12, 15, 0, 6, 7, 8, 9, 10, 13, 14],  # Hoodies
    12: [1, 0, 11, 15, 17, 6, 2, 5, 3, 4, 7, 8, 9, 10, 13, 14, 16],  # Hair
    13: [14, 9, 8, 3, 1, 7, 10, 12, 15, 0, 6, 4, 5, 11, 16, 17, 2],  # Swimwear
    14: [13, 9, 3, 1, 10, 8, 11, 12, 15, 0, 2, 4, 5, 16, 7, 17, 6],  # Underwear
    15: [12, 1, 11, 0, 5, 17, 2, 4, 3, 6, 7, 8, 9, 10, 13, 14, 16],  # Scarf
    16: [1, 17, 8, 5, 6, 2, 3, 12, 15, 0, 4, 7, 9, 10, 13, 14, 11],  # Jumpsuits
    17: [5, 1, 11, 16, 6, 2, 4, 12, 15, 0, 3, 7, 8, 9, 10, 13, 14]   # Jacket
}

# arbitrary palette, convert integer labels to 0-1 RGB colors 
COLOR_PALETTE = [
    [64, 64, 64], #dark gray
    [255, 0, 0], #red
    [0, 255, 0], #green
    [0, 0, 255], #blue
    [255, 255, 0], #yellow
    [255, 0, 255], #magenta
    [0, 255, 255], #cyan
    [255, 255, 255], #white
    [128, 0, 0], #maroon
    [0, 128, 0], #green
    [0, 0, 128], #navy
    [128, 128, 0], #olive
    [128, 0, 128], #purple
    [0, 128, 128], #teal
    [128, 128, 128], #gray
    [192, 192, 192], #silver
    [128, 128, 255], #light blue
    [128, 255, 128], #light green
    [255, 128, 128], #light red
    [255, 128, 255], #light magenta
    [128, 255, 255], #light cyan
    # so on and so forth
]

COLOR_PALETTE_0_1 = [tuple_uint8_2_tuple_float(color) for color in COLOR_PALETTE]



