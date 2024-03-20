import os
import scipy
import shutil

names = ["pink primrose",
"hard-leaved pocket orchid",
"canterbury bells",
"sweet pea",
"english marigold",
"tiger lily",
"moon orchid",
"bird of paradise",
"monkshood",
"globe thistle",
"snapdragon",
"colt's foot",
"king protea",
"spear thistle",
"yellow iris",
"globe-flower",
"purple coneflower",
"peruvian lily",
"balloon flower",
"giant white arum lily",
"fire lily",
"pincushion flower",
"fritillary",
"red ginger",
"grape hyacinth",
"corn poppy",
"prince of wales feathers",
"stemless gentian",
"artichoke",
"sweet william",
"carnation",
"garden phlox",
"love in the mist",
"mexican aster",
"alpine sea holly",
"ruby-lipped cattleya",
"cape flower",
"great masterwort",
"siam tulip",
"lenten rose",
"barbeton daisy",
"daffodil",
"sword lily",
"poinsettia",
"bolero deep blue",
"wallflower",
"marigold",
"buttercup",
"oxeye daisy",
"common dandelion",
"petunia",
"wild pansy",
"primula",
"sunflower",
"pelargonium",
"bishop of llandaff",
"gaura",
"geranium",
"orange dahlia",
"pink-yellow dahlia",
"cautleya spicata",
"japanese anemone",
"black-eyed susan",
"silverbush",
"californian poppy",
"osteospermum",
"spring crocus",
"bearded iris",
"windflower",
"tree poppy",
"gazania",
"azalea",
"water lily",
"rose",
"thorn apple",
"morning glory",
"passion flower",
"lotus lotus",
"toad lily",
"anthurium",
"frangipani",
"clematis",
"hibiscus",
"columbine",
"desert-rose",
"tree mallow",
"magnolia",
"cyclamen",
"watercress",
"canna lily",
"hippeastrum",
"bee balm",
"ball moss",
"foxglove",
"bougainvillea",
"camellia",
"mallow",
"mexican petunia",
"bromelia",
"blanket flower",
"trumpet creeper",
"blackberry lily"]

mat_data = scipy.io.loadmat("source/imagelabels.mat")
labels = mat_data['labels']

mat_data = scipy.io.loadmat("source/setid.mat")
train_split = mat_data['trnid']
test_split = mat_data['tstid']
val_split = mat_data['valid']

#train_split.shape, test_split.shape, val_split.shape

def process_split(root, split_list, split_name):
    for current in split_list:
        file_index = int(current)
        image_name = f"image_{file_index:05}.jpg"
        
        label = labels[0, file_index - 1]
        name = names[label - 1].replace(" ", "_").replace("-", "_")
        #print(f"index {current} | {label} | {name}")
        folder = os.path.join(root, split_name, name)
        
        if not os.path.isdir(folder):
            os.makedirs(folder)

        src = os.path.join("source", "jpg", image_name)
        dst = os.path.join(folder, image_name)
        shutil.copyfile(src, dst)

output_flower = "flowers"
process_split(output_flower, train_split[0], "train")
process_split(output_flower, test_split[0], "test")
process_split(output_flower, val_split[0], "val")