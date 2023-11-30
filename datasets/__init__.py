from .seq_cifar10 import get_cifar10_dataloaders
from .seq_cifar100 import get_cifar100_dataloaders, TCIFAR100
from .seq_tinyimagenet import get_tinyimagenet_dataloaders, TinyImagenet



def get_dataloaders(args, t, dataloaders_test):
    if args.dataset.name == 'seq-cifar100':
        train_dataloader, dataloaders_test, data_train_nums = get_cifar100_dataloaders(args, t, dataloaders_test)
    elif args.dataset.name == 'seq-cifar10':
        train_dataloader, dataloaders_test, data_train_nums = get_cifar10_dataloaders(args, t, dataloaders_test)
    elif args.dataset.name == 'seq-tinyimagenet':
        train_dataloader, dataloaders_test, data_train_nums = get_tinyimagenet_dataloaders(args, t, dataloaders_test)

    return train_dataloader, dataloaders_test, data_train_nums


def get_classnames(args):
    if args.dataset.name == 'seq-cifar10':
        classes_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    elif args.dataset.name == 'seq-cifar100' or args.dataset.name == 'seq-cifar100-fewshot':
        classes_name = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                        'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                        'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                        'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                        'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
                        'porcupine','possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
                        'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                        'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    elif args.dataset.name == 'seq-tinyimagenet' or args.dataset.name == 'seq-tinyimagenet-fewshot':
        classes_name = ['Egyptian Mau', 'fishing casting reel', 'volleyball', 'rocking chair', 'lemon', 'American bullfrog',
                 'basketball', 'cliff', 'espresso', 'plunger', 'parking meter', 'German Shepherd Dog', 'dining table',
                 'monarch butterfly', 'brown bear', 'school bus', 'pizza', 'guinea pig', 'umbrella', 'pipe organ', 'oboe',
                 'maypole', 'goldfish', 'pot pie', 'hourglass', 'beach', 'computer keyboard', 'arabian camel', 'ice cream',
                 'metal nail', 'space heater', 'cardigan', 'baboon', 'snail', 'coral reef', 'albatross', 'spider web',
                 'sea cucumber', 'backpack', 'Labrador Retriever', 'pretzel', 'king penguin', 'sulphur butterfly', 'tarantula',
                 'red panda', 'soda bottle', 'banana', 'sock', 'cockroach', 'missile', 'beer bottle', 'praying mantis',
                 'freight car', 'guacamole', 'remote control', 'fire salamander', 'lakeshore', 'chimpanzee', 'payphone',
                 'fur coat', 'mountain', 'lampshade', 'torch', 'abacus', 'moving van', 'barrel', 'tabby cat', 'goose', 'koala',
                 'high-speed train', 'CD player', 'teapot', 'birdhouse', 'gazelle', 'academic gown', 'tractor', 'ladybug',
                 'miniskirt', 'Golden Retriever', 'triumphal arch', 'cannon', 'neck brace', 'sombrero',
                 'gas mask or respirator', 'candle', 'desk', 'frying pan', 'bee', 'dam', 'spiny lobster', 'police van', 'iPod',
                 'punching bag', 'lighthouse', 'jellyfish', 'wok', "potter's wheel", 'sandal', 'pill bottle', 'butcher shop',
                 'slug', 'pig', 'cougar', 'construction crane', 'vestment', 'dragonfly', 'automated teller machine', 'mushroom',
                 'rickshaw', 'water tower', 'storage chest', 'snorkel', 'sunglasses', 'fly', 'limousine', 'black stork',
                 'dugong', 'sports car', 'water jug', 'suspension bridge', 'ox', 'popsicle', 'turnstile', 'Christmas stocking',
                 'broom', 'scorpion', 'wooden spoon', 'picket fence', 'rugby ball', 'sewing machine', 'through arch bridge',
                 'Persian cat', 'refrigerator', 'barn', 'apron', 'Yorkshire Terrier', 'swim trunks', 'stopwatch',
                 'lawn mower', 'thatched roof', 'fountain', 'southern black widow', 'bikini', 'plate', 'teddy bear',
                 'barbershop', 'candy store', 'station wagon', 'scoreboard', 'orange', 'flagpole', 'American lobster',
                 'trolleybus', 'drumstick', 'dumbbell', 'brass memorial plaque', 'bow tie', 'convertible', 'bighorn sheep',
                 'orangutan', 'American alligator', 'centipede', 'syringe', 'go-kart', 'brain coral', 'sea slug',
                 'cliff dwelling', 'mashed potatoes', 'viaduct', 'military uniform', 'pomegranate', 'chain', 'kimono',
                 'comic book', 'trilobite', 'bison', 'pole', 'boa constrictor', 'poncho', 'bathtub', 'grasshopper',
                 'stick insect', 'Chihuahua', 'tailed frog', 'lion', 'altar', 'obelisk', 'beaker', 'bell pepper',
                 'baluster', 'bucket', 'magnetic compass', 'meatloaf', 'gondola', 'Standard Poodle', 'acorn',
                 'lifeboat', 'binoculars', 'cauliflower', 'African bush elephant']

    return classes_name

