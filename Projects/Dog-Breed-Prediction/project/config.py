import os

from pathlib import Path


def get_project_root():
    """Returns root of the project.

        Parameters:
        -----------
        None

        Returns:
        -----------
        root: pathlib.Path
            Root to the project saved withing pathlib.Path object.
    """
    root = Path(__file__).parent.parent
    return root


APP_ROOT = get_project_root()
DATA_DIR = os.path.join(APP_ROOT, "static")
HAAR_CASCADES_DIR = os.path.join(APP_ROOT, "haarcascades")
SAVED_MODELS_DIR = os.path.join(APP_ROOT, "saved_models")

HUMAN_PREDICTOR_FILEPATH = os.path.join(HAAR_CASCADES_DIR, "haarcascade_frontalface_alt.xml")
DOGBREED_PREDICTOR_FILEPATH = os.path.join(SAVED_MODELS_DIR, "inceptionV3_dogbreed.h5")

DOG_BREED_CLASSES = ["Affenpinscher", "Afghan_hound", "Airedale_terrier", "Akita", "Alaskan_malamute",
                     "American_eskimo_dog", "American_foxhound", "American_staffordshire_terrier",
                     "American_water_spaniel", "Anatolian_shepherd_dog", "Australian_cattle_dog", "Australian_shepherd",
                     "Australian_terrier", "Basenji", "Basset_hound", "Beagle", "Bearded_collie", "Beauceron",
                     "Bedlington_terrier", "Belgian_malinois", "Belgian_sheepdog", "Belgian_tervuren",
                     "Bernese_mountain_dog", "Bichon_frise", "Black_and_tan_coonhound", "Black_russian_terrier",
                     "Bloodhound", "Bluetick_coonhound", "Border_collie", "Border_terrier", "Borzoi", "Boston_terrier",
                     "Bouvier_des_flandres", "Boxer", "Boykin_spaniel", "Briard", "Brittany", "Brussels_griffon",
                     "Bull_terrier", "Bulldog", "Bullmastiff", "Cairn_terrier", "Canaan_dog", "Cane_corso",
                     "Cardigan_welsh_corgi", "Cavalier_king_charles_spaniel", "Chesapeake_bay_retriever", "Chihuahua",
                     "Chinese_crested", "Chinese_shar-pei", "Chow_chow", "Clumber_spaniel", "Cocker_spaniel", "Collie",
                     "Curly-coated_retriever", "Dachshund", "Dalmatian", "Dandie_dinmont_terrier", "Doberman_pinscher",
                     "Dogue_de_bordeaux", "English_cocker_spaniel", "English_setter", "English_springer_spaniel",
                     "English_toy_spaniel", "Entlebucher_mountain_dog", "Field_spaniel", "Finnish_spitz",
                     "Flat-coated_retriever", "French_bulldog", "German_pinscher", "German_shepherd_dog",
                     "German_shorthaired_pointer", "German_wirehaired_pointer", "Giant_schnauzer",
                     "Glen_of_imaal_terrier", "Golden_retriever", "Gordon_setter", "Great_dane", "Great_pyrenees",
                     "Greater_swiss_mountain_dog", "Greyhound", "Havanese", "Ibizan_hound", "Icelandic_sheepdog",
                     "Irish_red_and_white_setter", "Irish_setter", "Irish_terrier", "Irish_water_spaniel",
                     "Irish_wolfhound", "Italian_greyhound", "Japanese_chin", "Keeshond", "Kerry_blue_terrier",
                     "Komondor", "Kuvasz", "Labrador_retriever", "Lakeland_terrier", "Leonberger", "Lhasa_apso",
                     "Lowchen", "Maltese", "Manchester_terrier", "Mastiff", "Miniature_schnauzer", "Neapolitan_mastiff",
                     "Newfoundland", "Norfolk_terrier", "Norwegian_buhund", "Norwegian_elkhound", "Norwegian_lundehund",
                     "Norwich_terrier", "Nova_scotia_duck_tolling_retriever", "Old_english_sheepdog", "Otterhound",
                     "Papillon", "Parson_russell_terrier", "Pekingese", "Pembroke_welsh_corgi",
                     "Petit_basset_griffon_vendeen", "Pharaoh_hound", "Plott", "Pointer", "Pomeranian", "Poodle",
                     "Portuguese_water_dog", "Saint_bernard", "Silky_terrier", "Smooth_fox_terrier", "Tibetan_mastiff",
                     "Welsh_springer_spaniel", "Wirehaired_pointing_griffon", "Xoloitzcuintli", "Yorkshire_terrier"]
