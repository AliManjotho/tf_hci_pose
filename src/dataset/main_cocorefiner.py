from src.dataset.COCORefiner import COCORefiner

refiner = COCORefiner()
refiner.generateRefinedTrainingSet(outputRootDir='datasets/coco2017_refined')