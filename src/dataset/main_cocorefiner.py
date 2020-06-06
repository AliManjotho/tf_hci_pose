from src.dataset.cocorefiner import COCORefiner

refiner = COCORefiner()
refiner.generateRefinedTrainingSet(outputRootDir='datasets/coco2017_refined')