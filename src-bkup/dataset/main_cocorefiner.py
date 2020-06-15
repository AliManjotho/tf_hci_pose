from src.dataset.cocorefiner import COCORefiner

refiner = COCORefiner()
refiner.generateRefinedDataset(type='train', outputRootDir='datasets/coco2017_refined')
refiner.generateRefinedDataset(type='val', outputRootDir='datasets/coco2017_refined')