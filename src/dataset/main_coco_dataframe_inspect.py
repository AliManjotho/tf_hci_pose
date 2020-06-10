from src.dataset.cocodataset import COCODataset

coco17 = COCODataset()
coco17.loadVal()
instances = coco17.getValidationInstancesAsPD()

print(instances.info())
