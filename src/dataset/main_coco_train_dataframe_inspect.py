from src.dataset.cocodataset import COCODataset

coco17 = COCODataset()
coco17.load()
trainDataInstances = coco17.getTrainInstancesAsPD()

print(trainDataInstances.info())
