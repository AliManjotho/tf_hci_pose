from src.dataset.coco_test import COCO17

#sigma = 5
#sigmax = 20
#sigmay = 20
#joint = (205, 270)
#img = cv2.imread("images\person.jpg")
#hmap = getHeatmap_OP(img, joint, sigma)
#hmapImage = getHeatmapImage_OP(img, joint, sigma)
#hmapImage2 = getHeatmapImage(img, joint, sigmax, sigmay)

#print(hmap)
#cv2.imshow("Heatmap Image", hmapImage2)
#key = cv2.waitKey(0)

#if key == 27:
#    exit(0)

coco17 = COCO17()