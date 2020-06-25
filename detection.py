
from .social_distancing_config import NMS_THRESH
from .social_distancing_config import MIN_CONF
import numpy as np
import cv2
#frame dedigimiz video dosyamiz veya kameramizdir, net dedigimiz yolo nesne algilama modelimiz, ln dedigimiz ise yolo cnn cikti katman adlari person idx sadece kisi icin
def detect_people(frame, net, ln, personIdx=0):
	# cerceve boyutlarini gosteriyoruz
	# results ile sonuc listemizi baslatiyoruz
	(H, W) = frame.shape[:2]
	results = []

	# blob ile degerleri belirliyoruz int yaziyordu cok hatirlayamadim asdfgh en olasi degerler bunlar bu sekilde nesne tespiyi yapabiliyoruz.
	
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# sinirlayici kutu, centroids ve nesne algilama guven degerini listede baslattik
	
	boxes = []
	centroids = []
	confidences = []

	# donguye sokarak mevcut algilanan nesnenin classid ve guven noktasini cikaririz
	for output in layerOutputs:
		
		for detection in output:
			
			
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# mevcut tespitin bir kisi oldugunu ve minumum guvenin karsilanmasi
			
			if classID == personIdx and confidence > MIN_CONF:
				# sinarlayici kutu koordinatlari hesaplar
				
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	# nmsboxes in amaci kutu icinde aldiigmizda cok fazla kutu olabiliyor onu bastirmak icin methodu uygulariz.
	
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	
	if len(idxs) > 0:
		
		for i in idxs.flatten():
			# koordinatlari cikarir
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# en son guven degerini sinirlayici kutuyu ve her insanin centroid ni cagirip donduruyoruz.
			
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	
	return results
