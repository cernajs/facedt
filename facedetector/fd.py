import cv2

trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('./group.jpg')

#print(img.shape)

#img = img.astype('uint8')

camera = cv2.VideoCapture(0)

#greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
while True:

	succes, frame = camera.read()

	cv2.imshow('img', frame)

	face_cor = trained_data.detectMultiScale(frame)

	for (x, y, width, height) in face_cor:
		cv2.rectangle(frame, (x, y), (x + width, y + height), (255,0,0), 2)	

	cv2.imshow('img', frame)	

	cv2.waitKey(1)

	#face_cor = trained_data.detectMultiScale(img)

	"""(x, y, width, height) = face_cor[0]
		cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
	 
	for (x, y, width, height) in face_cor:
		cv2.rectangle(img, (x, y), (x + width, y + height), (255,0,0), 2)

	cv2.imshow('img', img)
	#print(face_cor)

	cv2.waitKey()"""