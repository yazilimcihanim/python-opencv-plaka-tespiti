import cv2
import numpy as np
import pytesseract
import imutils

img1=cv2.imread("plaka.jpg")
boyut=(500,500)
img=cv2.resize(img1,boyut)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
filtre=cv2.bilateralFilter(gray,5,440,260)
kose_algilama=cv2.Canny(filtre,10,340)
sinir_bulma=cv2.findContours(kose_algilama,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
uygun_snr=imutils.grab_contours(sinir_bulma)
uygun_snr=sorted(uygun_snr,key=cv2.contourArea , reverse=True)[:10]
pencere=None

for c in uygun_snr:
    epsilon=0.018*cv2.arcLength(c ,True)
    approx=cv2.approxPolyDP(c , epsilon , True)
    if len(approx)==4:
        pencere=approx
        break

mask=np.zeros(gray.shape,np.uint8)
new_img=cv2.drawContours(mask, [pencere], 0 , (255,255,255),-1)
new_img=cv2.bitwise_and(img,img,mask=mask)

cv2.imshow("gri resim",gray)
#cv2.imshow("filtreli resim",filtre)
#cv2.imshow("köşeleri tespit edilmiş  resim",kose_algilama)
cv2.imshow("mask resim",new_img )
cv2.waitKey(0)
