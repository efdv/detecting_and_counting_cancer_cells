rootSave = '../dataset/crops/'

dirfile = glob.glob(general_root+'/*')
I = cv2.imread(general_root+'/Dataset_3/3-18.jpg')
#I = cv2.imread(r'C:\Users\ef.duquevazquez\Desktop'+'/img.jpeg')
u,v,c = I.shape

gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
_,thresholding  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

thresholding_neg =  negative(thresholding)



kernel = np.ones((7, 7), np.uint8)
imerode = cv2.erode(thresholding_neg, kernel, iterations=2)  

img_conn, bcentros, fcentros = bycompconn(imerode, 30)

w = h = 30
for c1, c2 in zip(bcentros, fcentros):
    #x1, y1, x2, y2 = getsquare(c1, w,h)
    #cv2.rectangle(I, (x1, y1), (x2, y2), (0, 255, 0), 1 )
    x1, y1, x2, y2 = getsquare(c2, w,h)
    cv2.rectangle(I, (x1, y1), (x2, y2), (0, 0, 255), 1 )

# img_over = over( I,img_conn)

# tamW = 60
# saveimages(I, bcentro, fcentros, tamW)



#img_over = img_over.astype(np.uint8)
showImg(I)