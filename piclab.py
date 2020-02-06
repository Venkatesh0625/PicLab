from tkinter import *
import matplotlib.pyplot as plt 
from tkinter import *
import matplotlib.image as mping
import matplotlib.pyplot as plt 
import cv2
import numpy as np
import sys
from tkinter import filedialog
from scipy import fftpack
gray =cv2.imread("E:/Venki2.jpg")
compressed_img1 = cv2.imread("E:/Venki2.jpg")
compressed_img2 = cv2.imread("E:/Venki2.jpg")
compressed_img3 = cv2.imread("E:/Venki2.jpg")
gui=Tk()
#For identification whether the image compressed or not
identity = [0,0,0,0,0]
file = "C:/Users/Faker/Downloads/123.jpg"
gui.configure(background="#313b4c")
gui.title("Image Processing")
gui.geometry("500x700") 

def load_image():
    gui.filename = filedialog.askopenfilename()
    global file
    file = gui.filename
    #Loading Image
    cv_img = cv2.imread(file)
    #Concerting into Gray Scale to obtain 2D Array Image
    global gray
    gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
    print(file)
def show_image_orig():
    plt.imshow(gray,cmap=plt.get_cmap('gray'))
    plt.show()

def denoise():
    im_fft = fftpack.fft2(gray)
    
    # In the lines following, we'll make a copy of the original spectrum and
    # truncate coefficients.

    # Define the fraction of coefficients (in each direction) we keep
    keep_fraction = 0.1

    # Call ff a copy of the original transform. Numpy arrays have a copy
    # method for this purpose.
    im_fft2 = im_fft.copy()

    # Set r and c to be the number of rows and columns of the array.
    r, c = im_fft.shape[:2]

    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
    
    # Similarly with the columns:
    im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
        
    # Reconstruct the denoised image from the filtered spectrum, keep only the
    # real part for display.
    im_new = fftpack.ifft2(im_fft2).real
    print("Denoised");
    global denoise_img 
    denoise_img = im_new
    identity[3]=1

def show_denoise():
    if(identity[3]==1):
        plt.figure()
        plt.imshow(denoise_img, plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
        plt.title('Reconstructed Image')
        

def compr(tem):
    print(file)
    cv_img = cv2.imread(file)
    gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
    #img=cv_img
    #Concerting into Gray Scale to obtain 2D Array Image

    #Getting Size 
    x,y=gray.shape[:2]
   
    #Changing Spatial Domain into Frequency Domain
    new=np.fft.fft2(gray)
    shift=np.fft.fftshift(new)

    #TakingAbsolute Values to ignore Negative values
    n=np.abs(shift)
    #Taking log to obtain high frequencies in the Middle
    f=np.log(n+1)
    plt.imshow(f,cmap=plt.get_cmap('gray'))
    temp=np.max(np.abs(new[:]))
    arr=np.array([0.005,0.01,0.05])
    if(tem == 1):
        t = arr[0]
    elif(tem == 2):
        t = arr[1]
    else:
        t = arr[2]
    print("sfhdgh",t)
    #print("Select A value from this to Compress ")
    #or i in range(len(arr)):
        #print(str(i+1)+" . "+str(a[i]))
    #t=int(input("Enter Choice : "))
    thresh_freq = 0.1*t*temp;
    #for thresh_freq in 0.1*arr*m:
    #This will bool with the value 1 if the frequency is greater than the threshhold frequency
    index=abs(new)>thresh_freq
    newfilter=np.multiply(new,index)
    count=x*y-np.sum(index[:])
    #Taking ifft to get back Spatial Domain Again 
    nfilt=np.fft.ifft2(newfilter)
    nfilt = np.abs(nfilt)
    print("adgrh")
    if(tem==1):
        global compressed_img1
        compressed_img1 = nfilt
        #To ensure Image is Compressed123
        identity[0]=1;
        print("sfhdgh")

    if(tem==2):
        global compressed_img2
        compressed_img2=np.abs(nfilt)
        #To ensure Image is Compressed123
        identity[1]=1;
    if(tem==3):
        global compressed_img3
        compressed_img3=np.abs(nfilt)
        #To ensure Image is Compressed123
        identity[2]=1;
    
    #To ensure Image is Compressed123
def compress1():
    compr(1)
def compress2():
    compr(2)
def compress3():
    compr(3)
def show_image_compr1():
    if(identity[0]==1):
        plt.imshow(compressed_img1,cmap=plt.get_cmap('gray'))
        plt.xticks([])
        plt.yticks([])
        plt.show()
def show_image_compr2():
    if(identity[1]==1):
        
        plt.imshow(compressed_img2,cmap=plt.get_cmap('gray'))
        plt.xticks([])
        plt.yticks([])
        plt.show()
def show_image_compr3():
    if(identity[2]==1):
        plt.imshow(compressed_img3,cmap=plt.get_cmap('gray'))
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
def exit_():
    gui.destroy()
   
open_=Button(gui,text=' Open File ',fg='black',bg='white',command=load_image,height=2,width=15) 
open_.place(x=60,y=20)
show=Button(gui,text=' Show Image ',fg='black',bg='white',command=show_image_orig,height=2,width=15) 
show.place(x=200,y=20)
sam3_=Button(gui,text=' Compress 1 ',fg='black',bg='white',command=compress1,height=2,width=15) 
sam3_.place(x=60,y=80)
sam4_=Button(gui,text=' Show Compressed 1 ',fg='black',bg='white',command=show_image_compr1,height=2,width=15) 
sam4_.place(x=200,y=80)
sam5_=Button(gui,text=' Compress 2 ',fg='black',bg='white',command=compress2,height=2,width=15) 
sam5_.place(x=60,y=140)
sam6_=Button(gui,text=' Show Compressed 2',fg='black',bg='white',command=show_image_compr2,height=2,width=15) 
sam6_.place(x=200,y=140)
sam7_=Button(gui,text=' Compress 3 ',fg='black',bg='white',command = compress3,height=2,width=15) 
sam7_.place(x=60,y=200)
sam8_=Button(gui,text=' Show Compress 3 ',fg='black',bg='white',command=show_image_compr3,height=2,width=15) 
sam8_.place(x=200,y=200)
sam9_=Button(gui,text=' Denoise ',fg='black',bg='white',command=denoise,height=2,width=15) 
sam9_.place(x=60,y=260)
sam10_=Button(gui,text=' Show Denoise ',fg='black',bg='white',command=show_denoise,height=2,width=15) 
sam10_.place(x=200,y=260)
sam16_=Button(gui,text=' Exit ',fg='black',bg='white',command=sys.exit,height=2,width=15) 
sam16_.place(x=200,y=320)

#Running Mainloop
gui.mainloop() 
