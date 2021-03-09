'''
original dimension = 875*656
left cropped = 562*656
right cropped = 721*656
up_cropped = 875*607
down_cropped = 875*528
'''
from PIL import Image 
  
#Open the image 
im = Image.open("c1_LH_1.png") 
  
#Setting the points for cropped image 
left = 313
top = 49
right = 721
bottom = 528

#Crop the image
im1 = im.crop((left, top, right, bottom))

#Save the image into directory
im1 = im1.save("test.png")
