# ImageFilter for using filter() function
from PIL import Image, ImageFilter

def GF(image):
    # Opening the image
    # (R prefixed to string in order to deal with '\' in paths)
    # image = Image.open("..//Image_output//n7.png")

    # Blurring image by sending the ImageFilter.
    # GaussianBlur predefined kernel argument
    im1 = image.filter(ImageFilter.GaussianBlur)
    # image.save("..//Image_output//GF.png")
    # # Displaying the image
    # image.show()
    return im1
