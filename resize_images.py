from PIL import Image
import os

landscapesRootDir = '/home/ec2-user/Documents/LandscapeImages'
listOfLandscapes = list()

for (dirpath, dirnames, filenames) in os.walk(landscapesRootDir):
    listOfLandscapes += [os.path.join(dirpath, file) for file in filenames]

for file in listOfLandscapes:
    if file.endswith(".jpg"):
        foo = Image.open(file)
        foo.save(file,optimize=True,quality=95)