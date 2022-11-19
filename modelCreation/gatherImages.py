import cv2
from PIL import ImageGrab, Image
import time
import numpy as np
import os

import parameters

class ImageFormatNotSupported(Exception):
    def __init__(self, imageFormat) -> None:
        self.imageFormat = imageFormat
        self.message = f"Given image format {imageFormat} is not supported!"
        super().__init__(self.message)

    def __str__(self):
        return f'{self.imageFormat} -> {self.message}'

class ImageSaver():
    def __init__(self) -> None:
        # Initialize constants
        self.ROOT_IMAGES_FOLDER = self.getImagesFolder()
        self.IMAGE_FORMATS = self.getImageFormats()

        self.currentImageId = self.getLastImageId()

    # Get folder based on paramters.SAVE_FOLDER_PATH value
    def getImagesFolder(self):
        fullPath = os.path.realpath(__file__)
        return parameters.SAVE_FOLDER_PATH.replace(".", os.path.dirname(fullPath))
    
    # Image formats are specified as folder names inside ROOT_IMAGES_FOLDER. 
    # Currently supported formats are shown in parameters.ALLOWED_IMAGE_FORMATS
    def getImageFormats(self):
        result = []
        formats = os.listdir(self.ROOT_IMAGES_FOLDER)
        for i,imageFormat in enumerate(formats):
            if imageFormat in parameters.ALLOWED_IMAGE_FORMATS:
                result.append(imageFormat)

        return result

    def getLastImageId(self):
        fullPath = os.path.realpath(__file__)
        dirList = os.listdir(parameters.SAVE_FOLDER_PATH.replace(".", os.path.dirname(fullPath)) + "/unlabeledRaw/")
        if not dirList:
            return 0
        else:
            return int(max(dirList).replace(".jpg", "")) + 1

    # Processes the image for edge detection using canny algorithm
    # Canny algorithm thresholds are defined in paramteres.CANNY_THRESHOLD1 and 2
    def processToEdgeImage(self, frame):
        processed_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        processed_img = cv2.Canny(
                                frame, 
                                threshold1=parameters.CANNY_THRESHOLD1, 
                                threshold2=parameters.CANNY_THRESHOLD2
                            )
        return processed_img

    # Converts a raw image to an appropriate format
    def convertToImageFormat(self, frame, imageFormat):
        match imageFormat:
            case "unlabeledEdge":
                return self.processToEdgeImage(frame)
            case "unlabeledRaw":
                return frame
            case _:
                raise ImageFormatNotSupported(imageFormat)

    # Saves next frame based on last saved image index
    def saveNextFrame(self, frame):
        for imageFormat in self.IMAGE_FORMATS:
            currentFrame = self.convertToImageFormat(frame, imageFormat)
            im = Image.fromarray(currentFrame)
            im.save(self.ROOT_IMAGES_FOLDER + imageFormat + "/" + str(self.currentImageId) + ".jpg")
        self.currentImageId += 1


# Gets current frame in RGB
def getCurrentFrame():
    frame = np.array(ImageGrab.grab
                                    (
                                    bbox=(
                                        parameters.GAME_WIDTH/2 - parameters.GAME_HEIGHT/2,
                                        parameters.GAME_CAPTURE_OFFSET,
                                        parameters.GAME_WIDTH/2 + parameters.GAME_HEIGHT/2,
                                        parameters.GAME_HEIGHT + parameters.GAME_CAPTURE_OFFSET
                                        )
                                    )
                                )
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def main():
    imageSaver = ImageSaver()
    
    lastTime = time.time()
    while(True):
        frame = getCurrentFrame()
        
        cv2.imshow("window", frame)
        

        print(f"Loop took {time.time()-lastTime} s")
        lastTime = time.time()
        
        k = cv2.waitKey(1) & 0xFF

        # Take a screenshot
        if k == ord('.'):
            imageSaver.saveNextFrame(frame)
        if k == ord('c'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()