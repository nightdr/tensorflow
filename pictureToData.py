from PIL import Image
def getData(folder):

    data = []

    for i in range(1, 27):

        im = Image.open("C:\\Users\\David\\Desktop\\Letter Grid\\Letters\\%s\\%d.png" % (folder, i))

        imgData = list(im.getdata())

        avgList = []

        for index in range(0, len(imgData)):
            r, g, b = imgData[index]
            avg = int((r + g + b) / 3)
            avgList.append(avg)

        data.append(avgList)

    return data

print(getData("Times"))