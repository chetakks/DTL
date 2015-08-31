import numpy as numpy

grey_levels = 256
# Generate a test image
test_image = numpy.random.randint(0,grey_levels, size=(11,11))

# Define the window size
windowsize_r = 5
windowsize_c = 5
count = 0
# Crop out the window and calculate the histogram
print test_image.shape[0]- windowsize_r
print test_image.shape[1]
print range(0,test_image.shape[0]+1 - windowsize_r, windowsize_r)
#print range(0,test_image.shape[0] - windowsize_r)
for r in range(0,test_image.shape[0]+1 - windowsize_r, windowsize_r):
    for c in range(0,test_image.shape[1]+1 - windowsize_c, windowsize_c):
        window = test_image[r:r+windowsize_r,c:c+windowsize_c]
        print window
        #hist = numpy.histogram(window,bins=grey_levels)
        #print hist
        count += 1
print count
        
print test_image