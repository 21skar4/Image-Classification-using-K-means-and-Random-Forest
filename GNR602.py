#importing required libraries
import numpy as np
import cv2 
from skimage.feature import graycomatrix, graycoprops # for GLCM
from sklearn.model_selection import train_test_split #for test and train data spliting
from sklearn.svm import SVC #SVM
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import time
import tkinter as tk
from tkinter import filedialog
#FOR OPENING THE FILE LOCATION FOR IMAGE
# Create a Tkinter window
root = tk.Tk()
# Hide the main window since we only need the file dialog
root.withdraw()
# Open a file dialog and allow the user to select a file
print("Select image file")
file_path = filedialog.askopenfilename()
# Display the selected file path
print("Selected file path:", file_path)
# Ask user to select a directory to save files
print("Select Output save location")
save_location = filedialog.askdirectory()
print("Selected Directory:", save_location)

#read the image
image = cv2.imread(file_path)
#convert to greyscale
img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#print(img_grey)
#print(image)
#shape of the image pixel array
image_shape = np.shape(image)
height = image_shape[0]
width = image_shape[1]
number_colors = image_shape[2]
print('Height',height,'Width',width,'Depth',number_colors)
print("Number of Pixels",height*width)
#display the image
# Create a named window
cv2.namedWindow('ORIGINAL IMAGE', cv2.WINDOW_NORMAL)
# Set the window size
cv2.resizeWindow('ORIGINAL IMAGE', 750, 750)
# Move the window to a new location
cv2.moveWindow('ORIGINAL IMAGE', 0, 0)
#display image
cv2.imshow("ORIGINAL IMAGE",image)
cv2.waitKey(10)
#save image
image_file_name = save_location + '/ORIGINAL IMAGE.jpg'
cv2.imwrite(image_file_name,image)
cv2.destroyAllWindows()




count=0
prev_count = 0
while (count== prev_count):
  # Ask user for input
  user_input = input("Do you want to blur the image? (yes/no): ")
  count+=1
# Check user input
  if user_input.lower() == "yes":
     blur = 1 #boolean to decide if need to use blur
  elif user_input.lower() == "no":
     blur = 0
  else:
     print("Invalid input.")
     prev_count+=1

if blur :
   # apply median smoothing with kernel size of 5x5
   img_grey = cv2.medianBlur(img_grey, 9)


#shape of the image pixel array
img_grey_shape = img_grey.shape
#print(img_grey_shape)

#display gray image
# Create a named window
cv2.namedWindow('GREY IMAGE', cv2.WINDOW_NORMAL)
# Set the window size
cv2.resizeWindow('GREY IMAGE', 750, 750)
# Move the window to a new location
cv2.moveWindow('GREY IMAGE', 751, 0)
cv2.imshow("GREY IMAGE",img_grey)
grey_file_name = save_location + '/GREY IMAGE.jpg'
cv2.imwrite(grey_file_name,img_grey)
cv2.waitKey(10)
cv2.destroyWindow('GREY IMAGE')


start_time = time.time()
#K means - Ready the data
# Convert the image to a 1D array of pixel values
pixels = img_grey.flatten()
# Convert the pixel values to float32 type
pixels = np.float32(pixels)
#print(pixels)
#print(pixels.size)

# Define the number of clusters (k)
print("Input number of clusters required")
k_input = input()
k_input = int(k_input)

# Define the criteria for the k-means algorithm
print("Input Number of Iterations for K means")
iterations = input()
iterations = int(iterations)
print("Input Epsilon for K Means")
epsilon = input()
epsilon = float(epsilon)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)

#Initializing cluster centers - Random method
flags1 = cv2.KMEANS_RANDOM_CENTERS 
# without library
#flags1 = np.random.randint(0, 255, size=(k, 1)).astype(np.float32)

# Perform k-means clustering on the pixel values
_, labels, centers = cv2.kmeans(pixels, k_input, None, criteria, 10, flags1)

#k means - Extraction
# Convert the centers of the clusters to integer values
centers = np.uint8(centers)

# Replace each pixel value in the original image with the center of its corresponding cluster
labels = labels.flatten()
segmented_img = centers[labels]

# Reshape the segmented image to its original shape
segmented_img = segmented_img.reshape(img_grey.shape)

# Display the original and segmented images
# Create a named window
cv2.namedWindow('SEGMENTED IMAGE', cv2.WINDOW_NORMAL)
# Set the window size
cv2.resizeWindow('SEGMENTED IMAGE', 750, 750)
# Move the window to a new location
cv2.moveWindow('SEGMENTED IMAGE', 751, 0)
cv2.imshow('SEGMENTED IMAGE',segmented_img)
segmented_file_name = save_location + '/SEGMENTED IMAGE.jpg'
cv2.imwrite(segmented_file_name,segmented_img)
#print(segmented_img.shape)
cv2.waitKey(10) 
cv2.destroyWindow("SEGMENTED IMAGE")

#segmented image in color
color_segmented = np.zeros((height,width,3))
unique_pixels = np.unique(centers)
for i in range(height):
  for j in range(width):
    if segmented_img[i,j] == unique_pixels[0]:
     color_segmented[i,j] = [255,0,0]
    elif segmented_img[i,j] == unique_pixels[1]:
     color_segmented[i,j] = [0,255,0]
    elif segmented_img[i,j] == unique_pixels[2]:
     color_segmented[i,j] = [0,0,255]
    elif segmented_img[i,j] == unique_pixels[3]:
     color_segmented[i,j] = [3,82,162]
    elif segmented_img[i,j] == unique_pixels[4]:
     color_segmented[i,j] = [106,0,106]
    elif segmented_img[i,j] == unique_pixels[5]:
     color_segmented[i,j] = [130,130,0]
    elif segmented_img[i,j] == unique_pixels[6]:
     color_segmented[i,j] = [0,161,161]
    elif segmented_img[i,j] == unique_pixels[7]:
     color_segmented[i,j] = [127,10,68] 
    elif segmented_img[i,j] == unique_pixels[8]:
     color_segmented[i,j] = [5,0,0]
    elif segmented_img[i,j] == unique_pixels[9]:
     color_segmented[i,j] =  [6,128,255]
    elif segmented_img[i,j] == unique_pixels[10]:
     color_segmented[i,j] =  [102,4,51]
    elif segmented_img[i,j] == unique_pixels[11]:
     color_segmented[i,j] = [153,5,6]


color_segmented_file_name = save_location + '/COLOR SEGMENTED IMAGE.jpg'
cv2.imwrite(color_segmented_file_name, color_segmented)
# Create a named window
cv2.namedWindow('COLOR SEGMENTED IMAGE', cv2.WINDOW_NORMAL)
# Set the window size
cv2.resizeWindow('COLOR SEGMENTED IMAGE', 750, 750)
# Move the window to a new location
cv2.moveWindow('COLOR SEGMENTED IMAGE', 751, 0)
cv2.imshow('COLOR SEGMENTED IMAGE',color_segmented)
cv2.waitKey(10)
cv2.destroyAllWindows()
#print(color_segmented,unique_pixels)

# display the original and segmented images side by side
combined = np.hstack((image,color_segmented))
combined_file_name = save_location + '/Original vs Segmented.jpg'
cv2.imwrite(combined_file_name,combined)
cv2.imshow('Original vs Segmented', combined)
cv2.waitKey(10)
cv2.destroyAllWindows()

end_time = time.time()

kmeans_run_time = end_time - start_time

""" #Compute GLCM and extract features


start_time = time.time()
img_features = [] #for storing the features

window_size = 5
#directions for which GLCM is to be measured 
angles = [0,np.pi/4,np.pi/2]
count2 = 0
#distance for which GLCM is to be measured 
distances =  [1, 2, 3]

for i in range(height*width):
        count2+=1
        row, col = np.unravel_index(i, (height, width))
        window = img_grey[row:row+window_size, col:col+window_size]
        glcm = graycomatrix(window, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        feature = np.hstack([graycoprops(glcm, prop).ravel() for prop in ['contrast', 'homogeneity', 'energy']])
        img_features.append(feature)
        print(count2)  

#Save GLCM data
np.save('GLCM_4', img_features)
     
end_time = time.time()
GLCM_extraction_time = end_time - start_time 
 """

#FOR OPENING THE FILE LOCATION FOR GLCM DATA
#For presentation purpose we are using saved GLCM as calculating GLCM is time consuming 
# Create a Tkinter window
root = tk.Tk()
# Hide the main window since we only need the file dialog
root.withdraw()
# Open a file dialog and allow the user to select a file
print("Select GLCM file")
file_path1 = filedialog.askopenfilename()
# Display the selected file path
print("Selected file path for GLCM:", file_path1)

#Loading the GLCM saved data
img_features = np.load(file_path1, allow_pickle=True)

start_time = time.time()  
# Select samples from each cluster
#it works on the fact that labels entrie are corresponding to the imgae entries in 1-D form ,i.e., they have same indices 
print("Input number of samples per cluster")
samples_per_cluster = input()
samples_per_cluster = int(samples_per_cluster)  # number of samples to select from each cluster

samples = []
for i in range(k_input):
    cluster_indices = np.where(labels == i)[0] #returns indices of the pixels in the cluster i 
    sample_indices = np.random.choice(cluster_indices, size=samples_per_cluster, replace=True) #randomly select indices as sample from the cluster indices
    samples.append(sample_indices) #add all the indices to the samples list

#@title GLCM texture feature extraction on each sample

train_test_features = [] #for storing the features
train_test_labels = [] #for storing labels
count = 0


for j in range(k_input):
    for sample in samples[j]:
        count+=1
        row, col = np.unravel_index(sample, (height, width))
        pos = row*width + col 
        feature = img_features[pos]
        train_test_features.append(feature)
        train_test_labels.append(j)
        #print(count) 


#Training and Testing data
X_train, X_test, y_train, y_test = train_test_split(train_test_features, train_test_labels, test_size=0.25, random_state=32)

end_time = time.time()
training_testing_data_extraction_time = end_time - start_time

print("Please wait.........")


# Random Forest (RF) classifier
start_time = time.time()
rf = RandomForestClassifier(n_estimators=200, max_depth=21, random_state=22)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
print("RF accuracy:", rf_score)

# Save SVM model to disk
Rf_model_file_name = save_location + '/rf_model2.joblib'
joblib.dump(rf, Rf_model_file_name)

# Load the model from the file
#load_rf = joblib.load('C:\\Users\\bhask\\OneDrive\\Desktop\\Final\\Sample 2\\rf_model2.joblib')
#if trained in real time
load_rf = rf

# Use the loaded model to make predictions on the image
predict_rf = load_rf.predict(img_features)
class_rf = np.unique(predict_rf)

#create an empty array of size that of image to create an classified image
img_classified_rf = np.zeros([height,width,3])

for i in range(k_input):
  image_indices = np.where(predict_rf == i)[0]
  for indices in image_indices:
      row, column = np.unravel_index(indices, (height, width))
      if i == 0:       
         img_classified_rf[row,column] = [255,0,0]
      elif i ==1:
        img_classified_rf[row,column] = [1,250,2]
      elif i ==2:
        img_classified_rf[row,column] = [1,3,250]
      elif i == 3:         
         img_classified_rf[row,column] = [3,2,4]
      elif i ==4:
        img_classified_rf[row,column] = [130,130,5]
      elif i ==5:
        img_classified_rf[row,column] = [4,161,161]
      elif i == 6:         
         img_classified_rf[row,column] = [127,10,68]
      elif i ==7:
        img_classified_rf[row,column] =  [106,3,106]
      elif i ==8:
        img_classified_rf[row,column] = [5,0,0]
      elif i == 9:         
         img_classified_rf[row,column] = [6,128,255]
      elif i ==10:
        img_classified_rf[row,column] =  [102,4,51]
      elif i ==11:
        img_classified_rf[row,column] = [153,5,6]


cv2.imshow('Classified Image RF',img_classified_rf)
classified_file_name = save_location + '/Classified Image RF.jpg'
cv2.imwrite(classified_file_name,img_classified_rf)#save the classified image
cv2.waitKey(10)  
cv2.destroyAllWindows()
end_time = time.time()
RF_run_time = end_time - start_time

print("kmeans_run_time: ", kmeans_run_time)
print("training_testing_data_extraction_time: ", training_testing_data_extraction_time)
#print("GLCM_extraction_time: ", GLCM_extraction_time)
print("RF_run_time: ", RF_run_time)
print("RF accuracy:", rf_score)

#Extract the given inputs in a inputs.txt file
parameters = save_location + '/inputs.txt'
with open(parameters, 'w') as f:
    f.write(f'Image location: {file_path}\n')
    f.write(f'GLCM location: {file_path1}\n')
    f.write(f'Number of pixels: {height*width}\n')
    f.write(f'Blured : {user_input}\n')
    f.write(f'Number of clusters: {k_input}\n')
    f.write(f'Max Iterations of kmeans: {iterations}\n')
    f.write(f'Epsilon of kmeans: {epsilon}\n')
    f.write(f'Number of Samples per cluster: {samples_per_cluster}\n')
    f.write(f'RF accuracy: {rf_score}\n')
 


# display the original and segmented images side by side
combined2 = np.hstack((image,img_classified_rf))
combined2_file_name = save_location + '/Original vs RF Classified.jpg'
cv2.imwrite(combined2_file_name,combined2)
cv2.imshow('Original vs RF Classified', combined2)
cv2.waitKey(10)
cv2.destroyAllWindows()

