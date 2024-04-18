import numpy as np
import cv2
import time 
from PyQt5.QtWidgets import QFileDialog
from collections import Counter
from Features.sift import SIFT




class TemplateMatching:
    def __init__(self,tab_widget) :
        self.ui = tab_widget
        self.img = None
        self.template_img = None
        self.descriptor_1, self.key_points_1 = None, None
        self.descriptor_2, self.key_points_2 = None, None


    def handle_buttons(self):
        self.ui.matching_method_selection.currentTextChanged.connect(self.matching_image)   
        self.ui.upload_image1.clicked.connect(lambda:self.browse_image(1))
        
        self.ui.upload_image2.clicked.connect(lambda:self.browse_image(0))
    

    def browse_image(self,idx):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self.ui, "Select Image", "",
                                                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.webp)",
                                                options=options)
        # sift = cv2.SIFT_create()


        
        if idx :
            self.img = cv2.imread(file_path)
            self.ui.display_image(self.ui.originial_image_graph,self.img)
            ## get descriptor and keypointsfor image 
            # self.key_points_1, self.descriptor_1 = sift.detectAndCompute(self.img,None)
            sift_instance = SIFT(self.img,self.ui)
            self.key_points_1, self.descriptor_1 = sift_instance.sift()
            
        else:
            self.template_img = cv2.imread(file_path)  
            self.ui.display_image(self.ui.template_image_graph,self.template_img)  
            ## get descriptor and keypointsfor image 
            # self.key_points_2, self.descriptor_2 = sift.detectAndCompute(self.template_img,None)
            sift_instance = SIFT(self.template_img,self.ui)
            self.key_points_2, self.descriptor_2 = sift_instance.sift()
                                                
  
    def Normalised_Cross_Correlation(self,roi, target):
       # Normalised Cross Correlation Equation
        # roi -=  np.mean(roi)
        # target -= np.mean(target)
        corr=np.sum(roi*target)
        norm = np.sqrt((np.sum(roi**2)))*np.sqrt(np.sum(target**2))
        return corr / norm      
    

    def Normalised_Sum_Square_difference(self,roi, target):
    # Normalised Cross Correlation Equation
        SSD=np.sum((roi-target)**2)
        norm = np.sqrt((np.sum(roi**2)))*np.sqrt(np.sum(target**2))
        return SSD / norm


    def template_matching(self):
        key2_indices = []
        matching_indics = []
        list_of_scores = []
        method = self.ui.matching_method_selection.currentText()
      

        for i in range(len(self.descriptor_1)):
            if method == "NCC":
                threshold_value = 0
            else:
                threshold_value = 10   
            target_index = 0
            for j in range(len(self.descriptor_2)):
                if method == "NCC":
                    score = self.Normalised_Cross_Correlation(self.descriptor_1[i],self.descriptor_2[j])
                    if score > threshold_value :
                        threshold_value  = score
                        target_index = j
            #         print(f"key:{i,target_index}")

                else:
                    score = self.Normalised_Sum_Square_difference(self.descriptor_1[i],self.descriptor_2[j])
                    if score < threshold_value :
                        threshold_value  = score
                        target_index = j
                
            key2_indices.append(target_index)
            list_of_scores.append(threshold_value)
            matching_indics.append([(self.key_points_1[i].pt[1],self.key_points_1[i].pt[0]),(self.key_points_2[target_index].pt[1],self.key_points_2[target_index].pt[0])])           
    
        return matching_indics,list_of_scores,key2_indices 
    
    
    def filter_repeated_points(self,matching_indics,list_of_scores,key2_indices):
        method = self.ui.matching_method_selection.currentText()
        repeated_val = Counter(key2_indices)
        repeated_val_indices = {}
        new_matching_indics = []
        new_list_of_scores = []
        for idx, val in enumerate(key2_indices):
            repeated_val_indices[val] = []
        
        for idx, val in enumerate(key2_indices):
            for key,no_of_repeated in repeated_val.items():
                if key == val:
                    if no_of_repeated > 1:
                        repeated_val_indices[key].append(idx)         
                    else:
                        new_matching_indics.append(matching_indics[idx])
                        new_list_of_scores.append(list_of_scores[idx])
    

        for key,indices_list in repeated_val_indices.items():
            if method == "NCC":
                threshold_value = 0
            else:
                threshold_value = 10

            for i in indices_list:
                if method == "NCC":
                    if list_of_scores[i] > threshold_value:
                        threshold_value = list_of_scores[i]
                        target_idx = i
                        new_matching_indics.append(matching_indics[target_idx])
                        new_list_of_scores.append(list_of_scores[target_idx])    
                else:
                    if list_of_scores[i] < threshold_value:
                        threshold_value = list_of_scores[i]
                        target_idx = i
                        new_matching_indics.append(matching_indics[target_idx])
                        new_list_of_scores.append(list_of_scores[target_idx])


            
    
    
        return new_matching_indics, new_list_of_scores


    def get_max_points_corr(self,list_of_scores,list_matches):
        max_matches_list = []
        method = self.ui.matching_method_selection.currentText()
        threshold_val = max(list_of_scores)
        for i ,score in enumerate(list_of_scores):
            if method == "NCC":
                if score > 0.95*threshold_val:
                    max_matches_list.append(list_matches[i])
            else:
                if score < 0.25*threshold_val:  ##0.08
                    max_matches_list.append(list_matches[i])        
        
        return max_matches_list                
    

    def draw_matches(self,list_matches,img1,img2):
        combined_image = np.zeros((img1.shape[0],img1.shape[1]+img2.shape[1],3),dtype="uint8")
        combined_image[0:img1.shape[0],0:img1.shape[1]] = img1
        combined_image[0:img2.shape[0],img1.shape[1]:] = img2

        for ky_pt1,ky_pt2 in list_matches:
            # print(ky_pt1,ky_pt2)
            pt1 = (int(round(ky_pt1[1])), int(round(ky_pt1[0])))  # x and y are swapped because of the row-column format
            pt2 = (int(round(ky_pt2[1] + img1.shape[1])), int(round(ky_pt2[0])))  # Add img1's width to x coordinate
            # print(f"points:{pt1,pt2}")
            # Draw line between keypoints
            combined_image = cv2.line(combined_image, pt1, pt2, (0, 0, 255), 1)

        
        return combined_image


    def matching_image(self):
        start_time = time.time()       
        list_matches, list_of_scores,key2_indices = self.template_matching()
        max_matches_list,new_list_of_scores = self.filter_repeated_points(list_matches, list_of_scores,key2_indices)
        max_matches_list = self.get_max_points_corr(new_list_of_scores,max_matches_list) 
        end_time = time.time()         
        computation_time = end_time - start_time      # Calculate the computation time
        print("Computation time:", computation_time, "seconds")  
        self.ui.computation_time_text.clear()
        self.ui.computation_time_text.append(str(np.round(computation_time,2))+str("  Sec"))
        image = self.draw_matches(max_matches_list,self.img,self.template_img)
       
        self.ui.display_image(self.ui.matching_result_graph,image)
