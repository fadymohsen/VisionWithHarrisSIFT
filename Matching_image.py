import numpy as np
import cv2
import time 
from PyQt5.QtWidgets import QApplication, QTabWidget, QFileDialog
class Template_Matching:
    def __init__(self,main_window) :
        self.ui = main_window
        self.img = None
        self.template_img = None


    # def handle_buttons(self):
    #     self.ui.matching_image_btn.clicked.connect(self.matching_image) 
    #     self.ui.matching_method_selection.currentTextChanged.connect(self.matching_image)   
    #     self.ui.upload_image1.clicked.connect(lambda:self.browse_image(1))
        
    #     self.ui.upload_image2.clicked.connect(lambda:self.browse_image(0))
    


    def browse_image(self,idx):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.webp)",
                                                options=options)
        
        if idx :
            self.img = cv2.imread(file_path,0)
            self.ui.display_image(self.ui.originial_image_graph,self.img)
        else:
            self.template_img = cv2.imread(file_path,0)  
            self.ui.display_image(self.ui.template_image_graph,self.template_img)  
                                                


    def read_image(self,idx):
        if idx :
            self.img = cv2.imread(self.file_path,0)
            self.ui.display_image(self.ui.originial_image_graph,self.img)
        else:
            self.template_img = cv2.imread(self.file_path,0)  
            self.ui.display_image(self.ui.template_image_graph,self.template_img)  


    def Normalised_Cross_Correlation(self,roi, target):
    # Normalised Cross Correlation Equation
        corr=np.sum(roi*target)
        norm = np.sqrt((np.sum(roi**2)))*np.sqrt(np.sum(target**2))
        return corr / norm      
    

    def Normalised_Sum_Square_difference(self,roi, target):
    # Normalised Cross Correlation Equation
        SSD=np.sum((roi-target)**2)
        norm = np.sqrt((np.sum(roi**2)))*np.sqrt(np.sum(target**2))
        return SSD / norm  

    def template_matching(self):
        target_indics = (0,0)
        threshold_value = 0  
        method = self.ui.matching_method_selection.currentText()
        if method == "Cross Correlation":
            threshold_value = 0
        else:
            threshold_value = 10    
        result_shape = (self.img.shape[0]-self.template_img.shape[0]+1,self.img.shape[1]-self.template_img.shape[1]+1)
        template_img = np.zeros(result_shape)
        self.img=np.array(self.img, dtype="int")
        self.template_img=np.array(self.template_img, dtype="int")
        for i in range(result_shape[0]):
            for j in range(result_shape[1]):
                roi=self.img[i:i+self.template_img.shape[0],j:j+self.template_img.shape[1]]
                if method == "Cross Correlation":              
                    template_img[i,j]= self.Normalised_Cross_Correlation(roi,self.template_img)
                    
                    if template_img[i,j] > threshold_value:
                        threshold_value = template_img[i,j]
                        target_indics = (i,j)
                else:
                    template_img[i,j]= self.Normalised_Sum_Square_difference(roi,self.template_img)
                    
                    if template_img[i,j] < threshold_value:
                        threshold_value = template_img[i,j]
                        target_indics = (i,j)        
                
        return target_indics  
    
    def matching_image(self):
        start_time = time.time()       
        pt = self.template_matching()
        end_time = time.time()         
        computation_time = end_time - start_time      # Calculate the computation time
        print("Computation time:", computation_time, "seconds")  
        
        self.img = cv2.rectangle(self.img, 
                              (pt[1],pt[0]), (pt[1]+self.template_img.shape[1],pt[0]+self.template_img.shape[0]), 
                              (0,255,0), 2) 
        self.ui.display_image(self.ui.matching_result_graph,self.img)

              