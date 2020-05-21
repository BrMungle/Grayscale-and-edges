import IPython.display as display
import PIL.Image
import numpy as np
#import matplotlib.pyplot as plt
#from skimage.color import rgb2gray

class my_image:
    def __init__(self,adress):
        self.image = np.array(PIL.Image.open(adress))
        self.rgb_to_gray()
        self.gray_to_rgb()
        self.get_gradient_magnitude_gray()
        self.get_gradients_rgb_Sobel()
        self.get_gradient_magnitude_rgb()
        
    def rgb_to_gray(self):
        self.image_gray = np.dot(self.image, [0.2989, 0.5870, 0.1140])
    
    def gray_to_rgb(self):
        self.image_gray_as_rgb = np.dstack([self.image_gray.astype('uint8')] * 3)
    
    def get_vertical_gradient(self,image):
        magnitude_x = 0.5 * (image[:,2:] - image[:,:-2])
        return(np.c_[np.zeros(image.shape[0]),  magnitude_x,np.zeros(image.shape[0])])
    
    def get_vertical_gradient_Sobel(self,image):
        magnitude_x =       (-1 * image[:-2,:-2] + 
                              1 * image[:-2,2:]  + 
                             -2 * image[1:-1,0:-2] +
                              2 * image[1:-1,2:] +
                             -1 * image[2:,:-2] +
                              1 * image[2:,2:] 
                             ) * (1.0/8.0)
        return(np.vstack([
               np.zeros(image.shape[1]), 
               np.c_[np.zeros(image.shape[0] -2),  magnitude_x,np.zeros(image.shape[0] - 2)],
               np.zeros(image.shape[1]) 
              ]))   

    def get_horisontal_gradient(self,image):
        magnitude_y = 0.5 * (image[2:,] - image[:-2,:])
        return(np.vstack([np.zeros(image.shape[1]),  magnitude_y,np.zeros(image.shape[1])]))
    
    def get_horisontal_gradient_Sobel(self,image):
        magnitude_y = 0.5 * (-1 * image[:-2,:-2] + 
                              1 * image[2:,:-2]  + 
                             -2 * image[:-2,1:-1] +
                              2 * image[2:,1:-1] +
                             -1 * image[:-2,2:] +
                              1 * image[2:,2:] 
                             )
        return(np.vstack([
               np.zeros(image.shape[1]), 
               np.c_[np.zeros(image.shape[0] -2),  magnitude_y,np.zeros(image.shape[0] - 2)],
               np.zeros(image.shape[1]) 
              ])) 
    
    def get_gradient_magnitude_gray(self):
        self.gradient_y_gray = self.get_horisontal_gradient(self.image_gray)
        self.gradient_x_gray = self.get_vertical_gradient(self.image_gray)
        self.gradient_magnitude_gray = (self.gradient_x_gray**2 + self.gradient_y_gray**2)**(0.5)
    
    def get_gradients_rgb(self):
        self.gradient_y_r = self.get_horisontal_gradient(self.image[...,0])
        self.gradient_x_r = self.get_vertical_gradient(self.image[...,0])
        self.gradient_y_g = self.get_horisontal_gradient(self.image[...,1])
        self.gradient_x_g = self.get_vertical_gradient(self.image[...,1])
        self.gradient_y_b = self.get_horisontal_gradient(self.image[...,2])
        self.gradient_x_b = self.get_vertical_gradient(self.image[...,2])
        
        self.gradient_magnitude_r = (self.gradient_x_r**2 + self.gradient_y_r**2)**(0.5)
        self.gradient_magnitude_g = (self.gradient_x_g**2 + self.gradient_y_g**2)**(0.5)
        self.gradient_magnitude_b = (self.gradient_x_b**2 + self.gradient_y_b**2)**(0.5)
    
    def get_gradients_rgb_Sobel(self):
        self.gradient_y_r = self.get_horisontal_gradient_Sobel(self.image[...,0])
        self.gradient_x_r = self.get_vertical_gradient_Sobel(self.image[...,0])
        self.gradient_y_g = self.get_horisontal_gradient_Sobel(self.image[...,1])
        self.gradient_x_g = self.get_vertical_gradient_Sobel(self.image[...,1])
        self.gradient_y_b = self.get_horisontal_gradient_Sobel(self.image[...,2])
        self.gradient_x_b = self.get_vertical_gradient_Sobel(self.image[...,2])
        
        self.gradient_magnitude_r = (self.gradient_x_r**2 + self.gradient_y_r**2)**(0.5)
        self.gradient_magnitude_g = (self.gradient_x_g**2 + self.gradient_y_g**2)**(0.5)
        self.gradient_magnitude_b = (self.gradient_x_b**2 + self.gradient_y_b**2)**(0.5)
    
    def get_gradient_magnitude_rgb(self,norm = -1):
        
        if norm == -1:
            self.gradient_magnitude_rgb = np.max(np.dstack([self.gradient_magnitude_r,
                                                            self.gradient_magnitude_g,
                                                            self.gradient_magnitude_b]),
                                                            axis = 2)
        else:
            self.gradient_magnitude_rgb = np.sum(np.dstack([self.gradient_magnitude_r**norm,
                                                            self.gradient_magnitude_g**norm,
                                                            self.gradient_magnitude_b**norm]),
                                                            axis = 2)**(1.0/norm) / 3**(1.0/norm)
    
    def get_orientation_gray(self):
        rounded_gradients_x = np.cos(np.pi/8) * self.gradient_x_gray - np.sin(np.pi/8) * self.gradient_y_gray
        rounded_gradients_y = np.sin(np.pi/8) * self.gradient_x_gray + np.cos(np.pi/8) * self.gradient_y_gray
        
        indicator_temp = (rounded_gradients_y < 0)
        
        rounded_gradients_x = rounded_gradients_x -2 * rounded_gradients_x * indicator_temp.astype('uint8')
        rounded_gradients_y = rounded_gradients_y -2 * rounded_gradients_y * indicator_temp.astype('uint8')
        
        direction_0 = (rounded_gradients_x >= 0) & (rounded_gradients_x >= rounded_gradients_y)
        direction_1 = (rounded_gradients_x >= 0) & (rounded_gradients_x < rounded_gradients_y)
        direction_2 = (rounded_gradients_x < 0) & ( -1 * rounded_gradients_x >= rounded_gradients_y)
        direction_3 = (rounded_gradients_x < 0) & ( -1 * rounded_gradients_x < rounded_gradients_y)
        
        self.gradient_directions_gray = np.dstack([direction_0,
                                                   direction_1,
                                                   direction_2,
                                                   direction_3])
    
        return(rounded_gradients_x,rounded_gradients_y)
    
    def get_local_maximum(self,t):
        
        left_point = (self.gradient_magnitude_gray[1:-1,0:-2] * self.gradient_directions_gray[1:-1,1:-1,0].astype('uint8') +
                      self.gradient_magnitude_gray[0:-2,0:-2] * self.gradient_directions_gray[1:-1,1:-1,1].astype('uint8') +
                      self.gradient_magnitude_gray[0:-2,1:-1] * self.gradient_directions_gray[1:-1,1:-1,2].astype('uint8') +
                      self.gradient_magnitude_gray[0:-2,2:] * self.gradient_directions_gray[1:-1,1:-1,3].astype('uint8')
               )
        
        right_point = (self.gradient_magnitude_gray[1:-1,2:] * self.gradient_directions_gray[1:-1,1:-1,0].astype('uint8') +
                      self.gradient_magnitude_gray[2:,2:] * self.gradient_directions_gray[1:-1,1:-1,1].astype('uint8') +
                      self.gradient_magnitude_gray[2:,1:-1] * self.gradient_directions_gray[1:-1,1:-1,2].astype('uint8') +
                      self.gradient_magnitude_gray[2:,0:-2] * self.gradient_directions_gray[1:-1,1:-1,3].astype('uint8')
               )
        
        gradient_magnitude_gray_center = self.gradient_magnitude_gray[1:-1,1:-1]
        
        edge_points = ((gradient_magnitude_gray_center > t) &
                       (gradient_magnitude_gray_center > left_point) &
                       (gradient_magnitude_gray_center > right_point) ).astype('uint8')
        
        self.edge_points_gray = (np.vstack([
               np.zeros(self.gradient_magnitude_gray.shape[1]), 
               np.c_[np.zeros(self.gradient_magnitude_gray.shape[0] -2),  
                     edge_points,
                     np.zeros(self.gradient_magnitude_gray.shape[0] - 2)],
               np.zeros(self.gradient_magnitude_gray.shape[1]) 
              ])) 
    
    # to do: tracing the points near edges
    
    @staticmethod
    def display_picture(image):
        display.display(PIL.Image.fromarray(image))
    
    def show_original(self):
        self.display_picture(self.image)
    
    def show_gray(self):
        self.display_picture(self.image_gray_as_rgb)
    
    def show_edges_gray(self):
        white = np.repeat(255,self.gradient_magnitude_gray.shape[0] * self.gradient_magnitude_gray.shape[1])
        white = white.reshape(self.gradient_magnitude_gray.shape).astype('uint8')
        self.display_picture(np.dstack([white - (3 * self.gradient_magnitude_gray).astype('uint8')] * 3))
    
    def show_edges_rgb(self):
        #self.display_picture(np.dstack([self.gradient_magnitude_rgb.astype('uint8')] * 3))
        white = np.repeat(255,self.gradient_magnitude_gray.shape[0] * self.gradient_magnitude_gray.shape[1])
        white = white.reshape(self.gradient_magnitude_gray.shape).astype('uint8')
        self.display_picture(np.dstack([white - self.gradient_magnitude_rgb.astype('uint8')] * 3))
        
    

rudy = my_image('dd_net_1.jpeg')
rudy.show_original()
rudy.show_gray()
rudy.show_edges_gray()
rudy.show_edges_rgb()
rudy.get_gradient_magnitude_rgb(-1)
rudy.show_edges_rgb()

tab_test_6, tab_test_7 = rudy.get_orientation_gray()
tab_test_5 = rudy.gradient_directions_gray
rudy.get_local_maximum(15)
display.display(PIL.Image.fromarray((200 * rudy.edge_points_gray).astype('uint8')))

tab_test_8 = rudy.edge_points_gray

rudy.get_vertical_gradient_Sobel(rudy.image[...,2])

pies_wiekszy = my_image('dd_1_net.jpeg')
pies_wiekszy.show_original()
pies_wiekszy.show_gray()
pies_wiekszy.show_edges_gray()
pies_wiekszy.show_edges_rgb()
pies_wiekszy.get_gradient_magnitude_rgb(2)
pies_wiekszy.show_edges_rgb()

image_test = np.array(PIL.Image.open('logos.jpg'))

'''
def get_vertical_gradient(self):
        img_shape = self.image_gray.shape
        vertical_filter = np.array([-0.5,0,0.5])
        magnitude_x = []
        for y in range(0,img_shape[0],1):
            row_temp = [0]
            for x in range(1,img_shape[1]-1,1):
                row_temp.append(np.dot(self.image_gray[y,x-1:x+2],vertical_filter))
            row_temp.append(0)    
            magnitude_x.append(row_temp)
        
        magnitude_x = np.array(magnitude_x)
        self.magnitude_x =magnitude_x

    def get_horisontal_gradient(self):
        img_shape = self.image_gray.shape
        vertical_filter = np.array([-0.5,0,0.5])
        magnitude_y = []
        for x in range(0,img_shape[1],1):
            col_temp = [0]
            for y in range(1,img_shape[0]-1,1):
                col_temp.append(np.dot(self.image_gray[y-1:y+2,x],vertical_filter))
            col_temp.append(0)    
            magnitude_y.append(col_temp)

'''