from datetime import datetime
import cv2
import numpy as np



# class ColorDetection():

    
    
#     def __init__(self): 
#         self.centroid_tracker_dict = {}

#     def processing(self, dict_params, ct, dict_tracking, img, plot_params, update_face_mask_photo_dict, aws_photo_dict, n_boxes):
                
        
#         def find_by_color(img, color_limits_1, color_limits_2 = None, min_h_w_ratio = 2):
#             # Convert image to hsv
#             hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#             ## mask 1
#             c_limit_1_ini, c_limit_1_end = color_limits_1
#             mask1 = cv2.inRange(hsv, tuple(c_limit_1_ini), tuple(c_limit_1_end))
#             if color_limits_2 is not None:
#                 ## mask 2
#                 c_limit_2_ini, c_limit_2_end = color_limits_2
#                 mask2 = cv2.inRange(hsv, tuple(c_limit_2_ini), tuple(c_limit_2_end))
#                 ## final mask and masked
#                 mask = cv2.bitwise_or(mask1, mask2)
#             else:
#                 mask = mask1
#             target = cv2.bitwise_and(img,img, mask=mask)
#             processed = target.copy()
#             target = preprocess_image(target)
#             rects,_ = find_contours(target,10)
#             blur=cv2.medianBlur(mask, 7)
#             color = np.max(blur)

#             filtered_rects = [rect for rect in rects if rect[3] / rect[2] >= min_h_w_ratio]
#             filtered_rects_centroids = [(rect[0] + rect[2] // 2, rect[1] + rect[3] // 2) for rect in filtered_rects]

#             return filtered_rects, filtered_rects_centroids, processed, color
#         def preprocess_image(image):
#             kernel = np.ones((3,3),np.uint8)
#             kernel2 = np.ones((3,3),np.uint8)
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#             thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
#             closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#             opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
#             return opening
#         def find_contours(image_proc,a):
#             h,w = image_proc.shape
#             n_imag = np.ones((h+a,w+a),np.uint8) * 255
#             n_imag[a//2:h+a//2,a//2:w+a//2] = image_proc
#             cnts, _ = cv2.findContours(n_imag.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#             rects = [cv2.boundingRect(c) for c in cnts]
#             rects = [(rect[0]-a//2,rect[1]-a//2,rect[2],rect[3]) for rect in rects]
#             return rects, cnts
#         def image_show(img, img_processed, filtered_rects):
    
#             for rect in filtered_rects:
#                 x1, y1, w, h = rect 
#                 #frame = cv2.rectangle(img, (1350, 370), (1370, 455), (0, 255, 0), 2)
#                 #frame = cv2.rectangle(img, (350, 370), (370, 455), (0, 255, 0), 2)

#             scale2=50
#             widthRes2 = int(img.shape[1]*scale2/100)
#             heigthRes2 = int(img.shape[0]*scale2/100)
#             dsize2 = (widthRes2, heigthRes2)
#             mask2 = cv2.resize(img_processed, dsize2)
#             img22 = cv2.resize(img, dsize2)
#         def ColorDetection():
#         #Main
#             #img_frac = img[370:455, 1350:1370]
#             img_frac = img[370:455, 350:370]

#             #Semaforo Rojo
#             color_limits_1 = [[0, 70, 50], [25, 255, 255]] #Mascara Color Rojo
#             color_limits_2 = [[170,70,50], [180, 255, 255]]
#             min_h_w_ratio = 1.3
#             filtered_rects, filtered_rects_centroids, img_processed, RedColor = find_by_color(img_frac, color_limits_1, color_limits_2, min_h_w_ratio)

#             #Semaforo Verde
#             color_limits_1 = [[36, 0, 0], [120, 255, 255]] #Mascara Color Verde
#             color_limits_2 = [[255,0,0], [180, 255, 255]]
#             min_h_w_ratio = 1.3
#             filtered_rects, filtered_rects_centroids, img_processed, GreenColor = find_by_color(img_frac, color_limits_1, color_limits_2, min_h_w_ratio)

#             if GreenColor==255:

#             # print('Semaforo en Verde')
#                 image_show(img, img_processed, filtered_rects)

#             elif RedColor==255:

#                 #print('Semaforo en rojo')
#                 image_show(img, img_processed, filtered_rects)
            
#             return GreenColor, RedColor
        
#         # green, red = ColorDetection()

#         # if green==255:
#         #     print('Semaforo en verde')

#             # for zone in dict_params['ZONES']:
#             #     for list_id in zone.accumulative_list_id:
#             #          object_id, class_id, direction, image, time_out, times_in, t_ini, box, speed=list_id.values()
                     
#             #          if times_in>1:


#             #             imagen = {"firts": zone.object_id[times_in]
                    
                    
#             #         }
                     
                    
#                     #  print('object_id :', object_id)
#                     #  print('times_in: ', times_in)
#                     #  print('class_id: ', class_id)
         

#             #for zone in dict_params['ZONES']:




#     # #     for idd, class_name in zone.list_id:

#     # #         print(dict_tracking[idd])
#     # #         time.sleep(100)
    

#      #               for list_id in zone.accumulative_list_id:
#      #                   object_id, class_id, direction, image, time_out, times_in, t_ini, box, speed=list_id.values()
#     #             print('object_id :', object_id)
#     #             print('times_in: ', times_in)
#     #             print('class_id: ', class_id)
#     #             print('time_out: ', time_out)
#     #             #print('box: ', box)


#     #             if times_in>1 & time_out>2 :
#      #                   print('Cruce en verde')  
#       #                  print('times_in: ', times_in)
# #                        print('class_id: ', class_id)
            
        

        
#     pass
    
#     def plot(self, plot_params, dict_plot, boxes, img):
#         pass
    
#     def reset_params(self, zone):
#         pass

#     def check_event(self, frame, zones, trackers, time_factor, verifier_model, verifier_shape, class_names, dict_params, horse_detected_in_dt, raw_image, dict_trackers, firsts_minutes, aws_photo_dict, face_mask_photo_dict, late_update_dicts, faces_sent, worker_name, faces_sent_file):

       
#         events_list = []
#         return events_list