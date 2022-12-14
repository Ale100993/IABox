from datetime import datetime
from utils import yolo_detect 
import numpy as np

class FaceMask():
    def __init__(self):
        self.centroid_tracker_dict = {'states': ['indeterminated'], 'proba': [0],
									   'average_state': 'indeterminated', 'total_proba': 50,
									   'counter': 0, 'face_mask_aws_states': ['indeterminated'], 
									   'face_mask_aws_probas': [0]}
    
    def processing(self, dict_params, ct, dict_tracking, img, plot_params, update_face_mask_photo_dict, aws_photo_dict, n_boxes):
        """
        Processing fuction of the algorithm. Apply secondary models and update params for plot and events
        :param dict_params: dictionary with params from the worker. It contains the zones (key ZONES),
        the modes (key MODES), graphic params (GRAPHIC), detection params (key DETECTION)
        and models (keys FaceMask, UsingPhone, PPEControl, FaceRecognition, FACE_MODEL)
        :param ct: centroid tracker object, to read and update params
        :param dict_tracking: dictionary with all tracking params for all tracking objects 
        :param img: raw camera image
        :param plot_params: dictionary with params to give to the plot function
        :param update_face_mask_photo_dict: list of ids to update face mask processed photo for event
        :param aws_photo_dict: dictionary to store the aws face detection photo for an id
        :param n_boxes: list of detected bounding boxes in the camera image
        """
        # Reset counters and states of face mask mode if not object in face mask zone
        for idd in [idd for idd in dict_tracking if 'box' in dict_tracking[idd]]:
            # Bool to indicate if object in face mask zone
            in_face_mask_zone = False
            for zone in dict_params['ZONES']:
                if idd in [list_id['id'] for list_id in zone.accumulative_list_id] and 'FaceMask' in zone.modes:
                    in_face_mask_zone = True
            # If not object in face mask zone, clear face mask state
            if not in_face_mask_zone:
                ct.objects[idd]['states'] = ['indeterminated']
                ct.objects[idd]['proba'] = [0]
                ct.objects[idd]['average_state'] = 'indeterminated'
                ct.objects[idd]['total_proba'] = 50
                ct.objects[idd]['counter'] = 0
        for zone in dict_params['ZONES']:
            # If face mask mode selected, perform face mask process
            if 'FaceMask' in zone.modes and not dict_params['DETECTION']['KALMAN']:
                # List of bounding boxes and ids of objects in zones
                face_mask_boxes = []
                face_mask_ids = []
                face_model = dict_params['FACE_MODEL']
                for idd, class_name in zone.list_id:
                    # If object class is face and current object
                    if class_name == 'face' and idd in dict_tracking and 'box' in dict_tracking[idd] and dict_tracking[idd]['disappeared_count'] == 0:
                        face_mask_boxes.append(dict_tracking[idd]['box'][:4])
                        face_mask_ids.append(idd)
                    # If object class is person and current object
                    elif class_name == 'person' and idd in dict_tracking and 'box' in dict_tracking[idd] and dict_tracking[idd]['disappeared_count'] == 0:
                        if face_model is None:
                            # Get box coordinates and width and height
                            x1, y1, x2, y2 = dict_tracking[idd]['box'][:4]
                            dx = x2 - x1
                            dy = y2 - y1
                            head_h = int(dy * 0.25)
                            head_w1 = int(0.2 * dx)
                            head_w2 = int(0.85 * dx)
                            # Correct if not square
                            if (head_w2 - head_w1) / head_h > 1.2:
                                head_w2 = head_h + head_w1
                            # Approximate bounding box of face
                            face_box = [head_w1 + x1, 0 + y1, head_w2 + x1, head_h + y1]
                            face_mask_boxes.append(face_box)
                            face_mask_ids.append(idd)
                        else:
                            x1, y1, x2, y2 = dict_tracking[idd]['box'][:4]
                            height_im, width_im = img.shape[:2]
                            h_person, w_person = y2 - y1, x2 - x1
                            a, b = 0.20, 0.20

                            nx1 = int(x1 - w_person * a) if int(x1 - w_person * a) > 0 else 0
                            ny1 = int(y1 - h_person * b) if int(y1 - h_person * b) > 0 else 0
                            nx2 = int(x2 + w_person * a) if int(x2 + w_person * a) < width_im else width_im
                            ny2 = int(y2 + h_person * b) if int(y2 + h_person * b) < height_im else height_im
                                
                            img_person = img[ny1:ny2, nx1:nx2].copy()
                            h_person_n, w_person_n = ny2 - ny1, nx2 - nx1
                            face_boxes = yolo_detect(img_person, face_model, dict_params['DETECTION']['BUSY_ENABLED'])
                            if face_boxes:
                                face_box = face_boxes[0]
                                cx = int(face_box[0] * w_person_n + nx1)
                                cy = int(face_box[1] * h_person_n + ny1)
                                wi = int(face_box[2] * w_person_n)
                                hi = int(face_box[3] * h_person_n)
                                x1i = cx - wi//2
                                x2i = cx + wi//2
                                y1i = cy - hi//2
                                y2i = cy + hi//2

                                wf, hf = x2i - x1i, y2i - y1i
                                a, b = 0.35, 0.25

                                nx1i = int(x1i - wf * a) if int(x1i - wf * a) > 0 else 0
                                ny1i = int(y1i - hf * b) if int(y1i - hf * b) > 0 else 0
                                nx2i = int(x2i + wf * a) if int(x2i + wf * a) < width_im else width_im
                                ny2i = int(y2i + hf * b) if int(y2i + hf * b) < height_im else height_im
                                
                                detected_face_area = (nx2i - nx1i) * (ny2i - ny1i)
                                if detected_face_area >= dict_params['DETECTION']['FACE_MASK_MIN_AREA']:
                                    face_mask_boxes.append([nx1i, ny1i, nx2i, ny2i])
                                    face_mask_ids.append(idd)
                # Process face mask of faces
                list_states, list_proba = process_mask(img,face_mask_boxes,dict_params['FaceMask'])
                # Update object dicts
                for i, idd in enumerate(face_mask_ids):
                    ct.objects[idd]['states'].append(list_states[i])
                    ct.objects[idd]['proba'].append(list_proba[i])
                    ct.objects[idd]['counter'] += 1
                    if ct.objects[idd]['counter'] >= zone.max_counter:
                        ct.objects[idd]['counter'] = 0
                        update_average_state(ct, idd)
                        update_face_mask_photo_dict.append(idd)
                        dict_tracking[idd]['total_proba'] = ct.objects[idd]['total_proba']
                        dict_tracking[idd]['average_state'] = ct.objects[idd]['average_state']
                        dict_tracking[idd]['last_proba'] = ct.objects[idd]['proba'][-1]
                        dict_tracking[idd]['last_type'] = ct.objects[idd]['states'][-1]
        # Plot params
        plot_params['states'] = [dict_tracking[idd]['average_state'] for idd in dict_tracking if 'box' in dict_tracking[idd]]
        plot_params['proba'] = [dict_tracking[idd]['total_proba'] for idd in dict_tracking if 'box' in dict_tracking[idd]]
        # Get list of last states and probabilities of each object (for debugging purposes)
        plot_params['last_state'] = [dict_tracking[idd].get('last_type', 'indeterminated') for idd in dict_tracking]
        plot_params['last_proba'] = [dict_tracking[idd].get('last_proba', 0) for idd in dict_tracking]

    def plot(self, plot_params, dict_plot, boxes, img):
        """
        Update plot labels dictionary and update color box and zones.
        :param plot_params: dictionary with the params to make the plot
        :param dict_plot: dictionary of labels and colors of boxes
        :param boxes: list of boxes (ordered by id)
        :param img: image to plot
        """
        for i in range(len(boxes)):
            # Select color depending of the state (with/without mask)
            if plot_params['states'][i] == 'without_mask':
                rgb = (0,0,255)
            elif plot_params['states'][i] == 'with_mask':
                rgb = (0,255,0)
            else:
                rgb = (0,255,0)
            dict_plot[i]['rectangle'].update({'color': rgb})
            # Labels
            if plot_params['PLOT_STATE']:
                dict_plot[i]['labels'].update({'state': {'text': plot_params['states'][i]}})
            if plot_params['PLOT_PROBABILITY']:
                dict_plot[i]['labels'].update({'proba': {'text': str(plot_params['proba'][i])}})
            if plot_params['PLOT_LAST_STATE'] and 'last_state' in plot_params:
                dict_plot[i]['labels'].update({'last_state': {'text': plot_params['last_state'][i]}})
            if plot_params['PLOT_LAST_PROBA'] and 'last_proba' in plot_params:
                dict_plot[i]['labels'].update({'last_proba': {'text': str(plot_params['last_proba'][i])}})
    
    def reset_params(self, zone):
        """
        Reset zone params when worker or zone goes from disabled to enabled
        :param zone: zone to reset the algorithms params
        """
        pass

    def check_event(self, frame, zones, trackers, time_factor, verifier_model, verifier_shape, class_names, dict_params, horse_detected_in_dt, raw_image, dict_trackers, firsts_minutes, aws_photo_dict, face_mask_photo_dict, late_update_dicts, faces_sent, worker_name, faces_sent_file):
        """
        Check if an event of the algorithm occurs
        :param frame: processed frame to send to an event or save
        :param zones: zones with params and internal variables
        :param trackers: dictionary of tracked objects with params 
        :param time_factor: inverse of mean fps to transform iterations to real time values
        :param verifier_model: model to verify detections
        :param verifier_shape: input image shape of verifier model
        :param class_names: list of class names of the primary model
        :param dict_params: dictionary of params from workers
        :param horse_detected_in_dt: bool to indicate that a horse was detected in a period of time (4 Hojas)
        :param raw_image: raw image of the camera to save
        :param dict_trackers: similar to trackers but for current tracked objects
        :param firsts_minutes: bool to indicate that it is the firsts minutes of processing (app slower)
        :param aws_photo_dict: dictionary of recognized or unrecognized faces
        :param face_mask_photo_dict: dictionary with the processed frame of each object tracked at the time of no face mask detected
        :param late_update_dicts: dictionary of aws face recognition results of people no more in frame
        :param faces_sent: list of WL people events sent (to avoid sending again in the same day)
        :param worker_name: name of the worker
        :param faces_sent_file: file to save and load on reboot the list of faces sent 
        """
        events_list = []
        for zone in zones:
            if 'FaceMask' in zone.modes and zone.zone_enabled:
                for list_id in zone.accumulative_list_id + late_update_dicts:
                    object_id, class_id, direction, image, _, _, _, _, _ = list_id.values()
                    if (class_id == 'person' or class_id == 'face') and object_id in trackers \
                            and object_id in dict_trackers and direction in zone.filtered_directions:
                        image2db = None
                        state = dict_trackers[object_id]['average_state']
                        now = datetime.now()
                        is_person_image = False
                        # If first time detected new track id, else same
                        flag_tracker = False if not object_id in zone.face_mask_event_sent else True
                        # If current state != indeterminated and first time detected or change of state 
                        if state != "indeterminated" and \
                                                    (not object_id in zone.face_mask_event_sent or \
                                                    (object_id in zone.face_mask_event_sent and zone.face_mask_event_sent[object_id] != state)):
                            # If state = 'without_mask', save frame and send alert
                            if state == "without_mask":
                                im = face_mask_photo_dict.get(object_id, None)
                                if im is not None:
                                    frame = im                            
                                image2db = frame.copy()

                            events_list.append({'event_type': state,
                                                'identity': object_id,
                                                'params': {'t_ini': now,
                                                            't_end': now,
                                                            'image2db': image2db,
                                                            'is_person_image': is_person_image,
                                                            'flag_tracker': flag_tracker,
                                                            'a': state,
                                                            'object_id': object_id,
                                                            'zone_id': zone.id,
                                                            'raw_image': image.copy() if image is not None else None}})                                        
                        # Add object id to face mask event sent
                        zone.face_mask_event_sent[object_id] = state 
            # Clean late_update_dicts to prevent repeated events in different zones
            late_update_dicts = []
        return events_list

def process_mask(original_img,boxes,mask_params):
    """
    Face mask detection (CNN model). Get list of detection (with/without mask),
    probabilities and the corresponding boxes
    :param original_img: original frame or image without processing
    :param boxes: boxes (faces) detected by yolo model
    :param mask_params: dictionary with the CNN model
    """
    # Initilize some lists
    imgs_face = []
    list_states, list_proba = [], []
    # For each box
    for box in boxes:
        # Get coordinates of boxes and crop original image (get only the image of the face)
        x1, y1, x2, y2 = box[:4]
        imgs_face.append(original_img[y1:y2, x1:x2])
    # If faces detected by yolo
    if len(imgs_face) > 0:
        # Process and get result of CNN
        while mask_params['MODEL'].busy:
            pass
        list_states, list_proba = mask_params['MODEL'].detect(imgs_face)
    # Return list of states, probabilities, boxes as numpy array and list of classes (0)
    return list_states, list_proba

def update_average_state(ct,objectID):
    """
    Update the face mask average state considering the last states
    """
    # List of last states and probabilities
    n = ct.numLastStates + 1
    last_states = ct.objects[objectID]['states'][-n:-1]
    last_probas = ct.objects[objectID]['proba'][-n:-1]
    #Get the most repeated state in the last n-1 states
    try:
        average_state = max(set(last_states), key=last_states.count)
    except:
        average_state = ct.objects[objectID]['states'][-1]
    
    # Set color considering the most repeated state
    ct.objects[objectID]['average_state'] = average_state

    # Get the indexes of the most repeated state
    ind_average_state = [i for i, j in enumerate(last_states) if j == average_state]
    s, count = 0, 0
    # Calculate the average probability of the last average states
    for i in ind_average_state:
        s += last_probas[i]
        count += 1
    try:
        ct.objects[objectID]['total_proba'] = int(np.round((s/count) * 100))
    except:
        ct.objects[objectID]['total_proba'] = int(ct.objects[objectID]['proba'][-1])