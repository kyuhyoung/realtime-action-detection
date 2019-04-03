'''
from keras import backend as K
K.set_image_dim_ordering('th')

from scipy.io import loadmat, savemat
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
import h5py
from keras.models import model_from_json

from theano import function, config, shared, sandbox
import theano.tensor as T
from threading import Lock

'''

import torch.backends.cudnn as cudnn
import torch
from torch.autograd import Variable
from ssd import build_ssd
from layers.box_utils import decode, nms
from data import base_transform

import datetime
import cv2
import os
import numpy as np
import time



Y_OFFSET_GT_BOX = 15
Y_OFFSET_DET_SCORE = 14

FONT_SCALE_DET_BOX = 0.5 
FONT_SCALE_CONFIDIENCE = 0.7 

class DataClass(object):

    def __init__(self):
        self.end_of_capture = False
        #self.li_det = []
        self.net_result = None
        self.im_bgr = None
        self.batch_rgb = None
        #self.im_bgr = 0
        #self.im_bgr_copy = None
        #self.im_rgb_copy = 0
        self.fps_det = None
        self.fps_fetch = None
        #self.prob_ano = -1
        '''        
        self.lock_rgb = Lock()
        #self.lock_bbox = Lock()
        self.lock_li_rgb = Lock()
        self.lock_li_det = Lock()
        self.lock_fps_det = Lock()
        self.lock_fps_fetch = Lock()
        '''

    def get_eoc(self):
        return self.end_of_capture

    def set_eoc(self, is_eoc):
        self.end_of_capture = is_eoc

    def set_net_result(self, net_result):
        self.net_result = net_result

    def get_net_result(self):
        return self.net_result




    def set_bgr(self, im_bgr):
        self.im_bgr = im_bgr

    def get_bgr(self):
        return self.im_bgr

    def set_batch_rgb(self, batch_rgb):
        self.batch_rgb = batch_rgb

    def get_batch_rgb(self):
        return self.batch_rgb



    #def set_li_rgb(self, li_im_rgb):
    def set_li_rgb(self, mat_im_rgb):
        #with self.lock_li_rgb:
        #    self.mat_im_rgb = mati_im_rgb
            #self.li_im_rgb = li_im_rgb
            #print('self.im_rgb is set by : ', str_from)
        self.mat_im_rgb = mat_im_rgb
    def get_li_rgb(self):
        #with self.lock_li_rgb:
        #    return self.mat_im_rgb
            #return self.li_im_rgb
        return self.mat_im_rgb

    def set_li_det(self, li_det):
        #with self.lock_li_det:
        #    self.li_det = li_det
        self.li_det = li_det
    def get_li_det(self):
        #with self.lock_li_det:
        #    return self.li_det
        return self.li_det
    '''
    def set_anomaly_prob(self, prob_ano):
        #with self.lock_fps_det:
        #    self.fps_det = fps_det
        self.prob_ano = prob_ano

    def get_anomaly_prob(self):
        #with self.lock_fps_det:
        #    return self.fps_det
        return self.prob_ano
    '''



    def set_fps_det(self, fps_det):
        #with self.lock_fps_det:
        #    self.fps_det = fps_det
        self.fps_det = fps_det
    def get_fps_det(self):
        #with self.lock_fps_det:
        #    return self.fps_det
        return self.fps_det

    def set_fps_fetch(self, fps_fetch):
        #with self.lock_fps_fetch:
        #    self.fps_fetch = fps_fetch
        self.fps_fetch = fps_fetch

    def get_fps_fetch(self):
        #with self.lock_fps_fetch:
        #    return self.fps_fetch
        return self.fps_fetch




class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
                                             
    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self
                                                  
    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def _elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        torch.cuda.synchronize()
        return (datetime.datetime.now() - self._start).total_seconds()
    
    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self._elapsed()



class Detection:
    def __init__(self, x, y, w, h, class_id, label, confidence):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.class_id = class_id
        self.label = label
        self.confidence = confidence


def create_c3d(path_json, summary = False):
    """ Return the Keras model of the network
    """
    model = Sequential()
                
    # 1st layer group
                        
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv1', subsample=(1, 1, 1), input_shape=(3, 16, 112, 112)))
    #model.add(Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv1', subsample=(1, 1, 1), input_shape=(3, 16, 320, 240)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid', name='pool1'))
                     
    # 2nd layer group
    
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same', name='conv2', subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool2'))
    
    # 3rd layer group
    
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3a', subsample=(1, 1, 1)))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3b', subsample=(1, 1, 1)))

    
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool3'))
                                                                                     
                                                                                     
    # 4th layer group
    
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4a', subsample=(1, 1, 1)))

    model.add(Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4b', subsample=(1, 1, 1)))
                                                                                                 
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool4'))
                                                                                                      
    # 5th layer group
    
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv5a', subsample=(1, 1, 1)))
                                                                                                
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv5b', subsample=(1, 1, 1)))
                                                                                                 
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
     
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
     
    border_mode='valid', name='pool5'))
     
    model.add(Flatten())
     
    # FC layers group
                          
    model.add(Dense(4096, activation='relu', name='fc6'))

    #'''
    model.add(Dropout(.5))
                  
    model.add(Dense(4096, activation='relu', name='fc7'))
                          
    model.add(Dropout(.5))
    
    model.add(Dense(487, activation='softmax', name='fc8'))
    #'''

    if summary:
                          
        print('model.summary() in created_c3d : ', model.summary())
                     
    if path_json:
        json_string = model.to_json()
        with open(path_json, 'w') as f:
            f.write(json_string)
            print('The model structure has been save at :', path_json)
    return model


def rescale_with_margin_ratio(boxes, li_margin_ratio_l_r_t_b):
    
    margin_ratio_l, margin_ratio_r, margin_ratio_t, margin_ratio_b = li_margin_ratio_l_r_t_b 
    if 0 == margin_ratio_t:
        bunmo = 1.0 - (margin_ratio_l + margin_ratio_r)
        boxes[:, 0] -= margin_ratio_l
        boxes[:, 0] /= bunmo
        boxes[:, 2] -= margin_ratio_l
        boxes[:, 2] /= bunmo
    elif 0 == margin_ratio_l:
        bunmo = 1.0 - (margin_ratio_t + margin_ratio_b)
        boxes[:, 1] -= margin_ratio_t
        boxes[:, 1] /= bunmo
        boxes[:, 3] -= margin_ratio_t
        boxes[:, 3] /= bunmo
    return boxes 



def mark_detections(im_bgr, conf_scores, tu_str_class, decoded_boxes, w_h_ori, li_margin_ratio_l_r_t_b, li_color_class, top_k, th_conf, th_nms, det_boxes):
    
    num_classes = len(tu_str_class) + 1
    w_ori, h_ori = w_h_ori
    for cl_ind in range(1, num_classes):
        #str_class = dataset.CLASSES[cl_ind - 1]
        str_class = tu_str_class[cl_ind - 1]
        #print('str_class : ', str_class)
        scores = conf_scores[:, cl_ind].squeeze()
        #c_mask = scores.gt(args.conf_thresh)  # greater than minmum threshold
        c_mask = scores.gt(th_conf)  # greater than minmum threshold
        scores = scores[c_mask]
        #print('scores.nelement() : ', scores.nelement())
        if scores.nelement() == 0:
            #print(len(''), ' dim ==0 ')
            if det_boxes is not None:
                det_boxes[cl_ind - 1].append(np.asarray([]))
            continue
        boxes = decoded_boxes.clone()
        #print('boxes.shape ori : ', boxes.shape)
        l_mask = c_mask.unsqueeze(1).expand_as(boxes)
        boxes = boxes[l_mask].view(-1, 4)
        # idx of highest scoring and non-overlapping boxes per class
        #ids, counts = nms(boxes, scores, args.nms_thresh, args.topk)  # idsn - ids after nms
        ids, counts = nms(boxes, scores, th_nms, top_k)  # idsn - ids after nms
        #print('counts : ', counts)
        scores = scores[ids[:counts]].cpu().numpy()
        boxes = boxes[ids[:counts]].cpu().numpy()
        boxes = rescale_with_margin_ratio(boxes, li_margin_ratio_l_r_t_b)
        # print('boxes sahpe',boxes.shape)
        boxes[:, 0] *= w_ori
        boxes[:, 2] *= w_ori
        boxes[:, 1] *= h_ori
        boxes[:, 3] *= h_ori
        rand_0_1 = np.random.rand(boxes.shape[0])
        #print('boxes.shape after : ', boxes.shape)
        for ik in range(boxes.shape[0]):
            #print('ids[ik] : ', ids[ik].cpu().numpy())
            boxes[ik, 0] = max(0, boxes[ik, 0])
            boxes[ik, 2] = min(w_ori, boxes[ik, 2])
            boxes[ik, 1] = max(0, boxes[ik, 1])
            boxes[ik, 3] = min(h_ori, boxes[ik, 3])
            #print('boxes[ik] : ', boxes[ik])
            cv2.rectangle(im_bgr, (boxes[ik, 0], boxes[ik, 1]), (boxes[ik, 2], boxes[ik, 3]), li_color_class[cl_ind - 1], 1)
            hei = boxes[ik, 3] - boxes[ik, 1]
            y_offset = Y_OFFSET_GT_BOX + (hei - Y_OFFSET_GT_BOX) * rand_0_1[ik] 
            #cv2.putText(t3_bgr, str_class, (int(boxes[ik, 0]), int(boxes[ik, 1] + y_offset)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_class[cl_ind - 1])
            cv2.putText(im_bgr, str_class, (int(boxes[ik, 0]), int(boxes[ik, 1] + y_offset)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_DET_BOX, li_color_class[cl_ind - 1])
            str_score = "{:.2f}".format(scores[ik]);    
            #print('scores[ik] : ', scores[ik]); print('str_score : ', str_score);   exit() 
            cv2.putText(im_bgr, str_score, (int(boxes[ik, 0]), int(boxes[ik, 1] + y_offset + Y_OFFSET_DET_SCORE)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_DET_BOX, li_color_class[cl_ind - 1])
            #cv2.imshow('t3_bgr', t3_bgr); cv2.waitKey();    #exit()

        if det_boxes is not None:
            #print("det_boxes is not none"); exit() 
            cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
            det_boxes[cl_ind - 1].append(cls_dets)

    return im_bgr, det_boxes




#def display_in_thread(class_data_proxy, COLORS):
def display_in_thread(class_data_proxy, shall_record, w_h_cam, w_h_net, tu_str_class, li_color_class, top_k, th_conf, th_nms):


    if shall_record:
        fn_record = 'action_recognition_cam_conf_thres_{:.2f}_nms_thres_{:.1f}.avi'.format(th_conf, th_nms)
        writer = make_video_recorder(fn_record, w_h_cam, 20)


    li_margin_ratio_l_r_t_b = compute_margin_ratio_l_r_t_b(w_h_cam, w_h_net)
    
    wid, hei = w_h_cam

    fps_disp = FPS().start()
    is_huda = False
    while not class_data_proxy.get_eoc():
        
        im_bgr = class_data_proxy.get_bgr()
    
        if im_bgr is None:
            if is_huda:
                class_data_proxy.set_eoc(True)
                #print('im_rgb of display is NOT None')
            else:
                #print('First frame of display thread has not been arrived')
                continue
        is_huda = True
        #time.sleep(0.5);        continue
        #hei, wid = im_bgr.shape[:2]
        #print('fps_disp._numFrames : ', fps_disp._numFrames)
        #print('hei : ', hei)
        #print('wid : ', wid)
        #im_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
       
        net_result = class_data_proxy.get_net_result()
        if net_result:
            decoded_boxes, conf_scores = net_result

            im_bgr, _ = mark_detections(im_bgr, conf_scores, tu_str_class, decoded_boxes, w_h_cam, li_margin_ratio_l_r_t_b, li_color_class, top_k, th_conf, th_nms, None)

        '''
        prob_ano_copy = class_data_proxy.get_anomaly_prob()
        text = "anomaly probability : {:.2f}".format(prob_ano_copy)
        cv2.putText(im_bgr, text, (int(wid * 0.25), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        '''
        '''
        li_det = class_data_proxy.get_li_det()
        for idx, det in enumerate(li_det):
            #print('idx : ', idx)
            x, y, w, h = det.x, det.y, det.w, det.h
            color = [int(c) for c in COLORS[det.class_id]]
            cv2.rectangle(im_bgr, (x, y), (x + w, y + h), color, 1)
            text = "{}: {:.4f}".format(det.label, det.confidence)
            cv2.putText(im_bgr, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        #print('AAA det')
        '''

        cv2.putText(im_bgr, "conf. thres. : {:.2f}".format(th_conf), (int(wid * 0.5 - 85), int(hei * 0.07)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_CONFIDIENCE, (0, 255, 0))

        fps_det = class_data_proxy.get_fps_det()
        if fps_det:
            text = "fps det : {:.1f}".format(fps_det)
            #print("fps det in display thread : {:.1f}".format(fps_det))
            cv2.putText(im_bgr, text, (int(wid * 0.35), hei - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
        
        fps_fetch = class_data_proxy.get_fps_fetch()
        if fps_fetch is not None:
            text = "fps fetch : {:.1f}".format(fps_fetch)
            #print("fps fetch in display thread : {:.1f}".format(fps_fetch))
            cv2.putText(im_bgr, text, (int(wid * 0.35), hei - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

        fps_disp.update();
        text = "fps disp : {:.1f}".format(fps_disp.fps())
        cv2.putText(im_bgr, text, (int(wid * 0.35), hei - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        cv2.imshow('im_bgr', im_bgr)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # esc key
            cv2.destroyAllWindows()
            class_data_proxy.set_eoc(True)
        #elif k = ord('s'): # 's' key
            #cv2.imwrite('lenagray.png',img)
            #cv2.destroyAllWindow()
        #print('fps_display : ', fps_disp.fps())
    print("class_data.end_of_capture is True : display_in_thread") 
    #return class_data_proxy



def rot90(W):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            for k in range(W.shape[2]):
                W[i, j, k] = np.rot90(W[i, j, k], 2)
    return W

'''
import caffe_pb2 as caffe
def create_and_load_c3d(path_weight_c3d, json_c3d, h5_c3d):


    model_c3d = create_c3d(True)

    p = caffe.NetParameter()
    p.ParseFromString(
        #open('model/conv3d_deepnetA_sport1m_iter_1900000', 'rb').read()
        open(path_weight_c3d, 'rb').read()
    )


    params = []
    conv_layers_indx = [1, 4, 7, 9, 12, 14, 17, 19]
    fc_layers_indx = [22, 25, 28]

    for i in conv_layers_indx:
        layer = p.layers[i]
        weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
        weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
            layer.blobs[0].num, layer.blobs[0].channels, layer.blobs[0].length,
            layer.blobs[0].height, layer.blobs[0].width
        )
        weights_p = rot90(weights_p)
        params.append([weights_p, weights_b])
    for i in fc_layers_indx:
        layer = p.layers[i]
        weights_b = np.array(layer.blobs[1].data, dtype=np.float32)
        weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(
            layer.blobs[0].num, layer.blobs[0].channels, layer.blobs[0].length,
            layer.blobs[0].height, layer.blobs[0].width)[0,0,0,:,:].T
                                                                        
        params.append([weights_p, weights_b])

    model_layers_indx = [0, 2, 4, 5, 7, 8, 10, 11] + [15, 17, 19] #conv + fc
    for i, j in zip(model_layers_indx, range(11)):
        model.layers[i].set_weights(params[j])



    model.save_weights('sports1M_weights.h5', overwrite=True)
    json_string = model.to_json()
    with open('sports1M_model.json', 'w') as f:
        f.write(json_string)

    return model
'''

def load_weights(model, weight_path):
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

def conv_dict(dict2): # Helper function to save the model
    i = 0
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict




def load_model_from_json(json_path):
    #model = model_from_json(open(json_path).read())
    model = model_from_json(open(json_path, 'r').read())
    return model




def load_c3d_from_h5(json_c3d, h5_c3d):
    #model = model_from_json(open('sports1M_model.json', 'r').read())
    print('json_c3d : ', json_c3d)
    #model = model_from_json(open(json_c3d, 'r').read())
    model = load_model_from_json(json_c3d)
    print('type(model) b4 : ', type(model))
    #model.load_weights('sports1M_weights.h5')
    model.load_weights(h5_c3d)
    print('type(model) after : ', type(model))
    #model.compile(loss='mean_squared_error', optimizer='sgd')
    print('model.summary() started')
    model.summary()
    print('model.summary finished')
    return model    

'''
def post_process_output(im_bgr, net, tu_str_class, loc_data, conf_preds, prior_data, w_h_cam, li_margin_ratio_l_r_t_b, li_color_class, th_conf, th_nms, det_boxes):
    decoded_boxes = decode(loc_data[0].data, prior_data.data, cfg['variance']).clone()
    conf_scores = net.softmax(conf_preds[0]).data.clone()  
    #im_bgr, det_boxes = mark_detections(im_bgr, conf_scores, dataset.CLASSES, decoded_boxes, w_h_cam, li_color_class, detboxes)
    im_bgr, det_boxes = mark_detections(im_bgr, conf_scores, tu_str_class, decoded_boxes, w_h_cam, li_margin_ratio_l_r_t_b, li_color_class, th_conf, th_nms, det_boxes)
    return im_bgr, det_boxes
'''


def compute_margin_ratio_l_r_t_b(w_h_ori, w_h_net):
    w_ori, h_ori = w_h_ori; w_net, h_net = w_h_net;
    ratio_l, ratio_r, ratio_t, ratio_b = 0, 0, 0, 0
    ratio_w = w_net / w_ori
    ratio_h = h_net / h_ori
    if ratio_w > ratio_h:
        ratio = ratio_h
        w_resized = w_ori * ratio
        ratio_l = (w_net - w_resized) / (w_net * 2.0)
        ratio_r = ratio_l
    else:
        ratio = ratio_w
        h_resized = h_ori * ratio
        ratio_t = (h_net - h_resized) / (h_net * 2.0)
        ratio_b = ratio_t
    return [ratio_l, ratio_r, ratio_t, ratio_b]

def init_ssd(num_classes, trained_model_path, use_cuda):
    net = build_ssd(300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(trained_model_path))
    net.eval()
    #if args.cuda:
    if use_cuda:
        net = net.cuda()
        cudnn.benchmark = True
    return net



def detect_in_thread(class_data_proxy, num_classes, trained_model_path, use_cuda, cfg):

    #li_margin_ratio_l_r_t_b = compute_margin_ratio_l_r_t_b(w_h_cam, w_h_net)
    net = init_ssd(num_classes, trained_model_path, use_cuda)

    fps_det = FPS().start()
    print('class_data.end_of_capture of detect in thread : ', class_data_proxy.get_eoc())#; exit()
    
    is_huda = False
    while not class_data_proxy.get_eoc():
        batch_rgb = class_data_proxy.get_batch_rgb()
        #print('batch_rgb.shape : ', batch_rgb.shape)
        if batch_rgb is None:
            print('batch_rgb is None !!!'); #exit()
            if is_huda:
                class_data_proxy.set_eoc()
                print('class_data.end_of_capture of detect in thread is True'); #exit()
            continue
        is_huda = True
        start = time.time()

        #   net forwarding
        loc_data, conf_preds, prior_data = net(batch_rgb)

        decoded_boxes = decode(loc_data[0].data, prior_data.data, cfg['variance']).clone()
        conf_scores = net.softmax(conf_preds[0]).data.clone()  

        class_data_proxy.set_net_result((decoded_boxes, conf_scores))
        #   post process output
        #li_det = post_process_output(im_bgr, net, CLASSES, loc_data, conf_preds, prior_data, w_h_cam, li_margin_ratio_l_r_t_b, li_color_class, th_conf, th_nms, None)
 
        #class_data_proxy.set_li_det(li_det)

        fps_det.update();   class_data_proxy.set_fps_det(fps_det.fps())
        #print('fps_det : ', fps_det.fps())
    print("class_data.end_of_capture is True : detect_in_thread") 
    #return class_data_proxy



def resize_and_fill(im_rgb, color_rgb, w_h_desired):
    w_desired, h_desired = w_h_desired
    h_old, w_old = im_rgb.shape[:2]
    #print('h_old : ', h_old);   print('w_old : ', w_old);   #exit()
    ratio_w = float(w_desired) / float(w_old)
    ratio_h = float(h_desired) / float(h_old)
    ratio = min(ratio_w, ratio_h)
    w_new = int(w_old * ratio)
    h_new = int(h_old * ratio)
    #print('w_new : ', w_new);   print('w_new : ', w_new);   #exit()
    im_rgb = cv2.resize(im_rgb, (w_new, h_new))
    if ratio_w != ratio_h:
        delta_w = w_desired - w_new
        delta_h = h_desired - h_new
        top = delta_h // 2; bottom = delta_h - top
        left = delta_w // 2;    right = delta_w - left
        im_rgb = cv2.copyMakeBorder(im_rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color_rgb)
    return im_rgb
    



def im2batch(im_bgr, means_bgr, w_h_net, use_cuda):
   
    '''
    im_bgr_resized = resize_and_fill(im_bgr, means_bgr, w_h_net)
    im_bgr_norm_resized = im_bgr_resized - means_bgr
    '''
    im_bgr_norm_resized = base_transform(im_bgr, w_h_net[0], means_bgr)
    im_rgb_norm_resized = cv2.cvtColor(im_bgr_norm_resized, cv2.COLOR_BGR2RGB)

    #im_rgb_norm_resized = im_rgb_norm_resized.transpose((2, 0, 1))   
    #ts_rgb_norm = torch.from_numpy(im_rgb_norm_resized).float()

    ts_rgb_norm = torch.from_numpy(im_rgb_norm_resized.transpose(2, 0, 1)).float()
    
    batch_rgb = ts_rgb_norm.unsqueeze(0)
    #if args.cuda:
    if use_cuda:
        batch_rgb = Variable(batch_rgb.cuda(), volatile=True)
    return batch_rgb


def is_video_file(fn): 
    ext = (".3g2", ".3gp", ".asf", ".asx", ".avi", ".flv", ".m2ts", ".mkv", ".mov", ".mp4", ".mpg", ".mpeg", ".rm", ".swf", ".vob", ".wmv")
    return fn.endswith(ext)
 

def init_cam(id_cam):
    if is_video_file(id_cam):
        #print('this is video file')
        kam = cv2.VideoCapture(id_cam)
    
    else:
        '''
        print('this is camera ID')
        print('id_cam b4 : ', id_cam)
        id_cam = int(id_cam)
        print('id_cam after : ', id_cam)
        '''
        kam = cv2.VideoCapture(int(id_cam))
    if kam is None or not kam.isOpened():
        print('Unable to open camera ID : ', id_cam);   exit()
    print('Camera : {} is opened'.format(id_cam))
    w_h_cam = (int(kam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(kam.get(cv2.CAP_PROP_FRAME_HEIGHT))) # float
    return kam, w_h_cam    




def fetch_in_thread(class_data_proxy, kam, means_bgr, w_h_net, use_cuda):
    
    #li_rgb_resized = []
    x_from, y_from, x_to, y_to = -1, -1, -1, -1
    w_h_resized = None
    is_huda = False
    #print("id_cam : ", id_cam)
    #kam, w_h_cam = init_cam(id_cam)
    #kapture = cv2.VideoCapture(fn_video_or_cam)
    #kapture = cv2.VideoCapture(1)
    class_data_proxy.set_eoc(not kam.isOpened())
    #if is_video:
    #    n_frame_total = int(kapture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_fetch = FPS().start()
    while not class_data_proxy.get_eoc():
        ret, im_bgr = kam.read()
        if ret:
            #print('im_bgr is retrived in fetch thread');
            is_huda = True
            #cv2.imshow("temp", im_bgr); cv2.waitKey(10000)
            #im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

            #im_bgr = cv2.imread('./ucf24/rgb-images/LongJump/v_LongJump_g06_c01/00037.jpg')
            #im_bgr = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            class_data_proxy.set_bgr(im_bgr)
            batch_rgb = im2batch(im_bgr, means_bgr, w_h_net, use_cuda)
            class_data_proxy.set_batch_rgb(batch_rgb)
            '''
            #print('w_h_net : ', w_h_net)
            #print('im_rgb.shape : ', im_rgb.shape)
            if w_h_resized is None:
                h_cam, w_cam = im_rgb.shape[:2]
                w_net, h_net = w_h_net
                w_ratio = w_net / w_cam;    h_ratio = h_net / h_cam;
                if h_ratio > w_ratio:
                    w_resized = int(w_cam * h_ratio)
                    w_margin = int(0.5 * (w_resized - w_net))
                    w_h_resized = (w_resized, h_net)
                    x_from = w_margin;  x_to = x_from + w_net
                    y_from = 0;  y_to = h_net

                else:
                    h_resized = int(h_cam * w_ratio)
                    h_margin = int(0.5 * (h_resized - h_net))
                    w_h_resized = (w_net, h_resized)
                    x_from = 0; x_to = w_net
                    y_from = h_margin;  y_to = y_from + h_net
            #im_rgb = cv2.resize(im_rgb, w_h_net)
            li_rgb_resized.append(cv2.resize(im_rgb, w_h_resized))
            #print('li_rgb_resize appended')
            if len(li_rgb_resized) >= len_li_rgb:
                #print("li_rgb is charged !!!"); exit()
                #class_data_proxy.set_li_rgb(li_rgb_resized)
                #class_data_proxy.set_li_rgb(np.array(li_rgb_resized, dtype=np.float32))
                vid = np.array(li_rgb_resized, dtype=np.float32)
                #print('vid.shape :', vid.shape); exit() # shape : num_frame - height - width - channel
                class_data_proxy.set_li_rgb(vid[:, y_from:y_to, x_from:x_to, :].transpose((3, 0, 1, 2))) #  shape : channel - num_frame - height - width
                del li_rgb_resized[:]
            '''
        else:
            if is_huda:
                #class_data_proxy.set_eoc(True) 
                print("frame drop happend !!!")
                #del li_rgb_resized[:]
        #if is_video:
        #    idx_frame = int(kapture.get(cv2.CAP_PROP_POS_FRAMES))
        #    if idx_frame >= n_frame_total - 1:
        #        class_data_proxy.end_of_capture = True

        #print('fps_fetch._numFrames : ', fps_fetch._numFrames)
        fps_fetch.update();   class_data_proxy.set_fps_fetch(fps_fetch.fps())
        #print('fps_fetch : ', fps_fetch.fps())
   
        #time.sleep(0.5)

    print("class_data.end_of_capture is True : fetch_in_thread") 
    #return class_data_proxy


def init_detection():
    '''  
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000
  
    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], T.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
    else:
        print('Used the gpu')
    exit()
    '''
    #w_h_net = (320, 240)
    w_h_net = (112, 112)
    fn_video = 0
    
    dir_data = "data"
    json_c3d = os.path.sep.join([dir_data, "c3d_sports_1M_new_theano.json"])
    h5_c3d = os.path.sep.join([dir_data, "c3d_sports_1M_weights.h5"])
    '''    
    json_fc = os.path.sep.join([dir_data, "fc_model.json"])
    weight_mat_fc = os.path.sep.join([dir_data, "fc_weights_L1L2.mat"])
    '''
    json_fc = "model.json"; weight_mat_fc = "weights_L1L2.mat";
    
    th_confidence = 0.5
    th_nms_iou = 0.3

    # load the COCO class labels our YOLO model was trained on
    #labelsPath = os.path.sep.join([dir_yolo, "coco.names"])
    #LABELS = open(labelsPath).read().strip().split("\n")
 
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    #COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    #return fn_video, dir_yolo, th_confidence, th_nms_iou, COLORS, LABELS
    return fn_video, w_h_net, json_fc, weight_mat_fc, json_c3d, h5_c3d


