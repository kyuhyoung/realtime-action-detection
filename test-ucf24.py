"""
    Copyright (c) 2017, Gurkirt Singh

    This code and is available
    under the terms of MIT License provided in LICENSE.
    Please retain this notice and LICENSE if you use
    this file (or any portion of it) in your project.
    ---------------------------------------------------------
"""
#'''
import random
#import matplotlib
#matplotlib.use('Agg')
#'''
import cv2
import torch
#import matplotlib.pyplot as plt
from torch.autograd import Variable
from data import AnnotationTransform, UCF24Detection, BaseTransform, CLASSES, detection_collate, v2
import torch.utils.data as data
from utils.evaluation import evaluate_detections
from layers.box_utils import decode
from util import DataClass, fetch_in_thread, display_in_thread, detect_in_thread, init_cam, init_ssd, mark_detections, Y_OFFSET_GT_BOX

from threading import Thread

import os, time
import argparse
import numpy as np
import pickle
import scipy.io as sio # to save detection as mat files
cfg = v2

INTERVAL_CONFIDIENCE = 0.02
CONFIDIENCE_MAX = 0.9
CONFIDIENCE_MIN = 0.001 


FONT_SCALE_GT_VID = 0.6 
FONT_SCALE_GT_BOX = 0.5 
FONT_SCALE_FPS = 0.6 

X_OFFSET_GT_VID = 80
Y_OFFSET_GT_VID = 20



#N_RECORD_PER_CLASS = -1
#N_RECORD_PER_CLASS = 100

#g_shall_record = N_RECORD_PER_CLASS > 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--dataset', default='ucf24', help='pretrained base model')
parser.add_argument('--ssd_dim', default=300, type=int, help='Input Size for SSD') # only support 300 now
parser.add_argument('--input_type', default='rgb', type=str, help='INput tyep default rgb can take flow as well')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
#parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
parser.add_argument('--eval_iter', default='120000', type=str, help='Number of training iterations')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--data_root', default='./', help='Location of VOC root directory')
#parser.add_argument('--data_root', default='/mnt/mars-fast/datasets/', help='Location of VOC root directory')
parser.add_argument('--save_root', default='./', help='Location to save checkpoint models')
#parser.add_argument('--save_root', default='/mnt/mars-gamma/datasets/', help='Location to save checkpoint models')
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')

#parser.add_argument('--conf_thresh', default=0.01, type=float, help='Confidence threshold for evaluation')
#parser.add_argument('--conf_thresh', default=0.05, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--conf_thresh', default=0.2, type=float, help='Confidence threshold for evaluation')

#parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
parser.add_argument('--nms_thresh', default=0.5, type=float, help='NMS threshold')

#parser.add_argument('--topk', default=20, type=int, help='topk for evaluation')
parser.add_argument('--topk', default=10, type=int, help='topk for evaluation')

parser.add_argument('--id_cam', default='-1', type=str, help='camera index, -1 for image folder input')

parser.add_argument('--n_record', default=0, type=int, help='number of frames per class to record')

args = parser.parse_args()

if args.input_type != 'rgb':
    args.conf_thresh = 0.05

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def make_class_color_list(n_class):
    li_color_bgr = []
    li_color_bgr.append((255, 0, 0))
    li_color_bgr.append((0, 255, 0))
    li_color_bgr.append((0, 0, 255))
    if len(li_color_bgr) < n_class:
        li_color_bgr.append((255, 255, 0))
        li_color_bgr.append((255, 0, 255))
        li_color_bgr.append((0, 255, 255))
        if len(li_color_bgr) < n_class:
            li_color_bgr.append((255, 128, 0))
            li_color_bgr.append((255, 0, 128))
            li_color_bgr.append((0, 255, 128))
            if len(li_color_bgr) < n_class:
                li_color_bgr.append((128, 255, 0))
                li_color_bgr.append((128, 0, 255))
                li_color_bgr.append((0, 128, 255))
                if len(li_color_bgr) < n_class:
                    li_color_bgr.append((128, 128, 0))
                    li_color_bgr.append((128, 0, 128))
                    li_color_bgr.append((0, 128, 128))
                    if len(li_color_bgr) < n_class:
                        more = n_class - len(li_color_bgr)                        
                        t1 = [(random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)) for k in range(more)]
                        li_color_bgr += t1 
    
    #print('li_color_bgr : ', li_color_bgr);  # exit()
    return li_color_bgr



def mark_ground_truth(im_bgr, gt, tu_str_class, li_color_class):
    n_gt = gt.shape[0]
    for ik in range(n_gt):
        #print('gt[ik] : ', gt[ik])
        #if 0 < np.sum(gt[ik, :4]):
        #print('gt[ik, 4] : ', gt[ik, 4]);  # exit()
        if 9999 > gt[ik, 4]:
            cv2.rectangle(im_bgr, (gt[ik, 0], gt[ik, 1]), (gt[ik, 2], gt[ik, 3]), (255, 255, 255), 1)
            id_class_gt = int(gt[ik, 4])
            #str_class = dataset.CLASSES[id_class_gt]
            str_class = tu_str_class[id_class_gt]
            #cv2.putText(t3_bgr, str_class, (int(gt[ik, 0]), int(gt[ik, 1] + Y_OFFSET)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_class[id_class_gt])
            cv2.putText(im_bgr, str_class, (int(gt[ik, 0]), int(gt[ik, 1] + Y_OFFSET_GT_BOX)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_GT_BOX, li_color_class[id_class_gt])
            #cv2.imshow('t3_bgr', t3_bgr); cv2.waitKey(1);    #exit()

    return im_bgr



def convert_vid_2_animated_gif(fn_record):
    return os.system('ffmpeg -i {} -r 10 {}'.format(fn_record, os.path.splitext(fn_record)[0] + '.gif'))
 

#def test_net(net, save_root, exp_name, input_type, dataset, iteration, num_classes, thresh=0.5 ):
#def test_net(net, save_root, exp_name, input_type, dataset, iteration, li_color_class, means_bgr, n_record_per_class, thresh=0.5 ):
def test_net(net, save_root, exp_name, input_type, dataset, iteration, li_color_class, means_bgr, n_record_per_class, th_iou):
    """ Test a SSD network on an Action image database. """
    '''
    print('type(means) : ', type(means))
    print('means : ', means)
    '''
    #li_color_class = make_class_color_list(num_classes)
    shall_record = n_record_per_class > 0
    th_conf = args.conf_thresh
    th_nms = args.nms_thresh
    top_k = args.topk
    t3 = np.asarray(means_bgr)
    means_rgb = np.flipud(t3)
    #means_rgb_2 = np.fliplr(t3)
    #print('t3 : ', t3); print('means_rgb_1 : ', means_rgb_1);   exit(); #print('means_rgb_2 : ', means_bgr_2);   exit()
    

    #val_data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=detection_collate, pin_memory=True)
    val_data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=detection_collate, pin_memory=True)
    image_ids = dataset.ids
    save_ids = []
    val_step = 250
    num_images = len(dataset)
    video_list = dataset.video_list
    '''
    print('type(dataset) : ', type(dataset)); 
    print('num_images : ', num_images); 
    print('len(video_list) : ', len(video_list));  exit()
    '''
    det_boxes = [[] for _ in range(len(CLASSES))]
    gt_boxes = []
    print_time = True
    batch_iterator = None
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    num_batches = len(val_data_loader)
    det_file = save_root + 'cache/' + exp_name + '/detection-'+ input_type + '_' + str(iteration).zfill(6)+'.pkl'
    print('det_file : ', det_file); #exit()
    print('Number of images ', len(dataset),' number of batchs', num_batches)
    frame_save_dir = save_root + 'detections/CONV-' + input_type + '-' + args.listid + '-' + str(iteration).zfill(6) + '/'
    print('\n\n\nDetections will be store in ',frame_save_dir,'\n\n')
    if shall_record:
        di_class_num_processed = {}
        fn_record = 'action_recognition_images_conf_thres_{:.2f}_nms_thres_{:.1f}_fpc_{}.avi'.format(th_conf, th_nms, n_record_per_class)
        writer = make_video_recorder(fn_record, (300, 300), 20)
    shall_stop = False
    for val_itr in range(len(val_data_loader)):
        print('\nval_itr : {} / {}'.format(val_itr, len(val_data_loader)))
        if not batch_iterator:
            batch_iterator = iter(val_data_loader)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        images_rgb, targets, img_indexs = next(batch_iterator)
        batch_size = images_rgb.size(0)
        if shall_record:
            skip_this_batch = False
            for b in range(batch_size):
                img_idx = img_indexs[b]
                annot_info = dataset.ids[img_idx]
                video_id = annot_info[0]
                video_name = dataset.video_list[video_id].split("/")[0]
                if video_name in di_class_num_processed:
                    if di_class_num_processed[video_name] > n_record_per_class:
                        skip_this_batch = True
                        break    
                    di_class_num_processed[video_name] += 1
                else:
                    di_class_num_processed[video_name] = 1
            if skip_this_batch:
                continue

        height, width = images_rgb.size(2), images_rgb.size(3)
        li_margin_ratio_l_r_t_b = [0, 0, 0, 0];
        if args.cuda:
            images_rgb = Variable(images_rgb.cuda(), volatile=True)
            #exit()
        #print('images_rgb.shape : ', images_rgb.shape)
########    networking forwarding ######################################################        
        output = net(images_rgb)
######################################################################################
        loc_data = output[0]
        conf_preds = output[1]
        prior_data = output[2]

        if print_time and val_itr % val_step == 0:
            torch.cuda.synchronize()
            tf = time.perf_counter()
            print('Forward Time {:0.3f}'.format(tf - t1))

        #   for each image in this batch
        for b in range(batch_size):
            #print('b : {} / {}'.format(b, batch_size)) 
            img_idx = img_indexs[b]
            annot_info = dataset.ids[img_idx]
            #print('annot_info : ', annot_info)
            video_id = annot_info[0]
            frame_num = annot_info[1]

            #print('video_id : ', video_id)
            video_name = dataset.video_list[video_id]
            video_class = video_name.split("/")[0]
            img_name = dataset._imgpath + '/{:s}/{:05d}.jpg'.format(video_name, frame_num)
            #print('video_name : ', video_name)
            #print('video_class : ', video_class)
            print('img_name : ', img_name)
            
            #t1_rgb = np.transpose(images_rgb[b].cpu().numpy(), (1, 2, 0))
            #exit()
            t1_rgb = np.transpose(images_rgb[b].cpu().data.numpy(), (1, 2, 0))

            t2_rgb = t1_rgb + means_rgb
            t3_bgr = cv2.cvtColor(t2_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            gt = targets[b].numpy()
            gt[:, 0] *= width
            gt[:, 2] *= width
            gt[:, 1] *= height
            gt[:, 3] *= height
            #print('type(gt) : ', type(gt)); exit()
            #cv2.putText(t3_bgr, video_name, (60, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255))
            id_vid = dataset.CLASSES.index(video_class)
            cv2.putText(t3_bgr, video_class, (X_OFFSET_GT_VID, Y_OFFSET_GT_VID), cv2.FONT_HERSHEY_DUPLEX, FONT_SCALE_GT_VID, li_color_class[id_vid])
            
            cv2.putText(t3_bgr, "conf. thres. : {:.2f}".format(th_conf), (int(width * 0.5 - 85), int(height - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
            
            if not shall_record:
                t3_bgr = mark_ground_truth(t3_bgr, gt, dataset.CLASSES, li_color_class)
            gt_boxes.append(gt)

           
            decoded_boxes = decode(loc_data[b].data, prior_data.data, cfg['variance']).clone()
            conf_scores = net.softmax(conf_preds[b]).data.clone()    
            
            t3_bgr, det_boxes = mark_detections(t3_bgr, conf_scores, dataset.CLASSES, decoded_boxes, (width, height), li_margin_ratio_l_r_t_b, li_color_class, top_k, th_conf, th_nms, det_boxes)

            #index = img_indexs[b]
            annot_info = image_ids[img_idx]
            #exit()

            frame_num = annot_info[1]; video_id = annot_info[0]; videoname = video_list[video_id]
            output_dir = frame_save_dir + videoname
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            output_file_name = output_dir+'/{:05d}.mat'.format(int(frame_num))
            save_ids.append(output_file_name)
            sio.savemat(output_file_name, mdict={'scores':conf_scores.cpu().numpy(),'loc':decoded_boxes.cpu().numpy()})
            if shall_record:
                writer.write(t3_bgr)
            count += 1
            cv2.imshow('t3_bgr', t3_bgr); 
            #cv2.waitKey(1)
            k = cv2.waitKey() & 0xFF
            #k = cv2.waitKey(1)
            '''
            if 255 != k:
                print('k : ', k)
            '''
            if 27 == k:
                shall_stop = True
        if val_itr % val_step == 0:
            torch.cuda.synchronize()
            te = time.perf_counter()
            print('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te - ts))
            torch.cuda.synchronize()
            ts = time.perf_counter()
        if print_time and val_itr % val_step == 0:
            torch.cuda.synchronize()
            te = time.perf_counter()
            print('NMS stuff Time {:0.3f}'.format(te - tf))
        if shall_stop:
            break
    print('Evaluating detections for itration number ', iteration)
    
    #Save detection after NMS along with GT
    with open(det_file, 'wb') as f:
        pickle.dump([gt_boxes, det_boxes, save_ids], f, pickle.HIGHEST_PROTOCOL)
    if shall_record:
        writer.release()
        convert_vid_2_animated_gif(fn_record)
    return evaluate_detections(gt_boxes, det_boxes, CLASSES, iou_thresh=th_iou)


def action_detection_images(num_classes, means_bgr, li_color_class):

    exp_name = 'CONV-SSD-{}-{}-bs-{}-{}-lr-{:05d}'.format(args.dataset, args.input_type, args.batch_size, args.basenet[:-14], int(args.lr * 100000))
    print('Exp name', exp_name, args.listid)
    for iteration in [int(itr) for itr in args.eval_iter.split(',')]:
        log_file = open(args.save_root + 'cache/' + exp_name + "/testing-{:d}.log".format(iteration), "w", 1)
        log_file.write(exp_name + '\n')
        #trained_model_path = args.save_root + 'cache/' + exp_name + '/ssd300_ucf24_' + repr(iteration) + '.pth'
        trained_model_path = args.save_root + 'cache/' + exp_name + '/' + args.input_type + '-ssd300_ucf24_' + repr(iteration) + '.pth'
        log_file.write(trained_model_path+'\n')
        net = init_ssd(num_classes, trained_model_path, args.cuda)
        print('Finished loading model %d !' % iteration)
        # Load dataset
        dataset = UCF24Detection(args.data_root, 'test', BaseTransform(args.ssd_dim, means_bgr), AnnotationTransform(), input_type=args.input_type, full_test=True)
        #print('dataset.CLASSES : ', dataset.CLASSES);   exit()
        # evaluation
        torch.cuda.synchronize()
        tt0 = time.perf_counter()
        log_file.write('Testing net \n')
        #mAP, ap_all, ap_strs = test_net(net, args.save_root, exp_name, args.input_type, dataset, iteration, num_classes)
        mAP, ap_all, ap_strs = test_net(net, args.save_root, exp_name, args.input_type, dataset, iteration, li_color_class, means_bgr, args.n_record, args.iou_thresh)
        for ap_str in ap_strs:
            log_file.write(ap_str + '\n')
        ptr_str = '\nMEANAP:::=>' + str(mAP) + '\n'
        print(ptr_str)
        log_file.write(ptr_str)

        torch.cuda.synchronize()
        print('Complete set time {:0.2f}'.format(time.perf_counter() - tt0))
        log_file.close()
    return

        

#  im_bgr, n_processed = compute_and_draw_fps(im_bgr, n_processed, sec_start)
def compute_and_draw_fps_and_cofidence_threshold(im_bgr, n_processed, sec_start, w_h_img, th_conf):
    wid, hei = w_h_img
    torch.cuda.synchronize()
    sec_end = time.perf_counter()
    fps = n_processed / (sec_end - sec_start)
    #print('fps : ', fps);   exit()
    cv2.putText(im_bgr, "conf. thres. : {:.2f}".format(th_conf), (int(wid * 0.5 - 80), int(hei * 0.97)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_CONFIDIENCE, (0, 255, 0))
    cv2.putText(im_bgr, "fps : {:.1f}".format(fps), (int(wid * 0.5 - 50), int(hei * 0.06)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_FPS, (0, 0, 255))
    return im_bgr

def make_video_recorder(fn_vid, w_h_vid, fps):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    return cv2.VideoWriter(fn_vid, fourcc, fps, w_h_vid)


def action_detection_cam(id_cam, num_classes, means_bgr, w_h_net, li_color_class):
    #print('CLASSES : ', CLASSES)
    th_conf = args.conf_thresh;    th_nms = args.nms_thresh
    use_cuda = args.cuda
    top_k = args.topk
    #print('args.n_record : ', args.n_record);   exit()
    shall_record = args.n_record > 0
    #   initialize camera
    kam, w_h_cam = init_cam(id_cam)
    #   initialize network
    exp_name = 'CONV-SSD-{}-{}-bs-{}-{}-lr-{:05d}'.format(args.dataset, args.input_type, args.batch_size, args.basenet[:-14], int(args.lr * 100000))
    trained_model_path = args.save_root + 'cache/' + exp_name + '/' + args.input_type + '-ssd300_ucf24_' + args.eval_iter + '.pth'
       
    #li_margin_ratio_l_r_t_b = compute_margin_ratio_l_r_t_b(w_h_cam, w_h_net)
    n_processed = 0
    torch.cuda.synchronize();   sec_start = time.perf_counter()

    class_data = DataClass()

    #thread_fetch  = Thread(target = fetcm62m65m q2n 6M M6_in_thread, args = (class_data, fn_video, 8))
    thread_fetch  = Thread(target = fetch_in_thread, args = (class_data, kam, means_bgr, w_h_net, use_cuda))
    thread_fetch.start()

    #thread_detect  = Thread(target = detect_in_thread, args = (class_data, num_classes, trained_model_path, CLASSES, w_h_cam, w_h_net, use_cuda, th_conf, th_nms))
    thread_detect  = Thread(target = detect_in_thread, args = (class_data, num_classes, trained_model_path,use_cuda, cfg))
    thread_detect.start()

    #thread_display  = Thread(target = display_in_thread, args = (class_data, COLORS))
    thread_display  = Thread(target = display_in_thread, args = (class_data, shall_record, w_h_cam, w_h_net, CLASSES, li_color_class, top_k, th_conf, th_nms))
    thread_display.start()

    thread_fetch.join()
    thread_detect.join()
    thread_display.join()

    '''
    
    #   for each frame
    while True:
        #   capture a frame
        ret, im_bgr = kam.read()
        if ret:
            #   make a batch
            batch_rgb = im2batch(im_bgr, means_bgr, w_h_net)
            #   net forwarding
            loc_data, conf_preds, prior_data = net(batch_rgb) 
            #   post process output
            im_bgr, _ = post_process_output(im_bgr, net, CLASSES, loc_data, conf_preds, prior_data, w_h_cam, li_margin_ratio_l_r_t_b, li_color_class, th_conf, th_nms, None)
            n_processed += 1
            im_bgr = compute_and_draw_fps_and_cofidence_threshold(im_bgr, n_processed, sec_start, w_h_cam, th_conf)
            #   display bboxes
            cv2.imshow('im_bgr', im_bgr)
            if shall_record:
                writer.write(im_bgr)    
            
            k = cv2.waitKey(1) & 0xFF
            #k = cv2.waitKey(1)
            if 255 != k:
                print('k : ', k)
            if 27 == k:
                break
            elif 82 == k or 84 == k:
                th_conf_b4 = th_conf
                if  82 == k:
                    th_conf += INTERVAL_CONFIDIENCE
                    th_conf = min(th_conf, CONFIDIENCE_MAX)
                else:
                    th_conf -= INTERVAL_CONFIDIENCE
                    th_conf = max(th_conf, CONFIDIENCE_MIN)
                print('th_conf changed from : {} to {}'.format(th_conf_b4, th_conf))  
 
    cv2.destroyAllWindows()
    if shall_record:
        writer.release()
        convert_vid_2_animated_gif(fn_record)
    '''
    return 



'''
def action_detection_cam(id_cam, num_classes, means_bgr, w_h_net, li_color_class):
    print('CLASSES : ', CLASSES)
    th_conf = args.conf_thresh
    th_nms = args.nms_thresh
    shall_record = args.n_record > 0
    #   initialize camera
    kam, w_h_cam = init_cam(id_cam)
    #   initialize network
    exp_name = 'CONV-SSD-{}-{}-bs-{}-{}-lr-{:05d}'.format(args.dataset, args.input_type, args.batch_size, args.basenet[:-14], int(args.lr * 100000))
    trained_model_path = args.save_root + 'cache/' + exp_name + '/' + args.input_type + '-ssd300_ucf24_' + args.eval_iter + '.pth'
    net = init_ssd(num_classes, trained_model_path)
    li_margin_ratio_l_r_t_b = compute_margin_ratio_l_r_t_b(w_h_cam, w_h_net)
    n_processed = 0
    torch.cuda.synchronize();   sec_start = time.perf_counter()
    if shall_record:
        fn_record = 'action_recognition_cam_conf_thres_{:.2f}_nms_thres_{:.1f}.avi'.format(th_conf, th_nms)
        writer = make_video_recorder(fn_record, w_h_cam, 20)
    #   for each frame
    while True:
        #   capture a frame
        ret, im_bgr = kam.read()
        if ret:
            #   make a batch
            batch_rgb = im2batch(im_bgr, means_bgr, w_h_net)
            #   net forwarding
            loc_data, conf_preds, prior_data = net(batch_rgb) 
            #   post process output
            im_bgr, _ = post_process_output(im_bgr, net, CLASSES, loc_data, conf_preds, prior_data, w_h_cam, li_margin_ratio_l_r_t_b, li_color_class, th_conf, th_nms, None)
            n_processed += 1
            im_bgr = compute_and_draw_fps_and_cofidence_threshold(im_bgr, n_processed, sec_start, w_h_cam, th_conf)
            #   display bboxes
            cv2.imshow('im_bgr', im_bgr)
            if shall_record:
                writer.write(im_bgr)    
            
            k = cv2.waitKey(1) & 0xFF
            #k = cv2.waitKey(1)
            if 255 != k:
                print('k : ', k)
            if 27 == k:
                break
            elif 82 == k or 84 == k:
                th_conf_b4 = th_conf
                if  82 == k:
                    th_conf += INTERVAL_CONFIDIENCE
                    th_conf = min(th_conf, CONFIDIENCE_MAX)
                else:
                    th_conf -= INTERVAL_CONFIDIENCE
                    th_conf = max(th_conf, CONFIDIENCE_MIN)
                print('th_conf changed from : {} to {}'.format(th_conf_b4, th_conf))  
 
    cv2.destroyAllWindows()
    if shall_record:
        writer.release()
        convert_vid_2_animated_gif(fn_record)
    return 
'''

def main():

    #print('main')
    means_bgr = (104, 117, 123)  # only support voc now


    args.save_root += args.dataset+'/'
    args.data_root += args.dataset+'/'
    args.listid = '01' ## would be usefull in JHMDB-21
    num_classes = len(CLASSES) + 1  #7 +1 background
    #print('args.eval_iter : ', args.eval_iter); exit()
    li_color_class = make_class_color_list(num_classes)
    if '-1' == args.id_cam:
        action_detection_images(num_classes, means_bgr, li_color_class)
    else:
        action_detection_cam(args.id_cam, num_classes, means_bgr, (300, 300), li_color_class)
        
if __name__ == '__main__':
    main()
