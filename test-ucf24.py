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
import matplotlib
matplotlib.use('Agg')
#'''
import cv2
import torch
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import AnnotationTransform, UCF24Detection, BaseTransform, CLASSES, detection_collate, v2
from ssd import build_ssd
import torch.utils.data as data
from layers.box_utils import decode, nms
from utils.evaluation import evaluate_detections
import os, time
import argparse
import numpy as np
import pickle
import scipy.io as sio # to save detection as mat files
cfg = v2

Y_OFFSET = 15

#N_RECORD_PER_CLASS = -1
N_RECORD_PER_CLASS = 100

g_shall_record = N_RECORD_PER_CLASS > 0

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
parser.add_argument('--data_root', default='/mnt/mars-fast/datasets/', help='Location of VOC root directory')
parser.add_argument('--save_root', default='/mnt/mars-gamma/datasets/', help='Location to save checkpoint models')
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--conf_thresh', default=0.01, type=float, help='Confidence threshold for evaluation')
#parser.add_argument('--conf_thresh', default=0.1, type=float, help='Confidence threshold for evaluation')
#parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
parser.add_argument('--nms_thresh', default=0.1, type=float, help='NMS threshold')
parser.add_argument('--topk', default=20, type=int, help='topk for evaluation')

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
    
    print('li_color_bgr : ', li_color_bgr);  # exit()
    
    return li_color_bgr


#def test_net(net, save_root, exp_name, input_type, dataset, iteration, num_classes, thresh=0.5 ):
def test_net(net, save_root, exp_name, input_type, dataset, iteration, num_classes, means_bgr, thresh=0.5 ):
    """ Test a SSD network on an Action image database. """
    '''
    print('type(means) : ', type(means))
    print('means : ', means)
    '''
    color_class = make_class_color_list(num_classes)
    t3 = np.asarray(means_bgr)
    means_rgb = np.flip(t3)
    #print('t4 : ', means_bgr)


    val_data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=detection_collate, pin_memory=True)
    #val_data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=detection_collate, pin_memory=True)
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
    frame_save_dir = save_root+'detections/CONV-'+input_type+'-'+args.listid+'-'+str(iteration).zfill(6)+'/'
    print('\n\n\nDetections will be store in ',frame_save_dir,'\n\n')
    di_class_num_processed = {}
    if g_shall_record:
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        writer = cv2.VideoWriter('output.avi', fourcc, 20, (300, 300))
    for val_itr in range(len(val_data_loader)):
        print('val_itr : {} / {}'.format(val_itr, len(val_data_loader)))
        if not batch_iterator:
            batch_iterator = iter(val_data_loader)

        #if 50 == val_itr:
        #    break
 



        torch.cuda.synchronize()
        t1 = time.perf_counter()

        images_rgb, targets, img_indexs = next(batch_iterator)
        batch_size = images_rgb.size(0)

        if g_shall_record:
            skip_this_batch = False
            #print('g_shall_record : ', g_shall_record); exit()
            for b in range(batch_size):
                img_idx = img_indexs[b]
                annot_info = dataset.ids[img_idx]
                video_id = annot_info[0]
                video_name = dataset.video_list[video_id].split("/")[0]
                if video_name in di_class_num_processed:
                    if di_class_num_processed[video_name] > N_RECORD_PER_CLASS:
                        skip_this_batch = True
                        break    
                    di_class_num_processed[video_name] += 1
                else:
                    di_class_num_processed[video_name] = 1
            if skip_this_batch:
                continue



        height, width = images_rgb.size(2), images_rgb.size(3)
        #print('type(images) : ', type(images_rgb))
        #print('batch_size : {}'.format(batch_size))
        #print('height : {}, width : {}'.format(height, width))

        if args.cuda:
            images_rgb = Variable(images_rgb.cuda(), volatile=True)
        output = net(images_rgb)

        loc_data = output[0]
        conf_preds = output[1]
        prior_data = output[2]

        if print_time and val_itr % val_step == 0:
            torch.cuda.synchronize()
            tf = time.perf_counter()
            print('Forward Time {:0.3f}'.format(tf - t1))
        for b in range(batch_size):
            #print('b : {} / {}'.format(b, batch_size)) 
            img_idx = img_indexs[b]
            annot_info = dataset.ids[img_idx]
            video_id = annot_info[0]
            video_name = dataset.video_list[video_id].split("/")[0]






            #print('img_idx : ', img_idx)
            #print('video_name : ', video_name); exit()
            
            
            t1_rgb = np.transpose(images_rgb[b].cpu().numpy(), (1, 2, 0))
            t2_rgb = t1_rgb + means_rgb
            t3_bgr = cv2.cvtColor(t2_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            '''
            v_min_1 = np.amin(t1); v_max_1 = np.amax(t1)
            print('t3 : ', t3)
            v_min_2 = np.amin(t2); v_max_2 = np.amax(t2)
            print('t1 : ', t1)
            print('t2 : ', t2)
            print('v_min_1 : {}, v_max_1 : {}'.format(v_min_1, v_max_1))
            print('v_min_2 : {}, v_max_2 : {}'.format(v_min_2, v_max_2))
            print('type(t1) : ', type(t1))
            print('t1.shape : ', t1.shape)
            '''
            '''
            random_image = np.random.random([500,500])
            print('random_image.dtype', random_image.dtype)
            print('t2.dtype', t2.dtype)
            plt.imshow(random_image, cmap='gray', interpolation='nearest')
            print('aaaa')
            plt.show()
            '''
            gt = targets[b].numpy()
            gt[:, 0] *= width
            gt[:, 2] *= width
            gt[:, 1] *= height
            gt[:, 3] *= height
            #print('type(gt) : ', type(gt)); exit()
            n_gt = gt.shape[0]
            cv2.putText(t3_bgr, video_name, (60, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255))


            if not g_shall_record:
                for ik in range(n_gt):
                    #print('gt[ik] : ', gt[ik])
                    #if 0 < np.sum(gt[ik, :4]):
                    #print('gt[ik, 4] : ', gt[ik, 4]);  # exit()
                    if 9999 > gt[ik, 4]:
                        cv2.rectangle(t3_bgr, (gt[ik, 0], gt[ik, 1]), (gt[ik, 2], gt[ik, 3]), (255, 255, 255), 1)
                        id_class_gt = int(gt[ik, 4])
                        str_class = dataset.CLASSES[id_class_gt]
                        cv2.putText(t3_bgr, str_class, (int(gt[ik, 0]), int(gt[ik, 1] + Y_OFFSET)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_class[id_class_gt])
                        #cv2.imshow('t3_bgr', t3_bgr); cv2.waitKey(1);    #exit()

            gt_boxes.append(gt)
            decoded_boxes = decode(loc_data[b].data, prior_data.data, cfg['variance']).clone()
            #print('decoded_boxes : ', decoded_boxes)
            conf_scores = net.softmax(conf_preds[b]).data.clone()
            index = img_indexs[b]
            annot_info = image_ids[index]

            frame_num = annot_info[1]; video_id = annot_info[0]; videoname = video_list[video_id]
            output_dir = frame_save_dir+videoname
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            output_file_name = output_dir+'/{:05d}.mat'.format(int(frame_num))
            save_ids.append(output_file_name)
            sio.savemat(output_file_name, mdict={'scores':conf_scores.cpu().numpy(),'loc':decoded_boxes.cpu().numpy()})

            for cl_ind in range(1, num_classes):
                #'''
                #print();
                #print('cl_ind : {} / {}'.format(cl_ind, num_classes));
                #'''
                str_class = dataset.CLASSES[cl_ind - 1]
                #print('str_class : ', str_class)
                scores = conf_scores[:, cl_ind].squeeze()
                c_mask = scores.gt(args.conf_thresh)  # greater than minmum threshold
                '''
                print('c_mask : ', c_mask)
                print('c_mask.size(): ', c_mask.size())
                print('scores size b4', scores.size())
                print('scores b4 : ', scores)
                print('scores[c_mask] : ', scores[c_mask])
                #scores = scores[c_mask].squeeze()
                '''
                scores = scores[c_mask]
                '''
                #print('scores size : {}'.format(scores.size()))
                print('scores after : ', scores)
                print('scores size after : ', scores.size())
                print('scores.dim() : ', scores.dim())
                print('scores.nelement() : ', scores.nelement())
                #if scores.dim() == 0:
                '''
                #print('scores.nelement() : ', scores.nelement())
                if scores.nelement() == 0:
                    #print(len(''), ' dim ==0 ')
                    det_boxes[cl_ind - 1].append(np.asarray([]))
                    #exit()
                    continue
                boxes = decoded_boxes.clone()
                #print('boxes.shape ori : ', boxes.shape)
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes = boxes[l_mask].view(-1, 4)

                #print('boxes.shape b4 : ', boxes.shape)
                #print('boxes : ', boxes)

                '''
                for ik in range(boxes.shape[0]):
                    cv2.rectangle(t3_bgr, (boxes[ik, 0] * width, boxes[ik, 1] * height), (boxes[ik, 2] * width, boxes[ik, 3] * height), (255, 0, 0), 2)

                    cv2.imshow('t3_bgr', t3_bgr); cv2.waitKey(1);    #exit()
                '''
                # idx of highest scoring and non-overlapping boxes per class
                ids, counts = nms(boxes, scores, args.nms_thresh, args.topk)  # idsn - ids after nms
                #print('ids : ', ids)
                #print('counts : ', counts)
                scores = scores[ids[:counts]].cpu().numpy()
                boxes = boxes[ids[:counts]].cpu().numpy()
                # print('boxes sahpe',boxes.shape)
                boxes[:, 0] *= width
                boxes[:, 2] *= width
                boxes[:, 1] *= height
                boxes[:, 3] *= height

                #print('boxes.shape after : ', boxes.shape)
                for ik in range(boxes.shape[0]):
                    #print('ids[ik] : ', ids[ik].cpu().numpy())
                    boxes[ik, 0] = max(0, boxes[ik, 0])
                    boxes[ik, 2] = min(width, boxes[ik, 2])
                    boxes[ik, 1] = max(0, boxes[ik, 1])
                    boxes[ik, 3] = min(height, boxes[ik, 3])
                    #print('boxes[ik] : ', boxes[ik])
                    cv2.rectangle(t3_bgr, (boxes[ik, 0], boxes[ik, 1]), (boxes[ik, 2], boxes[ik, 3]), color_class[cl_ind - 1], 1)
                    cv2.putText(t3_bgr, str_class, (int(boxes[ik, 0]), int(boxes[ik, 1] + Y_OFFSET)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_class[cl_ind - 1])
                    #cv2.imshow('t3_bgr', t3_bgr); cv2.waitKey();    #exit()

                #exit()
                cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
                det_boxes[cl_ind - 1].append(cls_dets)
                #cv2.imshow('t3_bgr', t3_bgr); cv2.waitKey();    #exit()

            count += 1
            cv2.imshow('t3_bgr', t3_bgr); cv2.waitKey(1)
            if g_shall_record:
                writer.write(t3_bgr)
                #writer.release();   exit()
            #exit()
        #exit()
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
    print('Evaluating detections for itration number ', iteration)
    if g_shall_record:
        writer.release()
    #exit()
    #Save detection after NMS along with GT
    with open(det_file, 'wb') as f:
        #print('b4 dump')
        pickle.dump([gt_boxes, det_boxes, save_ids], f, pickle.HIGHEST_PROTOCOL)
        #print('after dump')

    return evaluate_detections(gt_boxes, det_boxes, CLASSES, iou_thresh=thresh)


def main():

    #print('main')
    means_bgr = (104, 117, 123)  # only support voc now

    exp_name = 'CONV-SSD-{}-{}-bs-{}-{}-lr-{:05d}'.format(args.dataset, args.input_type,
                            args.batch_size, args.basenet[:-14], int(args.lr * 100000))

    args.save_root += args.dataset+'/'
    args.data_root += args.dataset+'/'
    args.listid = '01' ## would be usefull in JHMDB-21
    print('Exp name', exp_name, args.listid)
    '''
    print(args.eval_iter)
    t1 = [itr for itr in args.eval_iter.split(',')]
    print(t1)
    exit()
    '''
    for iteration in [int(itr) for itr in args.eval_iter.split(',')]:
        log_file = open(args.save_root + 'cache/' + exp_name + "/testing-{:d}.log".format(iteration), "w", 1)
        log_file.write(exp_name + '\n')
        #trained_model_path = args.save_root + 'cache/' + exp_name + '/ssd300_ucf24_' + repr(iteration) + '.pth'
        trained_model_path = args.save_root + 'cache/' + exp_name + '/' + args.input_type + '-ssd300_ucf24_' + repr(iteration) + '.pth'
        log_file.write(trained_model_path+'\n')
        num_classes = len(CLASSES) + 1  #7 +1 background
        net = build_ssd(300, num_classes)  # initialize SSD
        net.load_state_dict(torch.load(trained_model_path))
        net.eval()
        if args.cuda:
            net = net.cuda()
            cudnn.benchmark = True
        print('Finished loading model %d !' % iteration)
        # Load dataset
        dataset = UCF24Detection(args.data_root, 'test', BaseTransform(args.ssd_dim, means_bgr), AnnotationTransform(), input_type=args.input_type, full_test=True)
        #print('dataset.CLASSES : ', dataset.CLASSES);   exit()
        # evaluation
        torch.cuda.synchronize()
        tt0 = time.perf_counter()
        log_file.write('Testing net \n')
        #mAP, ap_all, ap_strs = test_net(net, args.save_root, exp_name, args.input_type, dataset, iteration, num_classes)
        mAP, ap_all, ap_strs = test_net(net, args.save_root, exp_name, args.input_type, dataset, iteration, num_classes, means_bgr)
        for ap_str in ap_strs:
            print(ap_str);  #exit()
            log_file.write(ap_str + '\n')
        ptr_str = '\nMEANAP:::=>' + str(mAP) + '\n'
        print(ptr_str)
        log_file.write(ptr_str)

        torch.cuda.synchronize()
        print('Complete set time {:0.2f}'.format(time.perf_counter() - tt0))
        log_file.close()

if __name__ == '__main__':
    main()
