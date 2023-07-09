import cv2
from subprocess import PIPE, run
import subprocess
import ast
import torch
import math
import json
from torch import tensor
import os
import numpy as np
import matplotlib.pylab as plt
from skimage import transform,io, color
import sys
import numpy as np
import torch
from torch import tensor
import cv2
import pandas as pd


people = {}
people_cropped_imgs = {}
human_has_appered = {}
tensor_3d ={}
tensor_reg ={}
tensor_3d_all ={}
tensor_reg_all ={}
img_size_all = {}
lst_size = []

def bounding_box(xy1,xy2,frame , img_size_dict , frame_num):
    x_left= int(min(xy1[0],xy2[0]))
    x_right= int(max(xy1[0],xy2[0]))
    dx=(x_right-x_left)
    y_down = int(max(xy1[1], xy2[1]))
    y_up = int(min(xy1[1], xy2[1]))
    dy = (y_down - y_up)
    print(f"bounding box size height: {abs(y_down-y_up)+1}")
    print(f"bounding box size wifth: {abs(x_left-x_right)+1}")
    lst_size = [abs(y_down-y_up)+1,abs(x_left-x_right)+1]
    img_size_dict[str(frame_num)] = lst_size
    print(img_size_dict)
    x_left=int(max(0,x_left-dx*0.2))
    x_right = int(min(frame.shape[0]-1, x_right + dx * 0.2))
    y_up = int(max(0, y_up - dy * 0.2))
    y_down = int(min(frame.shape[1]-1, y_down + dy * 0.2))
    #print(f"x_left={x_left},x_right={x_right},y_up={y_up},y_down={y_down}")
    after_cropping=frame[x_left:x_right+1,y_up:y_down+1]
    #print("xy1",xy1,"\nxy2,",xy2)
    return after_cropping

def string_to_tensor(tensor_string):
  tensor_string = tensor_string.replace(", device='cuda:0'", "")  # remove the device argument
  Tensor = eval(tensor_string).clone()
  return Tensor


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def send_to_yolov(frame_name, img_num, vid_name):
    """# Activate virtual environment using virtualenv
    venv_name = '/home/ofiryonatan/yolov7'
    activate_this = '\home\ofiryonatan\yolov7\\bin\\activate'
    #activate_this = os.path.abspath(os.path.join(venv_name, 'bin', 'activate'))
    #activate_this = os.path.join(venv_name, 'bin', 'activate')
    exec(open(activate_this).read())

    venv_name = 'yolov7_conda_env'
    activate_this = f'source activate {venv_name}'
    os.system(activate_this)
    """
    # Your code here (running in the virtual environment)
    FPS = 30
    command = [
        'python', '/home/ofiryonatan/yolov7/detect.py',
        '--weights', '/home/ofiryonatan/yolov7/yolov7.pt',
        '--conf', '0.25',
        '--img-size', '640',
        '--source', '/data/students/ofiryonatan_data/yolov_img/'+ path[1]+ str(img_num) + '.jpg',
        '--project', f'/data/students/ofiryonatan_data/yolov_out/{vid_name}',
        '--device', cuda_arg , '--name', f"{img_num}"
    ]
    #print(command)
    # Call the Python program with some command line arguments
    found = False
    search = True
    ENLARGE = 4
    while (not found):
        result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True) # call the yolov
        output = result.stdout #place it in some data_structure
        out_list = output.split("\n")
        relevent_row_count = 0
        output = ""
        for row in out_list:
            relevent_row_count+=1
            if(relevent_row_count>8):
                output+=row
        print(output)
        if("tensor" in out_list[8]):
            found =True
        if(out_list[8] == ""):
            found = True
            tensor_reg[str(img_num)] = None
            tensor_3d[str(img_num)] = None
            search = False
    #t = string_to_tensor(output)
    #print(people)
    print(tensor_3d_all)
    if search: #if we found someone
        t = string_to_tensor(output)#take all the objects that we have detected
        for human in t:
            if human[5] == 0 and human[4] > 0.4: #we take only the objects that were classified as humand (human[5] == 0) and are at a pretty high change of being this object (human[4] > 0.4)
                if len(people) != 0: #if there are people to compare to
                    found = False
                    for key in people:#check all the people
                        if len(key) > 0 :
                            if nearest(people[key][-1],human,FPS):
                                found = True
                                #tensor_3d_all[str(len(tensor_3d_all))] = {}
                                index = list(people.keys()).index(key)
                                people[key].append(human)
                                cropped_frame = bounding_box([human[1], human[0]], [human[3], human[2]], frame_name , img_size_all[str(list(people.keys()).index(key))],img_num )
                                #people_cropped_imgs[key].append(cropped_frame)
                                cv2.imwrite("/data/students/ofiryonatan_data/for_SR/"+path[1] +str(index)+'.jpg', cropped_frame)  # the frame is the single picture!!!
                                if ENLARGE > 1:
                                  send_to_SR(cropped_frame, '/data/students/ofiryonatan_data/for_SR/'+path[1]+str(index) +'.jpg', ENLARGE)
                                  img = cv2.imread('/home/ofiryonatan/our_code/SR_imgs/'+path[1]+str(index) +'_out.jpg')
                                else:
                                  img = cropped_frame
                                  print("check")
                                people_cropped_imgs[key].append(img)
                                # height, width, _ = img.shape
                                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                # out = cv2.VideoWriter('/data/students/ofiryonatan_data/video/'+path[1]+'_'+str(index)+'_'+str(img_num)+ 'output_video.mp4', fourcc, 1, (width, height))
                                # out.write(img)
                                # out.release()
                                # send_to_MMPOSE(img_num,tensor_3d_all[str(list(people.keys()).index(key))],index ,img_size_all[str(list(people.keys()).index(key))])


                                human_has_appered[key] = True
                    if not found:#create a new one if it is not close enough to the other therefore its not one of the previous people
                        people[str(len(people)+1)] = []
                        people_cropped_imgs[str(len(people))] = []
                        people[str(len(people))].append(human)
                        tensor_3d_all[str(len(tensor_3d_all))] = {}
                        img_size_all[str(len(img_size_all))] = {}
                        index = len(tensor_3d_all)-1
                        cropped_frame = bounding_box([human[1], human[0]], [human[3], human[2]], frame_name, img_size_all[str(len(tensor_3d_all)-1)],img_num )
                        #people_cropped_imgs[str(len(people))].append(cropped_frame)
                        cv2.imwrite("/data/students/ofiryonatan_data/for_SR/"+path[1]+str(index) +'.jpg', cropped_frame)  # the frame is the single picture!!!
                        if ENLARGE > 1:
                          send_to_SR(cropped_frame, '/data/students/ofiryonatan_data/for_SR/'+path[1]+str(index) +'.jpg', ENLARGE)
                          img = cv2.imread('/home/ofiryonatan/our_code/SR_imgs/'+path[1]+str(index) +'_out.jpg')
                        else:
                          img = cropped_frame
                          print("check")
                        people_cropped_imgs[str(len(people))].append(img)
                        # height, width, _ = img.shape
                        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        # out = cv2.VideoWriter('/data/students/ofiryonatan_data/video/'+path[1]+'_'+str(index)+'_'+str(img_num)+ 'output_video.mp4', fourcc, 1, (width, height))
                        # out.write(img)
                        # out.release()
                        # send_to_MMPOSE(img_num,tensor_3d_all[str(len(tensor_3d_all)-1)],index , img_size_all[str(len(tensor_3d_all)-1)])
                        human_has_appered[str(len(people))] = True

                else:#create the first one
                    #print("boop")
                    people['1'] = []
                    people_cropped_imgs['1'] = []
                    people['1'].append(human)
                    tensor_3d_all[str(len(tensor_3d_all))] = {}
                    img_size_all[str(len(img_size_all))] = {}
                    index = 0
                    #print(tensor_3d_all)
                    #cv2.imshow("shit", frame_name[541:625,300:458])
                    #cv2.imwrite('test_zone.jpg', frame_name[458:541,300:625])
                    cropped_frame = bounding_box([human[1], human[0]], [human[3], human[2]], frame_name, img_size_all[str(len(tensor_3d_all)-1)],img_num )
                    #people_cropped_imgs['1'].append(cropped_frame)
                    cv2.imwrite('/data/students/ofiryonatan_data/for_SR/'+path[1]+str(index) +'.jpg', cropped_frame)  # the frame is the single picture!!!
                    if ENLARGE > 1:
                      send_to_SR(cropped_frame, '/data/students/ofiryonatan_data/for_SR/'+path[1] +str(index)+'.jpg', ENLARGE)
                      img = cv2.imread('/home/ofiryonatan/our_code/SR_imgs/'+path[1] +str(index)+'_out.jpg')
                    else:
                      img = cropped_frame
                      print("check")
                    people_cropped_imgs['1'].append(img)
                    # height, width, _ = img.shape
                    # print(f"after SR h,w = {img.shape}")
                    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    # out = cv2.VideoWriter('/data/students/ofiryonatan_data/video/'+path[1]+'_'+str(index)+'_'+str(img_num)+ 'output_video.mp4', fourcc, 1, (width, height))
                    # out.write(img)
                    # out.release()
                    # send_to_MMPOSE(img_num,tensor_3d_all[str(len(tensor_3d_all)-1)],index , img_size_all[str(len(tensor_3d_all)-1)])
                    human_has_appered['1'] = True
        # for person in human_has_appered: #don't sure if it works
        #     if human_has_appered[person] is False:
        #         del human_has_appered[person]
        #         del people[person]
                #save him
        # Deactivate virtual environment

def nearest(previous_human_found, current_human,FPS):
    #close_enough = True
    prev_x_left = int(min(previous_human_found[1], previous_human_found[3]))
    prev_x_right = int(max(previous_human_found[1], previous_human_found[3]))
    prev_y_down = int(max(previous_human_found[0], previous_human_found[2]))
    prev_y_up = int(min(previous_human_found[0], previous_human_found[2]))
    x_left = int(min(current_human[1], current_human[3]))
    x_right = int(max(current_human[1], current_human[3]))
    y_down = int(max(current_human[0], current_human[2]))
    y_up = int(min(current_human[0], current_human[2]))
    prev_middle = [(prev_x_left+prev_x_right)/2,(prev_y_down+prev_y_up)/2]
    curr_middle = [(x_left+x_right)/2,(y_down+y_up)/2]
    prev_height = prev_y_down - prev_y_up
    curr_height = y_down - y_up
    distance = math.sqrt((prev_middle[0]-curr_middle[0])**2 + (prev_middle[1]-curr_middle[1])**2)
    if(distance < ((200/(51*FPS)) * ((prev_height +curr_height)/2)) ):
        return True
    return False
    """
    need to check the proximity to the previous image!
    """


    return  close_enough


def send_to_SR(frame_name, path,enlarge_resulosion_factor):
    command = []
    command = [
        'python', '/home/ofiryonatan/Real-ESRGAN/inference_realesrgan.py',
        '-n', 'RealESRGAN_x4plus',
        '-i', str(path),
        '-o', '/home/ofiryonatan/our_code/SR_imgs', '-s', str(enlarge_resulosion_factor)
    ]
    #comand = ['python /home/ofiryonatan/Real-ESRGAN/inference_realesrgan.py -n RealESRGAN_x4plus -i /home/ofiryonatan/our_code/test_zone.jpg -o /home/ofiryonatan/our_code/SR_imgs -s 4']
    #for key in people_cropped_imgs:
    #video_name = path+"_"+str(key) + ".mp4"
    #send_to_MMPOSE(1,{},1,[],video_name)
    #command = ['python', "/data/students/ofiryonatan_data/ofiryonatan/Real-ESRGAN/inference_realesrgan_video.py",
    #'-i', "/data/students/ofiryonatan_data/last_data/"+video_name ,
    #'-o' ,'/data/students/ofiryonatan_data/last_data/SR_videos/',
    # '-s', str(enlarge_resulosion_factor)]
    # Call the Python program with some command line arguments
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    output = result.stdout  # place it in some data_structure
    #process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    #stdout, stderr = process.communicate(timeout=10)
    """for line in iter(process.stdout.readline, b''):
         print(line.decode("utf-8"), end='')"""


def send_to_MMPOSE(img_num,tensor_3d_all,index, img_size_dict,video_name,SR):

    #path = arguments[0][:-4].split('/')
    #relevent from here!
    #com = 'python /home/ofiryonatan/mmpose/demo/body3d_two_stage_video_demo.py /home/ofiryonatan/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth /home/ofiryonatan/mmpose/configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/hrnet_w48_posetrack18_384x288_posewarper_stage2.py https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth /home/ofiryonatan/mmpose/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth --video-path "/data/students/ofiryonatan_data/video/'+path[1]+'_'+str(index)+'_'+str(img_num)+ 'output_video.mp4" --out-video-root "/data/students/ofiryonatan_data/test_test/1'+path[1] +'/'+str(index)+'/'+ str(img_num)+'" --rebase-keypoint-height --use-multi-frames --online --device=cuda:'+cuda_arg
    com = 'python /data/students/ofiryonatan_data/ofiryonatan/mmpose/demo/body3d_two_stage_video_demo2.py /data/students/ofiryonatan_data/ofiryonatan/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth /data/students/ofiryonatan_data/ofiryonatan/mmpose/configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/hrnet_w48_posetrack18_384x288_posewarper_stage2.py https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth /data/students/ofiryonatan_data/ofiryonatan/mmpose/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth --video-path "/data/students/ofiryonatan_data/last_data/'+str(video_name)+ '" --out-video-root "/data/students/ofiryonatan_data/test_test/1'+str(video_name)+'" --rebase-keypoint-height --use-multi-frames --online --device=cuda:'+cuda_arg
    
    if(SR):
      com = 'python /data/students/ofiryonatan_data/ofiryonatan/mmpose/demo/body3d_two_stage_video_demo2.py /data/students/ofiryonatan_data/ofiryonatan/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth /data/students/ofiryonatan_data/ofiryonatan/mmpose/configs/body/2d_kpt_sview_rgb_vid/posewarper/posetrack18/hrnet_w48_posetrack18_384x288_posewarper_stage2.py https://download.openmmlab.com/mmpose/top_down/posewarper/hrnet_w48_posetrack18_384x288_posewarper_stage2-4abf88db_20211130.pth /data/students/ofiryonatan_data/ofiryonatan/mmpose/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth --video-path "/data/students/ofiryonatan_data/last_data/SR_videos/'+str(video_name)+ '" --out-video-root "/data/students/ofiryonatan_data/test_test/1'+str(video_name)+'" --rebase-keypoint-height --use-multi-frames --online --device=cuda:'+cuda_arg
    
    
    #print(com)
    #result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    dict_str = ""
    dict_str2 = ""
    test = ""
    count = 0
    print("at mmpose: " + video_name)
    process = subprocess.Popen(com, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    for line in iter(process.stdout.readline, b''):
         print(line.decode("utf-8"), end='')
    #     test+=line.decode("utf-8")
    #     count+=1
    #     if(count > 21 and count <= 55): #23 , 57 for pycharm , 21 55 for command line
    #         dict_str += line.decode("utf-8")
    #     if(count > 23 and count <= 57): #23 , 57 for pycharm , 21 55 for command line
    #         dict_str2 += line.decode("utf-8")
    # dict_str = dict_str.replace(" ", '')
    # dict_str = dict_str.replace("'",'"')
    # dict_str = dict_str.replace("\n", "")
    # dict_str = dict_str.replace("(", "")
    # dict_str = dict_str.replace(")", "")
    # dict_str = dict_str.replace("array", "")
    # dict_str = dict_str.replace(",dtype=float32", "")
    #
    # if(len(dict_str)>0):
    #     if '{' in dict_str and '}' in dict_str:
    #       input_dict = ast.literal_eval(dict_str)
    #     else:
    #       input_dict = ast.literal_eval(dict_str2)
    #     keypoints_tensor = torch.from_numpy(np.array(input_dict['keypoints']))
    #     keypoints_3d_tensor = torch.from_numpy(np.array(input_dict['keypoints_3d']))
    #     #print(keypoints_tensor)
    #     #print(keypoints_3d_tensor)
    #     tensor_reg[str(img_num)] = keypoints_tensor
    #     tensor_3d[str(img_num)] = keypoints_3d_tensor
    #     tensor_3d_all[str(img_num)] = keypoints_3d_tensor
    #     #img_size_dict[str(img_num)] = lst_size
    #     print("succeded in frame")
    #     #print(tensor_reg)
    #
    # #output = process.stdout #place it in some data_structure
    
  


def make_a_vid(video_name):
    frame_rate = 30.0
    name_of_vid = ""
    image_directory = '/home/ofiryonatan/our_code/img_test/'
    i =0
    for key in people_cropped_imgs:
        name_of_vid = str(key)
        print("ofir print: "+str(people_cropped_imgs[key][0].shape))
        resolution = (people_cropped_imgs[key][0].shape[1] , people_cropped_imgs[key][0].shape[0])
        full_vid_name = video_name+ "_"+name_of_vid + ".mp4"
        print("resolution: " + str(resolution) + "\nname: " + full_vid_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # FourCC code for MP4 format
        print(resolution)
        video_writer = cv2.VideoWriter(full_vid_name, fourcc, frame_rate, resolution)
        #print(str(people_cropped_imgs[key]))
        for image in people_cropped_imgs[key]:
            cv2.imwrite('img_test/'+name_of_vid+"_"+video_name+str(i)+'.jpg', image)
            i = i+1
            print(image.shape)
            video_writer.write(image)
            #print("wrote")
        video_writer.release()
        """cv2.destroyAllWindows()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        output_filename = 'output.mp4'
        images = sorted(os.listdir(image_directory))

        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, resolution)
        
        for image_filename in images:
            image_path = os.path.join(image_directory, image_filename)
            img = cv2.imread(image_path)
            video_writer.write(img)
        
        video_writer.release()
        cv2.destroyAllWindows()"""


def ofir_make_images_same_res():
    for key in people_cropped_imgs:
        img_lst = people_cropped_imgs[key]
        max_height = max(img.shape[0] for img in img_lst)
        max_width = max(img.shape[1] for img in img_lst)
        new_img_lst = zero_pad(max_height,max_width,img_lst)
        people_cropped_imgs[key] = new_img_lst

def zero_pad(max_height,max_width,frame_list):
    new_frame_lst = []
    for img in frame_list:
        # org_height, org_width = np.array(frame).shape
        # new_frame = np.zeros((max_height, max_width))
        # #print(new_frame,'\n\n')
        # new_frame[(max_height-org_height)//2:(max_height-org_height)//2+org_height , (max_width-org_width)//2:(max_width-org_width)//2+org_width ]= frame
        # new_frame_lst.append(new_frame)
        pad_height_top = (max_height - img.shape[0]) // 2
        pad_height_bottom = max_height - pad_height_top - img.shape[0]
        pad_width_right = (max_width - img.shape[1]) // 2
        pad_width_left = max_width - pad_width_right - img.shape[1]

        # Pad the image using cv2.copyMakeBorder()
        padded_img = cv2.copyMakeBorder(img, pad_height_top, pad_height_bottom, pad_width_right, pad_width_left, cv2.BORDER_CONSTANT)
        print(str(padded_img.shape[0]) + "and " + str(padded_img.shape[1]))
        # Append the padded image to the result list
        new_frame_lst.append(padded_img)

        #print(new_frame,'\n\n')
    return new_frame_lst



def mmpose_manager(video_original_name,SR):
    for key in people_cropped_imgs:
      video_name = video_original_name+"_"+str(key) + ".mp4"
      send_to_MMPOSE(1,{},1,[],video_name,SR)


def crop_image(frame_name):
    print("blank")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

arguments = sys.argv[1:]
print(arguments[0])
path = arguments[0][:-4].split('/')
cuda_arg = arguments[1]
# Opens the Video file
#cap = cv2.VideoCapture('/data/students/ofiryonatan_data/drone2/drone2_vids/DJI_0002.mp4')
#cap = cv2.VideoCapture("/home/ofiryonatan/our_code/yonatan_printer_to_brodcom_closest_cropped_3.mp4")
#cap = cv2.VideoCapture("/data/students/ofiryonatan_data/short_videos/yonatan_90_deg_BAD/yonatan_90_angle_2.mp4")
#cap = cv2.VideoCapture("/data/students/ofiryonatan_data/last_data/" + arguments[0])
cap = cv2.VideoCapture("/data/students/ofiryonatan_data/last_data/" + arguments[0])

i = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    print("for frame " +str(i))
    #print(tensor_reg)
    tensor_3d[str(i)] = []
    tensor_reg[str(i)] = []
    cv2.imwrite('/data/students/ofiryonatan_data/yolov_img/' + path[1] + str(i) + '.jpg', frame)  # the frame is the single picture!!!
    send_to_yolov(frame,i,path[1])
    #crop_image(frame)
    #cv2.imwrite('kang' + str(i) + '.jpg', frame) # the frame is the single picture!!!
    i += 1
ofir_make_images_same_res()
make_a_vid(path[1])
#send_to_SR("nada", path[1],4)
#make_SR(path[1]))
mmpose_manager(path[1],0)
f = open("tensors_files/"+path[1]+".txt", "a")
#f.write("3d\n")
#f.write(str(tensor_3d))
f.write(str(tensor_3d_all))
#f.write("\nnot 3d\n")
#f.write(str(tensor_reg))
f.close()
#print("3d")
#print(tensor_3d)
#print("not 3d")
#print(tensor_reg)
g = open("tensors_files/sizes_"+path[1]+".txt", "a")
g.write(str(img_size_all))
g.close()
#print(tensor_3d_all)
print(img_size_all)
print()
print("done")

cap.release()
cv2.destroyAllWindows()
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




