import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import localLoaderTest
import modelsImproved
import numpy as np
from PIL import Image
import sys
import os
import glob

img_size = 448
grid_num = 7
grid_size = 64.0
NMS_confidence_threshold = 0.01
NMS_IOU_threshold = 0.5
current_validate_img = -1
images_filenames = glob.glob(os.path.join(sys.argv[1], '*.jpg'))
images_filenames = sorted(images_filenames, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

DOTA_CLASSES = (  # always index 0
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'small-vehicle', 'large-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field',
    'swimming-pool', 'container-crane')
     
def collate_fn(batch):
    img, label= zip(*batch)
    labels = []
    for i in range(len(label)):
        labels.append(label[i])
    return torch.stack(img), labels

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path,map_location='cuda')
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def get_IOU(rec1, rec2):
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        rec1_area = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        rec2_area = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        return (intersect) / (rec1_area + rec2_area - intersect)
    
def get_IOU_index(GT_x1, GT_y1, GT_x2, GT_y2, bbox1_cx, bbox1_cy, bbox1_w, bbox1_h, bbox2_cx, bbox2_cy, bbox2_w, bbox2_h, corner_x, corner_y):
    #x is horizon, y is vertical, different from traditional 2D data 
    bbox1_x1 = (corner_x + (bbox1_cx * grid_size)) - ((bbox1_w * img_size) / 2.0)
    bbox1_x2 = (corner_x + (bbox1_cx * grid_size)) + ((bbox1_w * img_size) / 2.0)
    bbox1_y1 = (corner_y + (bbox1_cy * grid_size)) - ((bbox1_h * img_size) / 2.0)
    bbox1_y2 = (corner_y + (bbox1_cy * grid_size)) + ((bbox1_h * img_size) / 2.0)
    bbox2_x1 = (corner_x + (bbox2_cx * grid_size)) - ((bbox2_w * img_size) / 2.0)
    bbox2_x2 = (corner_x + (bbox2_cx * grid_size)) + ((bbox2_w * img_size) / 2.0)
    bbox2_y1 = (corner_y + (bbox2_cy * grid_size)) - ((bbox2_h * img_size) / 2.0)
    bbox2_y2 = (corner_y + (bbox2_cy * grid_size)) + ((bbox2_h * img_size) / 2.0)
    
    #deal with IOU
    GT = [GT_y1, GT_x1, GT_y2, GT_x2]
    bbox1 = [bbox1_y1, bbox1_x1, bbox1_y2, bbox1_x2]
    bbox2 = [bbox2_y1, bbox2_x1, bbox2_y2, bbox2_x2]
    
    GT_bbox1_IOU = get_IOU(GT, bbox1)
    GT_bbox2_IOU = get_IOU(GT, bbox2)
    
    #return shift
    if GT_bbox1_IOU >= GT_bbox2_IOU:
        if GT_bbox1_IOU != 0:
            return 0, GT_bbox1_IOU
        else:
            return 0, 1
    else:
        return 5, GT_bbox2_IOU
        
def parse_target(output, target):
    shaped_output = torch.zeros((len(target),grid_num,grid_num,26), requires_grad=True)
    shaped_output = shaped_output.clone()
    for img_ind in range(len(target)):
            for j in range(grid_num):
                for i in range(grid_num):
                    include_GT = False
                    bbox_list = []
                    for bbox_ind in range(len(target[img_ind])):
                        GT_center_x = (target[img_ind][bbox_ind][0] + target[img_ind][bbox_ind][2]) / 2.0
                        GT_center_y = (target[img_ind][bbox_ind][1] + target[img_ind][bbox_ind][3]) / 2.0
                        current_x = i * grid_size
                        current_y = j * grid_size
                        if (current_x <= GT_center_x and GT_center_x <= current_x+grid_size) and (current_y <= GT_center_y and GT_center_y <= current_y+grid_size):
                            include_GT = True
                            bbox_list.append(bbox_ind)                        
                            break
                    if include_GT:
                        for current_bbox_ind in bbox_list:
                            #GT part
                            xi_hat = (((target[img_ind][current_bbox_ind][0] + target[img_ind][current_bbox_ind][2]) / 2.0) - (i * grid_size)) / grid_size
                            yi_hat = (((target[img_ind][current_bbox_ind][1] + target[img_ind][current_bbox_ind][3]) / 2.0) - (j * grid_size)) / grid_size
                            wi_hat = (target[img_ind][current_bbox_ind][2] - target[img_ind][current_bbox_ind][0]) / float(img_size)
                            hi_hat = (target[img_ind][current_bbox_ind][3] - target[img_ind][current_bbox_ind][1]) / float(img_size)
                            
                            #IOU part
                            
                            IOU_index_shift, IOU_area = get_IOU_index(target[img_ind][current_bbox_ind][0], target[img_ind][current_bbox_ind][1],
                                                            target[img_ind][current_bbox_ind][2], target[img_ind][current_bbox_ind][3],
                                                            output[img_ind][j][i][0].detach(), output[img_ind][j][i][1].detach(), output[img_ind][j][i][2].detach(), output[img_ind][j][i][3].detach(),
                                                            output[img_ind][j][i][5].detach(), output[img_ind][j][i][6].detach(), output[img_ind][j][i][7].detach(), output[img_ind][j][i][8].detach(), i*grid_size, j*grid_size)
                           
                            
                            shaped_output[img_ind][j][i][0+IOU_index_shift] = xi_hat
                            shaped_output[img_ind][j][i][1+IOU_index_shift] = yi_hat
                            shaped_output[img_ind][j][i][2+IOU_index_shift] = wi_hat
                            shaped_output[img_ind][j][i][3+IOU_index_shift] = hi_hat
                            shaped_output[img_ind][j][i][4+IOU_index_shift] = 1 * IOU_area
                            shaped_output[img_ind][j][i][target[img_ind][current_bbox_ind][4]+10] = 1
                            
    return shaped_output

class custom_loss(nn.Module):
    def __init__(self):
        super(custom_loss, self).__init__()
        
    def forward(self, outputs, targets, device):
        targets = targets.to(device)
        outputs = outputs.to(device)
        
        first_obj_mask = targets[:,:,:,4] > 0
        second_obj_mask = targets[:,:,:,9] > 0
        first_nobj_mask = targets[:,:,:,4] == 0
        second_nobj_mask = targets[:,:,:,9] == 0
        
        #x y w h loss
        firsh_coords_loss = F.mse_loss(targets[first_obj_mask][:,0:2], outputs[first_obj_mask][:,0:2], reduction='sum')
        firsh_sizes_loss = F.mse_loss(torch.sqrt(targets[first_obj_mask][:,2:4]), torch.sqrt(outputs[first_obj_mask][:,2:4]), reduction='sum')
        second_coords_loss = F.mse_loss(targets[second_obj_mask][:,5:7], outputs[second_obj_mask][:,5:7], reduction='sum')
        second_sizes_loss = F.mse_loss(torch.sqrt(targets[second_obj_mask][:,7:9]), torch.sqrt(outputs[second_obj_mask][:,7:9]), reduction='sum')
        coords_loss = 5 * (firsh_coords_loss + firsh_sizes_loss + second_coords_loss + second_sizes_loss)
        
        #cls loss
        first_obj_cls_loss = F.mse_loss(targets[first_obj_mask][:,4], outputs[first_obj_mask][:,4], reduction='sum')
        first_nobj_cls_loss = F.mse_loss(targets[first_nobj_mask][:,4], outputs[first_nobj_mask][:,4], reduction='sum')
        second_obj_cls_loss = F.mse_loss(targets[second_obj_mask][:,9], outputs[second_obj_mask][:,9], reduction='sum')
        second_nobj_cls_loss = F.mse_loss(targets[second_nobj_mask][:,9], outputs[second_nobj_mask][:,9], reduction='sum')
        cls_loss = (first_obj_cls_loss + second_obj_cls_loss) + 0.5 * (first_nobj_cls_loss + second_nobj_cls_loss)
        
        #16D predict loss
        first_D_predict_loss = F.mse_loss(targets[first_obj_mask][:,10:], outputs[first_obj_mask][:,10:], reduction='sum')
        second_D_predict_loss = F.mse_loss(targets[second_obj_mask][:,10:], outputs[second_obj_mask][:,10:], reduction='sum')
        D_predict_loss = first_D_predict_loss + second_D_predict_loss

        return (coords_loss + cls_loss + D_predict_loss) / targets.shape[0]
        
def NMS_and_output(datas):
    #sort datas
    datas = datas[datas[:,4].argsort()[::-1]]
    
    #make filename and open
    global current_validate_img
    current_validate_img += 1
    current_output_filename = images_filenames[current_validate_img]
    current_output_filename = current_output_filename.split('/')[-1]
    current_output_filename = current_output_filename.replace('.jpg', '.txt')
    output_filename = sys.argv[2] + current_output_filename
    
    with open(output_filename, 'w') as f:
        while datas[0][4] != 0:
            #pick candidate
            x1 = datas[0][0] * 512 / img_size
            y1 = datas[0][1] * 512 / img_size
            x2 = datas[0][2] * 512 / img_size
            y2 = datas[0][3] * 512 / img_size
            cls = DOTA_CLASSES[int(datas[0][5])]
            cof = datas[0][4]
            print("{} {} {} {} {} {} {} {} {} {}".format(x1, y1, x2, y1, x2, y2, x1, y2, cls, cof), file=f)
            datas[0][4] = 0
            
            #filter low IOU
            for i in range(1,len(datas)):
                if datas[i][4] == 0:
                    continue
                bbox1 = [datas[0][1], datas[0][0], datas[0][3], datas[0][2]]
                bbox2 = [datas[i][1], datas[i][0], datas[i][3], datas[i][2]]
                if get_IOU(bbox1, bbox2) > NMS_IOU_threshold:
                    datas[i][4] = 0
            
            #sort again
            datas = datas[datas[:,4].argsort()[::-1]]
    
def parse_output(output):
    #output = batchsize * 7 * 7 * 26
    output = output.cpu().numpy()
    
    for img_ind in range(len(output)):
        datas = np.zeros((1, 6))
        for j in range(grid_num):
            for i in range(grid_num):
                #bbox1
                x1 = ((output[img_ind][j][i][0] * grid_size) + (i * grid_size)) - ((img_size * output[img_ind][j][i][2]) / 2.0)
                x2 = ((output[img_ind][j][i][0] * grid_size) + (i * grid_size)) + ((img_size * output[img_ind][j][i][2]) / 2.0)
                y1 = ((output[img_ind][j][i][1] * grid_size) + (j * grid_size)) - ((img_size * output[img_ind][j][i][3]) / 2.0)
                y2 = ((output[img_ind][j][i][1] * grid_size) + (j * grid_size)) + ((img_size * output[img_ind][j][i][3]) / 2.0)
                confidence = output[img_ind][j][i][4] * max(output[img_ind][j][i][10:])
                cls = output[img_ind][j][i][10:].argmax()
                if confidence > NMS_confidence_threshold:
                    bbox1 = np.array((x1, y1, x2, y2, confidence, cls))
                    datas = np.vstack((datas, bbox1.reshape((1, 6))))  
                #bbox2
                x1 = ((output[img_ind][j][i][5] * grid_size) + (i * grid_size)) - ((img_size * output[img_ind][j][i][7]) / 2.0)
                x2 = ((output[img_ind][j][i][5] * grid_size) + (i * grid_size)) + ((img_size * output[img_ind][j][i][7]) / 2.0)
                y1 = ((output[img_ind][j][i][6] * grid_size) + (j * grid_size)) - ((img_size * output[img_ind][j][i][8]) / 2.0)
                y2 = ((output[img_ind][j][i][6] * grid_size) + (j * grid_size)) + ((img_size * output[img_ind][j][i][8]) / 2.0)
                confidence = output[img_ind][j][i][9] * max(output[img_ind][j][i][10:])
                cls = output[img_ind][j][i][10:].argmax()
                if confidence > NMS_confidence_threshold:
                    bbox2 = np.array((x1, y1, x2, y2, confidence, cls))
                    datas = np.vstack((datas, bbox2.reshape((1, 6))))  
        NMS_and_output(datas)
       
def test(model, testset_loader, device, log_interval=25):
    load_checkpoint('hw2-97.pth', model, 111)
    model.eval()
    iteration = 0
    with torch.no_grad():
        for batch_idx, (data) in enumerate(testset_loader):
            data = data.to(device)
            output = model(data)
            parse_output(output) 
            iteration += 1
        global current_validate_img
        current_validate_img = -1
        
def train(model, trainset_loader, testset_loader, device, epoch, log_interval=100):
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr = 0.0001, weight_decay=0.00001)
    criterion = custom_loss()
    iteration = 0
    for ep in range(epoch):
        '''
        model.train()
        for batch_idx, (data, target) in enumerate(trainset_loader):
            optimizer.zero_grad()
            data = data.to(device)
            output = model(data)
            target = parse_target(output, target)
            loss = criterion(output, target, device)
            loss.backward()
            optimizer.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    ep, batch_idx * len(data), len(trainset_loader.dataset),
                    100. * batch_idx / len(trainset_loader), loss.item()))
            iteration += 1
        save_checkpoint('checkpoints_grid13/hw2-%i.pth' % ep, model, optimizer)
        '''
        test(model, testset_loader, device)

def main():
    print('-------------start predicting-------------')
    #check device
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Device used:', device)
    
    #model create and to device
    model = modelsImproved.Yolov1_vgg16bn(pretrained=True).to(device)
    
    if not os.path.exists(sys.argv[2]):
        os.mkdir(sys.argv[2])
    
    testset = localLoaderTest.AERIAL(sys.argv[1], transform=transforms.ToTensor())
    testset_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=1)
    test(model, testset_loader, device)
    print('------------finish predicting-------------')

if __name__ == '__main__':
    main()
