import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from utils.losses import BoundaryLoss, DiceLoss, FocalLoss
import torch.nn.functional as F
from Data_Loader import *
from models.UNet import Unet,resnet34_unet
from utils.metrics import *
from torchvision.transforms import transforms
from utils.plot import loss_plot
from utils.plot import metrics_plot
import os

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument("--epoch", type=int, default=300)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='UNet',
                       help='UNet')
    parse.add_argument("--batch_size", type=int, default=10)
    parse.add_argument('--dataset', default='HK2_DIC',
                       help='dataset name:/NMuMg/T47D/HK2_DIC')
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold",type=float,default=None)
    parse.add_argument("--num_classes", type=int, default=3)
    args = parse.parse_args()
    return args

def getLog(args):
    dirname = os.path.join(args.log_dir,args.arch,str(args.batch_size),str(args.dataset),str(args.epoch))
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

def getModel(args):
    if args.arch == 'UNet':
        model = Unet(1, 3).to(device)
    if args.arch == 'resnet34_unet':
        model = resnet34_unet(pretrained=True).to(device)
    return model

def getDataset(args):
    train_dataloaders, val_dataloaders ,test_dataloaders= None,None,None
    if args.dataset == 'NMuMg':
        train_dataset = NMuMgDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = NMuMgDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = NMuMgDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'T47D':
        train_dataset = T47DDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = T47DDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = T47DDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'HK2_DIC':
        train_dataset = HK2_DICDataset(r'train',transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = HK2_DICDataset(r"val",transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = HK2_DICDataset(r"test",transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    return train_dataloaders,val_dataloaders,test_dataloaders

def val(model,best_iou,val_dataloaders):
    model= model.eval()
    with torch.no_grad():
        i=0
        miou_total = 0
        pixel_accuracy_total= 0
        boundary_iou_total = 0
        num = len(val_dataloaders)
        print('test set img:',num)
        for x, _, dist_map_label,pic,mask in val_dataloaders:
            x = x.to(device)
            y = model(x)
            if args.deepsupervision:
                img_y = torch.squeeze(y).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()
            img_gt = Image.open(mask[0])
            img_gt = Image.fromarray(np.uint8(img_gt))
            img_gt = np.asarray(img_gt)
            pred=np.argmax(img_y,axis=0)
            pixel_accuracy_total+= pixel_accuracy(pred,img_gt)
            miou_total += mean_IU(pred,img_gt)
            boundary_iou_total += mean_IU_boundary(img_gt, pred)
            if i < num:i+=1
        torch.save(model.state_dict(), r'./saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(
            args.dataset) + '_' + str(args.epoch) + '.pth')
        aver_pixel_accuracy=pixel_accuracy_total/num
        aver_iou = miou_total / num
        aver_boundary_iou = boundary_iou_total / num
        print('aver_pixel_accuracy=%f, Miou=%f, boundary_iou=%f' % (aver_pixel_accuracy,aver_iou,aver_boundary_iou))
        logging.info('aver_pixel_accuracy=%f, Miou=%f, boundary_iou=%f' % (aver_pixel_accuracy,aver_iou,aver_boundary_iou))

        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            torch.save(model.state_dict(), r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth')
        return best_iou,aver_iou,aver_pixel_accuracy, aver_boundary_iou


def train(model, criterion, optimizer, train_dataloader,val_dataloader, args):
    best_iou,aver_iou,aver_pixel_accuracy, aver_boundary_iou= 0,0,0,0
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    iou_list = []
    pixel_accuracy_list = []
    boundary_iou_list = []
    global w
    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('w={}'.format(w))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0

        for x, y, dist_map_label, _, mask in train_dataloader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            if args.deepsupervision:
                outputs = model(inputs)
                loss = 0
                for output in outputs:
                    output = output.to(device)
                    loss += criterion(output, torch.argmax(labels, dim=1))
                loss /= len(outputs)
            else:
                output = model(inputs)
                output = output.to(device)
                #original loss
                #loss = criterion(output, torch.argmax(labels, dim=1))

                #boundary loss
                #l2_loss = L2Loss(model, 0.0001)
                #l1_loss = L1Loss(model, 0.00001)
                dist_map_label = dist_map_label.to(device)
                pred_probs = F.softmax(output, dim=1)
                r_loss = dice_loss(pred_probs, labels)
                loss = w*b_loss(pred_probs, dist_map_label) + r_loss
            if threshold!=None:
                if loss > threshold:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
            logging.info("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
        loss_list.append(epoch_loss)
        best_iou,aver_iou,aver_pixel_accuracy,aver_boundary_iou = val(model,best_iou,val_dataloader)
        iou_list.append(aver_iou)
        pixel_accuracy_list.append(aver_pixel_accuracy)
        boundary_iou_list.append(aver_boundary_iou)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        if epoch > 200:
            w = w + 0.0001
    loss_plot(args, loss_list)
    metrics_plot(args, 'iou',iou_list)
    metrics_plot(args,'pixel accuracy',pixel_accuracy_list)
    metrics_plot(args, 'boundary iou', boundary_iou_list)
    return model

def test(val_dataloaders,save_predict=False):
    logging.info('final test........')
    if save_predict ==True:
        dir = os.path.join(r'./saved_predict',str(args.arch),str(args.batch_size),str(args.epoch),str(args.dataset))
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')
    model.load_state_dict(torch.load(r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth', map_location='cpu'))  # 载入训练好的模型
    model.eval()
    #plt.ion()
    with torch.no_grad():
        i=0
        miou_total = 0
        pixel_accuracy_total = 0
        boundary_iou_total = 0
        avgHausdorff_total = 0
        Hausdorff_total = 0
        dice_total = 0
        num = len(val_dataloaders)
        pred_np=[]
        for pic,_,dist_map,pic_path,mask_path in val_dataloaders:
            pic = pic.to(device)
            predict = model(pic)
            if args.deepsupervision:
                predict = torch.squeeze(predict[-1]).cpu().numpy()
            else:
                predict = torch.squeeze(predict).cpu().numpy()
            img_gt = Image.open(mask_path[0])
            img_gt = Image.fromarray(np.uint8(img_gt))
            img_gt = np.asarray(img_gt)
            pred = np.argmax(predict, axis=0)
            cv2.imwrite(dir+'/'+str(i)+".png",pred)
            pred_np.append(pred)
            pixel_acc=pixel_accuracy(pred, img_gt)
            pixel_accuracy_total += pixel_acc
            iou=mean_IU(pred, img_gt)
            miou_total += iou
            boundary_iou_total += boundary_iou(img_gt, pred)
            quality = computeQualityMeasures(pred, img_gt)
            avgHausdorff_total += quality["avgHausdorff"]
            Hausdorff_total += quality["Hausdorff"]
            dice_total += quality["dice"]
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.set_title('input')
            plt.imshow(Image.open(pic_path[0]))
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.set_title('predict')
            plt.imshow(pred)
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.set_title('mask')
            plt.imshow(img_gt)
            if save_predict == True:
                if args.dataset == 'driveEye':
                    saved_predict = dir + '/' + mask_path[0].split('\\')[-1]
                    saved_predict = '.'+saved_predict.split('.')[1] + '.tif'
                    plt.savefig(saved_predict)
                else:
                    plt.savefig(dir +'/'+ mask_path[0].split('/')[-1])
            if i < num:i+=1
        plt.show()
        aver_pixel_accuracy = pixel_accuracy_total / num
        aver_iou = miou_total / num
        aver_boundary_iou = boundary_iou_total / num
        aver_Hausdorff = Hausdorff_total /num
        aver_avgHausdorff = avgHausdorff_total / num
        aver_dice = dice_total / num
        np.save(os.path.join(dir,'ori_dic.npy'),pred_np)

        print('**************************')
        print('pixel_accuracy=%f, mIoU=%f, boundary_iou=%f, Hausdorff=%f, avgHausdorff=%f, dice= %f' % (aver_pixel_accuracy, aver_iou, aver_boundary_iou,aver_Hausdorff, aver_avgHausdorff,aver_dice))
        logging.info('aver_pixel_accuracy=%f, Miou=%f, aver_boundary_iou=%f' % (aver_pixel_accuracy, aver_iou, aver_boundary_iou))


if __name__ =="__main__":
    palette = [[0], [1], [2]]
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize((0.5,), (0.5,)),
    ])
    y_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    print(device)

    args = getArgs()
    logging = getLog(args)
    print('**************************')
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    print('**************************')
    model = getModel(args)
    #model.load_state_dict(torch.load('./saved_model/myChannelUnet_4_T47D_41.pth'))  # 再加载网络的参数
    #model = model.to(device)
    #print("load success")
    train_dataloaders,val_dataloaders,test_dataloaders = getDataset(args)
    #criterion = torch.nn.CrossEntropyLoss()

    dice_loss = DiceLoss(idc=[0,1,2])
    focal_loss = FocalLoss(idc=[0,1,2],gamma=2)
    b_loss = BoundaryLoss(idc=[2])
    ce_loss = torch.nn.CrossEntropyLoss()
    global w 
    w = 0
    criterion = dice_loss

    optimizer = optim.Adam(model.parameters())
    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders,val_dataloaders, args)
    if 'test' in args.action:
        test(test_dataloaders, save_predict=True)
