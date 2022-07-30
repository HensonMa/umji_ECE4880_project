import torch
import torch.quantization as tq
import torch.nn as nn
import os
from vgg19_brevitas import *
from vgg import *
from torchvision import datasets
import torchvision.transforms as transforms
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from utils import *
import argparse


class Normal_trainer:
    def __init__(self, args):
        self.path = {}
        self.path["root"] = os.getcwd()
        self.path["result_path"] = os.path.join(self.path["root"], args.result_path)
        self.path["data_path"] = args.data_path
        if not os.path.exists(self.path["result_path"]):
            os.makedirs(self.path["result_path"])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.args = args
        self.bs = args.batchsize
        self.start_epoch = args.start_epoch
        self.epoch_num = args.epoch_num
        self.list = [[], [], [], [], []]
        self.num_class = args.num_class
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.type = args.type
        self.random_crop = args.random_crop
        self.horizontal_flip = args.horizontal_flip
        self.model_name = args.model_name
        self.seed_num = args.seed

        self.lr = args.lr
        self.lr_schedule = args.lr_schedule
        self.logspace = args.logspace
        self.lr_step_size = args.lr_step_size
        self.lr_step_gamma = args.lr_step_gamma
        self.bit_width = args.bit_width
        self.depth = args.depth

        self.save_hparam(args)

    def save_hparam(self, args):
        savepath = os.path.join(self.path["result_path"], "hparam.txt")
        with open(savepath, "w") as fp:
            args_dict = args.__dict__
            for key in args_dict:
                fp.write("{} : {}\n".format(key, args_dict[key]))

    def prepare_model(self):
        if self.model_name == "vgg19":
            self.model = vgg19_bn(num_class=self.num_class, dataset=self.type)
        elif self.model_name == "vgg19_quantized":
            self.model = vgg19_quantized(num_class=self.num_class, dataset=self.type, batch_norm=True, bit_width=self.bit_width, depth=self.depth)

        if self.start_epoch != 0:
            self.load_latest_epoch()

        self.loss_func = nn.CrossEntropyLoss(reduction="mean")
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                             weight_decay=self.weight_decay)

    def load_latest_epoch(self):
        model_path = os.path.join(self.path["result_path"], "model_" + str(self.start_epoch - 1) + ".bin")
        list_path = os.path.join(self.path["result_path"], "list_" + str(self.start_epoch - 1) + ".bin")

        self.model.load_state_dict(torch.load(model_path))
        self.list = torch.load(list_path)

    def save_latest_epoch(self, epoch):
        model_path = os.path.join(self.path["result_path"], "model_" + str(epoch) + ".bin")
        list_path = os.path.join(self.path["result_path"], "list_" + str(epoch) + ".bin")

        torch.save(self.model.state_dict(), model_path)
        torch.save(self.list, list_path)

    def prepare_dataset(self):
        if self.type == 'cifar10':
            self.img_size = 32
            self.crop_size = 36
            self.num_class = 10
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2023, 0.1994, 0.2010]
        elif self.type == "imagenet":
            self.img_size = 224
            self.crop_size = 256
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        if self.random_crop:
            train_transform_list = [
                transforms.Resize((self.crop_size, self.crop_size)),
                transforms.RandomCrop(self.img_size),
            ]
        else:
            train_transform_list = [
                transforms.Resize((self.img_size, self.img_size))
            ]
        if self.horizontal_flip:
            train_transform_list.append(transforms.RandomHorizontalFlip(0.5))

        train_transform_list.append(transforms.ToTensor())
        train_transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))
        train_transform = transforms.Compose(train_transform_list)
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        if self.type == 'cifar10':
            self.train_set = datasets.CIFAR10(root=self.path["data_path"], train=True, download=True, transform=train_transform)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.bs, shuffle=True, num_workers=2)

            self.test_set = datasets.CIFAR10(root=self.path["data_path"], train=False, download=True, transform=test_transform)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.bs, shuffle=False, num_workers=2)

        elif self.type == "imagenet":
            self.train_set = datasets.ImageFolder(root=self.path['data_path'] + '/train/', transform=train_transform)
            self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.bs, shuffle=True,
                                                            num_workers=2)

            self.test_set = datasets.ImageFolder(root=self.path['data_path'] + '/val_split/', transform=test_transform)
            self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.bs, shuffle=False,
                                                           num_workers=2)

    def train(self, epoch):
        self.model.train()
        self.model.to(self.device)
        torch.autograd.set_detect_anomaly(True)

        Loss, Acc_num = 0, 0
        bt_id = 0
        for (img, lbl) in tqdm(self.train_loader):
            bt_id += 1
            img = img.to(self.device)
            lbl = lbl.to(self.device)
            output = self.model(img)
            if type(output).__name__ == 'tuple':
                output = output[0]
            loss = self.loss_func(output, lbl)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pre = output.detach().max(1)[1]
            acc_num = (pre == lbl).sum().float().item()
            Acc_num += acc_num

            Loss += loss.cpu().item()

        print("[train epoch {}] loss: %.3f, acc: %.3f".format(epoch) % (Loss / (bt_id + 1), Acc_num / len(self.train_set)))
        self.list[0].append(Loss / (bt_id + 1))
        self.list[1].append(Acc_num / len(self.train_set))

    def test(self, epoch):
        self.model.eval()
        self.model.to(self.device)

        Loss, Acc_num = 0, 0
        bt_id = 0
        inference_time = []
        for (img, lbl) in tqdm(self.test_loader):
            with torch.no_grad():
                bt_id += 1
                img = img.to(self.device)
                lbl = lbl.to(self.device)

                start_time = time.time()
                output = self.model(img)
                end_time = time.time()
                if type(output).__name__ == 'tuple':
                    output = output[0]
                loss = self.loss_func(output, lbl)
                pre = output.detach().max(1)[1]
                acc_num = (pre == lbl).sum().float().item()
                Acc_num += acc_num

                Loss += loss.cpu().item()
            inference_time.append(end_time-start_time)

        print("[test epoch {}] loss: %.3f, acc: %.3f, infer(gpu): %.3f s / per batch of size {}".format(epoch, self.bs) % (Loss / (bt_id + 1), Acc_num / len(self.test_set), sum(inference_time)/len(inference_time)))

        self.list[2].append(Loss / (bt_id + 1))
        self.list[3].append(Acc_num / len(self.test_set))
        self.list[4].append(sum(inference_time)/len(inference_time))

    def draw_figure(self):
        x = np.arange(0, len(self.list[0]), 1)
        y = np.arange(0, len(self.list[2]), 1)
        train_l, train_e = np.array(self.list[0]), np.array(self.list[1])
        test_l, test_e = np.array(self.list[2]), np.array(self.list[3])
        plt.figure()
        plt.subplot(221)
        plt.plot(x, train_l, color='red')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.subplot(222)
        plt.plot(y, test_l, color='blue')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.subplot(223)
        plt.plot(x, train_e, color='red', label='train')
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.subplot(224)
        plt.plot(y, test_e, color='blue', label='test')
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.legend()

        plt.savefig(os.path.join(self.path["result_path"], "curve.png"))
        plt.close()

    def print_and_save_list(self):
        save_path = os.path.join(self.path["result_path"], "list.txt")
        with open(save_path, "w") as fp:
            train_loss = self.list[0]
            train_acc = self.list[1]
            test_loss = self.list[2]
            test_acc = self.list[3]
            inference_time = self.list[4]

            for i in range(len(train_loss)):
                fp.write("epoch:{}, train_loss:{}, train_acc:{}, test_loss:{}, test_acc:{}, infer(cpu): {}s/ per batch of size {}\n".format(
                    i + 1,
                    train_loss[i],
                    train_acc[i],
                    test_loss[i],
                    test_acc[i],
                    inference_time[i],
                    self.bs
                ))
            fp.write("the last epoch, train_loss:{}, train_acc:{}, test_loss:{}, test_acc:{}, infer(cpu): {}s/ per batch of size {}\n".format(
                train_loss[-1],
                train_acc[-1],
                test_loss[-1],
                test_acc[-1],
                inference_time[i],
                self.bs
            ))

    def work(self):
        self.prepare_model()
        self.prepare_dataset()

        if self.lr_schedule == 'log':
            if self.logspace != 0:
                self.logspace_lr = np.logspace(np.log10(self.lr), np.log10(self.lr) - self.logspace, self.epoch_num)
        elif self.lr_schedule == 'step':
            if self.start_epoch != 0:
                self.lr_scheduler = my_step_lr_Scheduler(self.lr, self.optimizer, step_size=self.lr_step_size, gamma=self.lr_step_gamma, last_epoch=self.start_epoch-1)
                self.lr_scheduler.resume_the_last_epoch()
            else:
                self.lr_scheduler = StepLR(self.optimizer, step_size=self.lr_step_size, gamma=self.lr_step_gamma)

        for epoch in range(self.start_epoch, self.epoch_num):
            seed_torch(epoch + self.bs)
            if self.lr_schedule == 'log':
                if self.logspace != 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.logspace_lr[epoch]

            if epoch % 50 == 0 and epoch != 0:
                self.save_latest_epoch(epoch)

            self.train(epoch)
            self.test(epoch)

            if epoch == self.epoch_num - 1:
                self.save_latest_epoch(epoch + 1)

            if self.lr_schedule == 'step':
                self.lr_scheduler.step()

            self.draw_figure()
        self.draw_figure()
        self.print_and_save_list()

    def benchmark(self):
        self.prepare_model()
        self.prepare_dataset()
        self.test(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_crop", default=True, type=bool)
    parser.add_argument("--horizontal_flip", default=True, type=bool)
    parser.add_argument("--result_dir", default='pretrained_models', type=str)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--root", default='./cifar10_data', type=str)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--lr-schedule", default='step', type=str)
    parser.add_argument("--logspace", default=1, type=int)
    parser.add_argument("--lr-step-size", default=50, type=int)
    parser.add_argument("--lr-step-gamma", default=0.1, type=float)
    parser.add_argument("--epoch-num", default=100, type=int)  # 150
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batchsize", default=512, type=int)
    parser.add_argument("--type", default="cifar10", type=str)
    parser.add_argument("--num-class", default=10, type=int)
    parser.add_argument("--model-name", default="vgg19_quantized", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--bit-width", default=4, type=int)
    parser.add_argument("--depth", default=[5], type=int, nargs='+')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    if args.lr_schedule == 'log':
        lr_setting = "{}_{}".format(args.lr_schedule, args.logspace)
    elif args.lr_schedule == 'step':
        lr_setting = "{}_{}_{}".format(args.lr_schedule, args.lr_step_size, args.lr_step_gamma)
    else:
        raise RuntimeError("no such learning rate!")

    if "quantized" in args.model_name:
        args.result_path = "./" + args.result_dir + "/" + args.model_name + "_" + args.type + \
                           "/bitWidth{}_depth{}_seed{}_lr{}_{}_epoch{}".format(args.bit_width, args.depth, args.seed, args.lr, lr_setting, args.epoch_num)
    else:
        args.result_path = "./" + args.result_dir + "/" + args.model_name + "_" + args.type + \
                           "/seed{}_lr{}_{}_epoch{}".format(args.seed, args.lr, lr_setting, args.epoch_num)

    print(args.result_path)
    seed_torch(args.seed)
    args.data_path = args.root
    trainer = Normal_trainer(args)
    trainer.work()


if __name__ == "__main__":
    main()

