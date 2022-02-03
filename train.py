#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from torch.utils.data import DataLoader
import argparse
from dataset.Imagefolder_modified import Imagefolder_modified
from utils.resnet import *
from PIL import ImageFile # Python：IOError: image file is truncated 的解决办法
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.utils import *
from utils.OLS import OnlineLabelSmoothing

import time

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Manager(object):
    def __init__(self, args):
        """
        Prepare the network, criterion, Optimizer and data
        """
        print('------------------------------------------------------------------------------')
        print('Preparing the network and data ... ')
        self._args = args
        self._path = os.path.join(os.popen('pwd').read().strip(), args.path)
        os.popen('mkdir -p ' + self._path)
        self._data_base = args.data_base
        self._class = args.n_classes
        self._warmup= args.warmup

        print('Basic information: ', 'data:', self._data_base, '    lr:', self._args.base_lr, ' w_decay:', self._args.weight_decay)
        print('Parameter information: ', 'nh:', self._args.nh,
              '  denoise:', self._args.denoise,
              '  relabel:', self._args.relabel,
              '  weight:', self._args.weight,
              '   alpha:', self._args.alpha,
              '   tau:', self._args.tau,)

        if args.net == 'resnet18':
            net = ResNet18(n_classes=args.n_classes, pretrained=True)
        elif args.net == 'resnet50':
            net = ResNet50(n_classes=args.n_classes, pretrained=True)
        else:
            raise AssertionError('Not implemented yet')

        if torch.cuda.device_count() >= 1:
            self._net = torch.nn.DataParallel(net).cuda()
            print('cuda device : ', torch.cuda.device_count())
        else:
            raise EnvironmentError('This is designed to run on GPU but no GPU is found')
        # Criterion
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        self._ols = OnlineLabelSmoothing(num_classes=args.n_classes, use_gpu=True, momentum=self._args.momentum)
        # Optimizer

        params_to_optimize = self._net.parameters()

        self._optimizer = torch.optim.SGD(params_to_optimize, lr=self._args.base_lr,
                                          momentum=0.9, weight_decay=self._args.weight_decay)

        if self._warmup > 0:
            lr_lambda = lambda epoch: epoch / self._warmup
            self._warmupscheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda = lr_lambda)
        else:
            print('no warmup')

        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self._args.epochs - max(0, self._warmup -1))

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        # Load data
        self.train_data = Imagefolder_modified(os.path.join(self._data_base, 'train'), transform=train_transform)
        self.test_data = torchvision.datasets.ImageFolder(os.path.join(self._data_base, 'val'), transform=test_transform)
        assert len(self.train_data.classes) == args.n_classes and len(self.test_data.classes) == args.n_classes, 'number of classes is wrong'
        self._train_loader = DataLoader(self.train_data, batch_size=self._args.batch_size,
                                        shuffle=True, num_workers=8, pin_memory=True)
        self._test_loader = DataLoader(self.test_data, batch_size=self._args.batch_size,
                                       shuffle=False, num_workers=8, pin_memory=True)

        self._js = torch.zeros(len(self.train_data)).cuda()

    def js_selection(self, logits, y, id, epoch):
        losses = (1 - self._args.weight) * F.cross_entropy(logits, y, reduction='none') + self._args.weight * self._ols(logits, y, reduction='none')
        js_batch = js_div(torch.softmax(logits.detach(), dim=1), self._ols.matrix.detach()[y])
        if epoch > 0:
            self._js[id] = js_batch
        if epoch < self._warmup - 1:
            return losses.mean(), losses.size(0)

        # smooth = max(0, (1 - self._args.alpha) / length * (length - (epoch - self._warmup))) # for aircraft and car
        smooth = 0  # for bird
        threshold_clean = self._js[self._js >= 0].mean() + (self._args.alpha + smooth) * self._js[self._js >= 0].std()

        losses = losses[js_batch <= threshold_clean + torch.isnan(js_batch)]
        numb = losses.size(0)
        if numb == 0:
            loss = 0
        else:
            loss = losses.mean()
        return loss, numb

    def js_selection_relabel(self, logits, y, id_global, epoch):
        prob = torch.softmax(logits.detach(), dim=1)
        js_y = js_div(prob, self._ols.matrix.detach()[y])
        if epoch > 0:
            self._js[id_global] = js_y
        if epoch < self._warmup - 1:
            losses = (1 - self._args.weight) * F.cross_entropy(logits, y, reduction='none') + self._args.weight * self._ols(logits, y, reduction='none')
            return losses.mean(), losses.size(0)

        # smooth = max(0, (1 - self._args.alpha) / length * (length - (epoch - self._warmup)))  # for aircraft and car
        smooth = 0  # for bird
        threshold_clean = self._js[self._js >= 0].mean() + (self._args.alpha + smooth) * self._js[self._js >= 0].std()
        threshold_relabel = self._args.tau

        _, pred = torch.max(prob, 1)
        js_pred = js_div(prob, self._ols.matrix.detach()[pred])

        id_batch=torch.arange(0, y.size(0))
        id_noise = id_batch[js_y > threshold_clean]
        js_noise_pred = js_pred[id_noise]
        id_relabel = id_noise[js_noise_pred <= threshold_relabel]

        y[id_relabel]=pred[id_relabel].cuda()

        losses = (1 - self._args.weight) * F.cross_entropy(logits, y, reduction='none') + self._args.weight * self._ols(logits, y, reduction='none')
        js_relabel = js_div(prob, self._ols.matrix.detach()[y])
        self._js[id_global] = js_relabel

        losses = losses[js_relabel <= threshold_clean + torch.isnan(js_relabel)]
        numb_train = losses.size(0)
        if numb_train == 0:
            loss = 0
        else:
            loss = losses.mean()
        return loss, numb_train


    def train_fg(self):
        """
        Train the network
        """
        print('Training ... ')
        best_accuracy = 0.0
        best_epoch = None
        print('Epoch\tTrain Loss\tTrain Acc\tTest Acc\tEpoch Runtime')
        for t in range(self._args.epochs):
            if self._warmup > t:
                self._warmupscheduler.step()
                print('warmup learning rate',self._optimizer.state_dict()['param_groups'][0]['lr'])

            epoch_start = time.time()
            epoch_loss = []
            num_correct = 0
            num_total = 0
            batch_relabel = 0
            for X, y, id, path in self._train_loader:
                # Enable training mode
                self._net.train(True)
                # Data
                X = X.cuda()
                y = y.cuda()

                num_total += y.size(0)  # y.size(0) is the batch size
                # Forward pass
                logits = self._net(X)  # logits is in shape (N, 200)
                # Prediction
                closest_dis, prediction = torch.max(logits.data, 1)
                num_correct += torch.sum(prediction == y.data).item()

                if self._args.denoise:
                    if self._args.relabel:
                        loss, batch_train = self.js_selection_relabel(logits, y, id, t)
                    else:
                        loss, batch_train = self.js_selection(logits, y, id, t)

                else:
                    loss = (1 - self._args.weight) * self._criterion(logits, y) + self._args.weight * self._ols(logits, y)
                    batch_train = y.size(0)
                if self._args.nh > 0:
                    loss += self._args.nh * entropy_loss(logits)

                epoch_loss.append(loss.item())


                # Clear the existing gradients
                self._optimizer.zero_grad()
                # Backward
                loss.backward()
                self._optimizer.step()

            self._ols.update()
            # Record the train accuracy of each epoch
            train_accuracy = 100 * num_correct / num_total
            test_accuracy = self.test(self._test_loader)

            if self._warmup <= t+1:
                self._scheduler.step()  # the scheduler adjust lr based on test_accuracy
                # print('cos learning rate',self._optimizer.state_dict()['param_groups'][0]['lr'])

            epoch_end = time.time()

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = t + 1  # t starts from 0
                print('*', end='')
                # Save mode
                torch.save(self._net.state_dict(), os.path.join(self._path, self._args.net + 'best.pth'))
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f' % (t + 1, sum(epoch_loss) / len(epoch_loss),
                                                            train_accuracy, test_accuracy,
                                                            epoch_end - epoch_start ))

        torch.save({'ols': self._ols.matrix.cpu().data}, 'ols.pth')
        print('-----------------------------------------------------------------')
        print('Best at epoch %d, test accuracy %f' % (best_epoch, best_accuracy))
        print('-----------------------------------------------------------------')

    def test(self, dataloader):
        """
        Compute the test accuracy

        Argument:
            dataloader  Test dataloader
        Return:
            Test accuracy in percentage
        """
        self._net.train(False) # set the mode to evaluation phase
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for X, y in dataloader:
                # Data
                X = X.cuda()
                y = y.cuda()

                logits = self._net(X)
                _, prediction = torch.max(logits, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data).item()
        self._net.train(True)  # set the mode to training phase
        return 100 * num_correct / num_total

    def test_categories(self, dataloader, model =None):
        self._net.train(False) # set the mode to evaluation phase
        if model!= None:
            model_dict = torch.load(model)
            self._net.load_state_dict(model_dict)

        num_correct = torch.zeros(self._class)
        num_total = torch.zeros(self._class)

        with torch.no_grad():
            for X, y in dataloader:
                # Data
                X = X.cuda()
                y = y.cuda()

                logits = self._net(X)
                _, prediction = torch.max(logits, 1)

                for i in range(y.size(0)):
                    if prediction[i] == y[i]:
                        num_correct[y[i]] +=1
                    num_total[y[i]]+=1

        result = 100 * num_correct / num_total
        self._net.train(True)  # set the mode to training phase
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')
    parser.add_argument('--net', dest='net', type=str, default='resnet18',
                        help='supported options: resnet18, resnet50')
    parser.add_argument('--n_classes', dest='n_classes', type=int, default=200,
                        help='number of classes')
    parser.add_argument('--lr', dest='base_lr', type=float, default=1e-2)
    parser.add_argument('--w_decay', dest='weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--path', dest='path', type=str, default='model')
    parser.add_argument('--data_base', dest='data_base', type=str, default='/home/zcy/data/fg-web-data/web-bird')
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--denoise', action='store_true', default=False)
    parser.add_argument('--nh', type=float, default=0)
    parser.add_argument('--relabel', action='store_true', default=False)
    parser.add_argument('--alpha', dest='alpha',  type=float, default=0.5)
    parser.add_argument('--weight', dest='weight', type=float, default=0.5)
    parser.add_argument('--tau', dest='tau', type=float, default=0.04)
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.5)

    args = parser.parse_args()

    model = args.path

    print(os.path.join(os.popen('pwd').read().strip(), model))

    if not os.path.isdir(os.path.join(os.popen('pwd').read().strip(), model)):
        print('>>>>>> Creating directory \'model\' ... ')
        os.mkdir(os.path.join(os.popen('pwd').read().strip(), model))

    manager = Manager(args)
    manager.train_fg()