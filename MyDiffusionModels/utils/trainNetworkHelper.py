import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoints.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TrainerBase(nn.Module):
    def __init__(self,
                 epoches,
                 train_loader,
                 optimizer,
                 device,
                 IFEarlyStopping,
                 IFadjust_learning_rate,
                 **kwargs):
        super(TrainerBase, self).__init__()

        self.epoches = epoches
        if self.epoches is None:
            raise ValueError("请传入训练总迭代次数")

        self.train_loader = train_loader
        if self.train_loader is None:
            raise ValueError("请传入train_loader")

        self.optimizer = optimizer
        if self.optimizer is None:
            raise ValueError("请传入优化器类")

        self.device = device
        if self.device is None:
            raise ValueError("请传入运行设备类型")

        # 如果启用了提前停止策略则必须进行下面一系列判断
        self.IFEarlyStopping = IFEarlyStopping
        if IFEarlyStopping:
            if "patience" in kwargs.keys():
                self.early_stopping = EarlyStopping(patience=kwargs["patience"], verbose=True)
            else:
                raise ValueError("启用提前停止策略必须输入{patience=int X}参数")

            if "val_loader" in kwargs.keys():
                self.val_loader = kwargs["val_loader"]
            else:
                raise ValueError("启用提前停止策略必须输入验证集val_loader")

        # 如果启用了学习率调整策略则必须进行下面一系列判断
        self.IFadjust_learning_rate = IFadjust_learning_rate
        if IFadjust_learning_rate:
            if "types" in kwargs.keys():
                self.types = kwargs["types"]
                if "lr_adjust" in kwargs.keys():
                    self.lr_adjust = kwargs["lr_adjust"]
                else:
                    self.lr_adjust = None
            else:
                raise ValueError("启用学习率调整策略必须从{type1 or type2}中选择学习率调整策略参数types")

    def adjust_learning_rate(self, epoch, learning_rate):
        # lr = args.learning_rate * (0.2 ** (epoch // 2))
        if self.types == 'type1':
            lr_adjust = {epoch: learning_rate * (0.1 ** ((epoch - 1) // 10))}  # 每10个epoch,学习率缩小10倍
        elif self.types == 'type2':
            if self.lr_adjust is not None:
                lr_adjust = self.lr_adjust
            else:
                lr_adjust = {
                    5: 1e-4, 10: 5e-5, 20: 1e-5, 25: 5e-6,
                    30: 1e-6, 35: 5e-7, 40: 1e-8
                }
        else:
            raise ValueError("请从{{0}or{1}}中选择学习率调整策略参数types".format("type1", "type2"))

        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

    @staticmethod
    def save_best_model(model, path):
        torch.save(model.state_dict(), path+'/'+'BestModel.pth')
        print("成功将此次训练模型存储(储存格式为.pth)至:" + str(path))

    def forward(self, model, *args, **kwargs):

        pass


class SimpleDiffusionTrainer(TrainerBase):
    def __init__(self,
                 epoches=None,
                 train_loader=None,
                 optimizer=None,
                 device=None,
                 IFEarlyStopping=False,
                 IFadjust_learning_rate=False,
                 **kwargs):
        super(SimpleDiffusionTrainer, self).__init__(epoches, train_loader, optimizer, device,
                                                     IFEarlyStopping, IFadjust_learning_rate,
                                                     **kwargs)

        if "timesteps" in kwargs.keys():
            self.timesteps = kwargs["timesteps"]
        else:
            raise ValueError("扩散模型训练必须提供扩散步数参数")

    def forward(self, model, *args, **kwargs):

        for i in range(self.epoches):
            losses = []
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for step, (features, labels) in loop:
                features = features.to(self.device)
                batch_size = features.shape[0]

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

                loss = model(mode="train", x_start=features, t=t, loss_type="huber")
                losses.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 更新信息
                loop.set_description(f'Epoch [{i}/{self.epoches}]')
                loop.set_postfix(loss=loss.item())

        if "model_save_path" in kwargs.keys():
            self.save_best_model(model=model, path=kwargs["model_save_path"])

        return model



