import torch
import numpy as np

# 该文件中封装了一个学习率优化类，其可以实现学习率动态变化，结合了退火处理


# 为学习率更新封装的类
class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config, model_config, current_step):

        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=train_config["optimizer"]["betas"], # [0.9, 0.98]
            eps=train_config["optimizer"]["eps"], # 0.000000001
            weight_decay=train_config["optimizer"]["weight_decay"], # 0.0
        )
        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"] # warmup的步数 4000
        self.anneal_steps = train_config["optimizer"]["anneal_steps"] # 退火步数 [300000, 400000, 500000]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"] # 退火率 0.3
        self.current_step = current_step # 训练时的当前步数
        self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5) # 初始学习率

    # 使用设置的学习率方案进行参数更新
    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    # 清楚梯度
    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    # 加载保存的优化器参数
    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    # 学习率变化规则
    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:  # 如果当前训练步数大于设置的回火步数，进一步对学习率进行设置
                lr = lr * self.anneal_rate
        return lr

    # 该学习方案中每步的学习率
    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale() # 计算当前步数的学习率
        # 给所有参数设置学习率
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
