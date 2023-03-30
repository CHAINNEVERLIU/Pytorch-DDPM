from utils.networkHelper import *


def cosine_beta_schedule(timesteps, s=0.008, **kwargs):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class VarianceSchedule(nn.Module):
    def __init__(self, schedule_name="linear_beta_schedule", beta_start=None, beta_end=None):
        super(VarianceSchedule, self).__init__()

        self.schedule_name = schedule_name

        # 定义函数字典
        beta_schedule_dict = {'linear_beta_schedule': linear_beta_schedule,
                              'cosine_beta_schedule': cosine_beta_schedule,
                              'quadratic_beta_schedule': quadratic_beta_schedule,
                              'sigmoid_beta_schedule': sigmoid_beta_schedule}

        # 根据输入字符串选择相应的函数
        if schedule_name in beta_schedule_dict:
            self.selected_schedule = beta_schedule_dict[schedule_name]
        else:
            raise ValueError('Function not found in dictionary')

        if beta_end and beta_start is None and schedule_name != "cosine_beta_schedule":
            self.beta_start = 0.0001
            self.beta_end = 0.02
        else:
            self.beta_start = beta_start
            self.beta_end = beta_end

    def forward(self, timesteps):
        # print(self.beta_start)
        return self.selected_schedule(timesteps=timesteps) if self.schedule_name == "cosine_beta_schedule" \
            else self.selected_schedule(timesteps=timesteps, beta_start=self.beta_start, beta_end=self.beta_end)



