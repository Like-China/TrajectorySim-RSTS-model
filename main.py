from trainer.train import train
from settings import set_args

if __name__ == "__main__":
    # porto读取需要半小时，beijing读取需要4分钟
    print("train111")
    train(set_args())
