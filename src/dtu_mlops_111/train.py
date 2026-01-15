from dtu_mlops_111.data import MyDataset
from dtu_mlops_111.model import Model


def train():
    dataset = MyDataset("data/raw") # noqa
    model = Model() # noqa
    # add rest of your training code here

if __name__ == "__main__":
    train()
