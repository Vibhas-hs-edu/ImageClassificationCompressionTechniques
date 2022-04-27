import os

class BaseModel:
    def __init__(self, name, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_dir = save_dir
        self.name = name

    def save_model(self, model, epoch):
        model.save(f'{self.save_dir}/{self.name}_{epoch}.h5')