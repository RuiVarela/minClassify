import argparse
from dataloader import createCIFAR10Loaders, createFashionLoaders, createFlowers102Loaders
from model import createAlexNetLightModel, createMobileNetLargeModel, createMobileNetSmallModel, createResNet34Model, createSqueezeNetModel

from trainer import Trainer

def trainFashion(resume, load, export):
    t = Trainer()
    t.dataloader_creator = createFashionLoaders
    t.model_creator = createAlexNetLightModel

    # hyper parameters
    t.seed = 230 
    t.batch_size = pow(2, 7)
    t.epochs = 30

    t.learning_rate = 1e-2
    t.momentum = 0.9
    t.weight_decay = 1e-4
    t.step_size = 10
    t.gamma = 0.1

    t.train(resume, load, export)

def trainCIFAR10(resume, load, export):
    t = Trainer()
    t.dataloader_creator = createCIFAR10Loaders
    t.model_creator = createSqueezeNetModel


    # hyper parameters
    t.seed = 231 
    t.batch_size = pow(2, 8)
    t.epochs = 100

    t.learning_rate = 0.008 #1e-2
    t.momentum = 0.9
    t.weight_decay = 0 #1e-4
    t.step_size = 50
    t.gamma = 0.7

    t.train(resume, load, export)

def trainFlowers(resume, load, export):
    t = Trainer()

    t.dataloader_creator = createFlowers102Loaders
    t.model_creator = createSqueezeNetModel
    # t.model_creator = createMobileNetSmallModel
    # t.model_creator = createMobileNetLargeModel
    # t.model_creator = createResNet34Model

    # hyper parameters
    t.seed = 11101984
    t.epochs = 100
    t.use_adam = False

    if t.use_adam:
        t.batch_size = pow(2, 8)
        t.learning_rate = 0.00005 #1e-2
        t.weight_decay = 0
        t.step_size = 50
        t.gamma = 0.1
    else:
        t.batch_size = pow(2, 8)
        t.learning_rate = 0.005 #1e-2
        t.momentum = 0.9
        t.weight_decay = 1e-4
        t.step_size = 50
        t.gamma = 0.1

    t.train(resume, load, export)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='classifier', description='a simple classifier trainer')
    parser.add_argument('-r', '--resume', action='store_true', help="continue the last training session")
    parser.add_argument('-l', '--load', help="initialize network with specified weights")
    parser.add_argument('-e', '--export', help="export a trained model")

    args = parser.parse_args()

    #trainFlowers(args.resume, args.load, args.export)
    #trainFashion(args.resume, args.load, args.export)
    trainCIFAR10(args.resume, args.load, args.export)