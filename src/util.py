from model_ops.lenet import LeNet, LeNetSplit
from model_ops.resnet import *
from model_ops.resnet_split import *
from model_ops.vgg import *
from model_ops.fc_nn import FC_NN, FC_NN_Split
from model_ops.utils import err_simulation

def build_model(model_name):
    pass

def _load_data(dataset, seed):
    if seed:
        # in normal method we do not implement random seed here
        # same group should share the same shuffling result
        torch.manual_seed(seed)
        random.seed(seed)
    if dataset == "MNIST":
        training_set = datasets.MNIST('./mnist_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))]))
        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
        test_loader = None
    elif dataset == "Cifar10":
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        # data prep for training set
        # note that the key point to reach convergence performance reported in this paper (https://arxiv.org/abs/1512.03385)
        # is to implement data augmentation
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                                Variable(x.unsqueeze(0), requires_grad=False, volatile=True),
                                (4,4,4,4),mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        # data prep for test set
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        # load training and test set here:
        training_set = datasets.CIFAR10(root='./cifar10_data', train=True,
                                                download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                                  shuffle=True)
        testset = datasets.CIFAR10(root='./cifar10_data', train=False,
                                               download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                                 shuffle=False)
    return train_loader, training_set, test_loader


def group_assign(world_size, group_size, rank):
    if world_size % group_size == 0:
        ret_group_dict, group_list = _assign(world_size, group_size, rank)  
    else:
        ret_group_dict, group_list = _assign(world_size-1, group_size, rank)
        group_list[-1].append(world_size)
    group_num, group_seeds = _group_identify(group_list, rank)
    return ret_group_dict, group_num, group_seeds
    

def _group_identify(group_list, rank):
    group_seeds = [0]*len(group_list)
    if rank == 0:
        return -1, group_seeds
    for i,group in enumerate(group_list):
        group_seeds[i] = np.random.randint(0, 20000)
        if rank in group:
            group_num = i
    return group_num, group_seeds


def _assign(world_size, group_size, rank):
    np.random.seed(SEED_)
    ret_group_dict={}
    k = world_size/group_size
    group_list=[[j+i*group_size+1 for j in range(group_size)] for i in range(k)]
    for i, l in enumerate(group_list):
        ret_group_dict[i]=l
    return ret_group_dict, group_list