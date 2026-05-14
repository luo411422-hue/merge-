import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


class CIFAR10():
    @staticmethod
    def get_loader(
        root=r'E:\Python\datasets\CIFAR-10',
        batch_size=128,
        test_batch_size=100,
        val_ratio=0.1,
        seed=42,
        num_workers=2,
    ):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,
                                                            0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,
                                                            0.2010)),
        ])

        base_trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=None
        )
        total_train = len(base_trainset)
        val_size = int(total_train * val_ratio)
        train_size = total_train - val_size
        if val_size <= 0 or train_size <= 0:
            raise ValueError("val_ratio creates empty split; please use a value in (0, 1)")

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_train, generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        trainset_aug = torchvision.datasets.CIFAR10(
            root=root, train=True, download=False, transform=transform_train
        )
        trainset_eval = torchvision.datasets.CIFAR10(
            root=root, train=True, download=False, transform=transform_test
        )

        trainset = Subset(trainset_aug, train_indices)
        valset = Subset(trainset_eval, val_indices)

        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test
        )

        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        valloader = DataLoader(
            valset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
        )
        testloader = DataLoader(
            testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers
        )
        return trainloader, valloader, testloader


class CIFAR100():
    @staticmethod
    def get_loader(
        root=r'E:\Python\datasets\CIFAR-100',
        batch_size=128,
        test_batch_size=100,
        val_ratio=0.1,
        seed=42,
        num_workers=4,
    ):
        transform_train = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),#反常填充
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
        ])

        base_trainset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=None
        )
        total_train = len(base_trainset)
        val_size = int(total_train * val_ratio)
        train_size = total_train - val_size
        if val_size <= 0 or train_size <= 0:
            raise ValueError("val_ratio creates empty split; please use a value in (0, 1)")

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_train, generator=generator).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        trainset_aug = torchvision.datasets.CIFAR100(
            root=root, train=True, download=False, transform=transform_train
        )
        trainset_eval = torchvision.datasets.CIFAR100(
            root=root, train=True, download=False, transform=transform_test
        )

        trainset = Subset(trainset_aug, train_indices)
        valset = Subset(trainset_eval, val_indices)

        testset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True, transform=transform_test
        )
        pin =  pin = torch.cuda.is_available()
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=pin
        )
        valloader = DataLoader(
            valset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers,pin_memory=pin
        )
        testloader = DataLoader(
            testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers,pin_memory=pin
        )
        return trainloader, valloader, testloader


class imagenet():
    def get_lodar():
        transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        transforms_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        trainset = torchvision.datasets.ImageFolder(
            root=r'E:\Python\datasets\imagenet', transform=transforms_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2,pin_memory=True)

        testset = torchvision.datasets.ImageFolder(
            root=r'E:\Python\datasets\imagenet', transform=transforms_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=2,pin_memory=True)
        return trainloader, testloader


def get_cifar100_imagenet_style_loader(
    root=r'E:\Python\datasets\CIFAR-100',
    batch_size=128,
    test_batch_size=100,
    val_ratio=0.1,
    seed=42,
    num_workers=2,
):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    base_trainset = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=None
    )
    total_train = len(base_trainset)
    val_size = int(total_train * val_ratio)
    train_size = total_train - val_size
    if val_size <= 0 or train_size <= 0:
        raise ValueError("val_ratio creates empty split; please use a value in (0, 1)")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_train, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    trainset_aug = torchvision.datasets.CIFAR100(
        root=root, train=True, download=False, transform=transform_train
    )
    trainset_eval = torchvision.datasets.CIFAR100(
        root=root, train=True, download=False, transform=transform_eval
    )
    testset = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=transform_eval
    )

    trainset = Subset(trainset_aug, train_indices)
    valset = Subset(trainset_eval, val_indices)
    pin = torch.cuda.is_available()
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin
    )
    valloader = DataLoader(
        valset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    testloader = DataLoader(
        testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )
    return trainloader, valloader, testloader


