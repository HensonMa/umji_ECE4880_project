import torch
import torch.quantization as tq
from vgg19_quantized import *
from tqdm import tqdm
from torchvision import datasets
import torchvision.transforms as transforms


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print("model Size (KB):", size/1e3)
    os.remove('temp.p')
    return size


def test():
    # quantized_vgg19 = vgg19_quantized(num_class=200, dataset="imagenet")
    # quantized_vgg19.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # quantized_vgg19 = torch.quantization.prepare(quantized_vgg19, inplace=True)
    #
    # quantized_vgg19_fusion = vgg19_quantized(num_class=200, dataset="imagenet")
    # quantized_vgg19_fusion.eval()
    # quantized_vgg19_fusion.fuse_model()
    # quantized_vgg19_fusion.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # quantized_vgg19_fusion = torch.quantization.prepare(quantized_vgg19_fusion, inplace=True)
    #
    # # Calibrate with the training set
    # input_fp32 = torch.randn(4, 3, 224, 224)
    # quantized_vgg19(input_fp32)
    # quantized_vgg19_fusion(input_fp32)
    #
    # # Convert to quantized model
    # quantized_vgg19 = torch.quantization.convert(quantized_vgg19, inplace=True)
    # quantized_vgg19_fusion = torch.quantization.convert(quantized_vgg19_fusion, inplace=True)
    #
    # print("Size of model after quantization")
    # print_size_of_model(quantized_vgg19)
    # print("Size of model after quantization and fusion")
    # print_size_of_model(quantized_vgg19_fusion)

    myvgg19 = vgg19(num_class=200, dataset="imagenet")
    print("Size of the original model")
    print_size_of_model(myvgg19)

    input_fp = torch.randn(512, 3, 224, 224)
    # start = time.time()
    # quantized_vgg19(input_fp)
    # end = time.time()
    # print("quantized model's inference time is ", end-start, "s")
    # start = time.time()
    # quantized_vgg19_fusion(input_fp)
    # end = time.time()
    # print("fused quantized model's inference time is ", end-start, "s")

    start = time.time()
    myvgg19(input_fp)
    end = time.time()
    print("original model's inference time is ", end-start, "s")


def load_model(model_file):
    model = vgg19_quantized(num_class=200, dataset="imagenet")
    model.load_state_dict(torch.load(model_file))
    model.to('cpu')
    return model


def prepare_validation_dataset(type, bs):
    if type == 'cifar10':
        img_size = 32
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif type == "imagenet":
        img_size = 224
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise RuntimeError("No such dataset")

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    if type == 'cifar10':
        test_set = datasets.CIFAR10(root="./cifar10_data", train=False, download=True, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=2)
    elif type == "imagenet":
        test_set = datasets.ImageFolder(root="../Region/data/tiny-imagenet-200" + '/val_split/', transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=2)
    else:
        raise RuntimeError("No such dataset")

    return test_set, test_loader


def validation(model, data_type, bs):
    model.eval()
    model.cpu()
    test_set, test_loader = prepare_validation_dataset(data_type, bs)

    loss_func = nn.CrossEntropyLoss(reduction="mean")
    Loss, Acc_num = 0, 0
    bt_id = 0
    inference_time = []
    for (img, lbl) in tqdm(test_loader):
        with torch.no_grad():
            bt_id += 1

            start_time = time.time()
            output = model(img)
            end_time = time.time()
            if type(output).__name__ == 'tuple':
                output = output[0]
            loss = loss_func(output, lbl)
            pre = output.max(1)[1]
            acc_num = (pre == lbl).sum().float().item()
            Acc_num += acc_num

            # Loss += loss.cpu().item()
            Loss += loss.item()
        inference_time.append(end_time - start_time)

    print("[test epoch {}] loss: %.3f, acc: %.3f, infer(cpu): %.3f s / per batch of size {}".format(0, bs) % (
    Loss / (bt_id + 1), Acc_num / len(test_set), sum(inference_time) / len(inference_time)))


def fine_tune(file_path):
    myModel = load_model(file_path)
    myModel.eval()
    # myModel.fuse_model()

    myModel.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    # myModel.conv1.qconfig = None
    # myModel.conv2.qconfig = None
    # myModel.conv3.qconfig = None
    # myModel.conv4.qconfig = None
    torch.quantization.prepare(myModel, inplace=True)

    # calibration
    validation(myModel, "imagenet", 512)

    #  Convert to quantized model
    myModel = torch.quantization.convert(myModel, inplace=True)

    print("Size of model after quantization")
    print_size_of_model(myModel)
    validation(myModel, "imagenet", 512)


# fine_tune("./pretrained_models/vgg19_imagenet/Normal_seed0_lr0.1_step_50_0.1_epoch100/model_100.bin")

