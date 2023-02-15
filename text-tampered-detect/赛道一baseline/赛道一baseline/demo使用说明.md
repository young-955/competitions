ICDAR 2023挑战赛baseline
运行demo文件即可调用预训练模型
网络模型：(timm==0.4.9)
model = timm.create_model('tf_efficientnet_b0', pretrained=True, num_classes=2)