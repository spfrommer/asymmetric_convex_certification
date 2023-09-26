mkdir ../../pretrain
mkdir ../../pretrain/mnist_38
mkdir ../../pretrain/fashion_mnist_shirts

bash command/mnist_38.sh
mv result/*/model.pth ../../pretrain/mnist_38/model.pth
bash command/fashion_mnist_shirts.sh
mv result/*/model.pth ../../pretrain/fashion_mnist_shirts/model.pth
