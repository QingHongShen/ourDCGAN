import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import time as t
import os

from utils.ThreeZeroEight import class_directories
from utils.tensorboard_logger import Logger

from utils.inception_score import get_inception_score
from itertools import chain
from torchvision import utils
from torch.nn import functional as F

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)

class Generator(torch.nn.Module):
    def __init__(self, channels):
        super(Generator, self).__init__()
        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            PixelNorm(),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            PixelNorm(),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),  # 新增层
            PixelNorm(),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1))



        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.main_module = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1),  # 新增层
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True))

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)

class DCGAN_MODEL(object):
    def __init__(self, args):
        print("DCGAN model initalization.")
        self.G = Generator(args.channels==3)
        self.D = Discriminator(args.channels==3)
        self.C = args.channels

        # binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()

        self.cuda = False
        self.cuda_index = 0
        # check if cuda is available
        self.check_cuda(args.cuda)

        # Using lower learning rate than suggested by (ADAM authors) lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0003, betas=(0.5, 0.999))

        self.epochs = args.epochs
        self.batch_size = args.batch_size

        # Set the logger
        self.logger = Logger('./logs')
        self.number_of_images = 10

    # cuda support
    def check_cuda(self, cuda_flag=False):
        if cuda_flag:
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            self.loss = nn.BCELoss().cuda(self.cuda_index)
            print("Cuda enabled flag: ")
            print(self.cuda)


    def train(self, train_loader):
        self.t_begin = t.time()
        generator_iter = 0
        #self.file = open("inception_score_graph.txt", "w")

        for epoch in range(self.epochs):
            self.epoch_start_time = t.time()

            for i, (images, _) in enumerate(train_loader):
                # Check if round number of batches
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break

                z = torch.rand((self.batch_size, 100, 1, 1))
                real_labels = torch.ones(self.batch_size)
                fake_labels = torch.zeros(self.batch_size)

                if self.cuda:
                    images, z = Variable(images).cuda(self.cuda_index), Variable(z).cuda(self.cuda_index)
                    real_labels, fake_labels = Variable(real_labels).cuda(self.cuda_index), Variable(fake_labels).cuda(self.cuda_index)
                else:
                    images, z = Variable(images), Variable(z)
                    real_labels, fake_labels = Variable(real_labels), Variable(fake_labels)


                # Train discriminator
                # Compute BCE_Loss using real images
                if i % 5 == 0:
                    outputs = self.D(images)
                    d_loss_real = self.loss(outputs.flatten(), real_labels)
                    real_score = outputs

                    # Compute BCE Loss using fake images
                    if self.cuda:
                        z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
                    else:
                        z = Variable(torch.randn(self.batch_size, 100, 1, 1))
                    fake_images = self.G(z)
                    outputs = self.D(fake_images)
                    d_loss_fake = self.loss(outputs.flatten(), fake_labels)
                    fake_score = outputs

                    # Optimize discriminator
                    d_loss = d_loss_real + d_loss_fake
                    self.D.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                # Train generator
                # Compute loss with fake images
                if self.cuda:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
                else:
                    z = Variable(torch.randn(self.batch_size, 100, 1, 1))
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                g_loss = self.loss(outputs.flatten(), real_labels)

                # Optimize generator
                self.D.zero_grad()
                self.G.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                generator_iter += 1


                if generator_iter % 1000 == 0:
                    # 确保生成的目录路径正确无误
                    base_save_dir = '/root/autodl-tmp/ourDCGAN/308_generate_images'
                    if not os.path.exists(base_save_dir):
                        os.makedirs(base_save_dir)

                    for class_name, _ in class_directories.items():

                        # 对于每个类别，生成图片并保存
                        z_class = Variable(torch.randn(self.batch_size, 100, 1, 1))  # 假设不对z进行类别条件化，仅随机生成
                        if self.cuda:
                            z_class = z_class.cuda(self.cuda_index)
                        samples_class = self.G(z_class)
                        samples_class = samples_class.mul(0.5).add(0.5)  # 反归一化

                        # 创建并保存图片
                        class_save_dir = os.path.join(base_save_dir, class_name)
                        if not os.path.exists(class_save_dir):
                            os.makedirs(class_save_dir)

                        for idx, img in enumerate(samples_class.data.cpu()[:240]):  # 保存最多120张图片
                            img_path = os.path.join(class_save_dir, f'{class_name}_{generator_iter}_{idx:04d}.png')
                            torchvision.utils.save_image(img, img_path, normalize=True)
                    #
                    print('Epoch-{}'.format(epoch + 1))
                    #self.save_model()

                    if not os.path.exists('training_result_images/'):
                        os.makedirs('training_result_images/')

                    # Denormalize images and save them in grid 8x8
                    z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                    samples = self.G(z)
                    samples = samples.mul(0.5).add(0.5)
                    samples = samples.data.cpu()[:64]
                    grid = utils.make_grid(samples)
                    utils.save_image(grid, 'training_result_images/img_generatori_iter_{}.png'.format(str(generator_iter).zfill(3)))

                    time = t.time() - self.t_begin
                    #print("Inception score: {}".format(inception_score))
                    print("Generator iter: {}".format(generator_iter))
                    print("Time {}".format(time))

                    # Write to file inception_score, gen_iters, time
                    #output = str(generator_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
                    #self.file.write(output)


                #if ((i + 1) % 100) == 0:
                if ((i + 1) % 1) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (i + 1), train_loader.dataset.__len__() // self.batch_size, d_loss.data, g_loss.data))

                    z = Variable(torch.randn(self.batch_size, 100, 1, 1).cuda(self.cuda_index))

                    # TensorBoard logging
                    # Log the scalar values
                    info = {
                        'd_loss': d_loss.data,
                        'g_loss': g_loss.data
                    }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, generator_iter)

                    # Log values and gradients of the parameters
                    for tag, value in self.D.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, self.to_np(value), generator_iter)
                        self.logger.histo_summary(tag + '/grad', self.to_np(value.grad), generator_iter)

                    # Log the images while training
                    info = {
                        'real_images': self.real_images(images, self.number_of_images),
                        'generated_images': self.generate_img(z, self.number_of_images)
                    }

                    for tag, images in info.items():
                        self.logger.image_summary(tag, images, generator_iter)

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        #self.file.close()

        # Save the trained parameters
        #self.save_model()

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 64, 64)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 64, 64)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 64, 64))
            else:
                generated_images.append(sample.reshape(3, 64, 64))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        # Interpolate between twe noise(z1, z2) with number_int steps between
        number_int = 10
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(3, 32, 32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images to interpolated_images/interpolated_{}.".format(str(number).zfill(3)))
