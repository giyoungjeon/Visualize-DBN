% Visulization of Deep Belief Network
% Understanding Representations Learned in Deep Architectures, D. Erhan
%
% Sourced from deepmat:
%                dbn.m
%                dbn_mnist_D.mat
%
% Written by Giyoung Jeon
% Probabilistic Artificial Intelligence Lab at UNIST
% v1.0 April, 8th, 2015


load('dbn_mnist_D.mat');

rho=10;
warning('off','all');
figure;
mult = 0:10:990;
u_idx = floor(mod(rand(1,100)*100,10));
u_idx = u_idx + mult;
x_ = cell(1,100);
parfor idx = 1:100
    disp(sprintf('optimizing unit %d\n',u_idx(idx)));
    x_{idx} = grad_ascent(D,rho,u_idx(idx));
end
for idx = 1:100
    subplot(10,10,idx);
    imshow(reshape(x_{idx}, 14, 14)');
end