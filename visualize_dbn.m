load('dbn_mnist_D.mat');

rho=10;
warning('off','all');
figure;
for i=1:100
    disp(sprintf('optimizing unit %d\n',i));
    x_ = grad_ascent(D,rho,i);
    subplot(10,10,i);
    imshow(reshape(x_, 14, 14)');
%     title(sprintf('unit %d\n',i));
end