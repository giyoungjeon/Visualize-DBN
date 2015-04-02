load('dbn_mnist_D.mat');

rho=10;
x0 = rand(14);
x0 = x0 / norm(x0)*rho;
x0 = reshape(x0, 14*14,1)';
x_ = grad_ascent2(D,x0,rho);


% figure;
% imshow(reshape(x0, 14,14)');
figure;
imshow(reshape(x_, 14, 14)');