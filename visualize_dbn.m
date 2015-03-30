load('dbn_mnist_D.mat');


x0 = rand(14);
x0 = x0 / norm(x0)*7;
x0 = reshape(x0, 14*14,1)';

x_ = grad_ascent(D,x0);



figure;
imagesc(reshape(x0, 14,14)');
figure;
imagesc(reshape(x_, 14, 14)');


