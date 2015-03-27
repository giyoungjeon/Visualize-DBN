load('example\dbn_mnist_D.mat');

x0 = rand(14);
x0 = x0 / norm(x0);
x0 = reshape(x0, 14*14,1)';

h0 = cell(3,1);
p0 = cell(3,1);
h0{1} = x0;
for i=2:size(h0,1)
    p0{i} = sigmoid(bsxfun(@plus, h0{i-1} * D.rec.W{i-1}, D.rec.biases{i}'));
    h0{i} = binornd(1, p0{i});
end
p_top = sigmoid(bsxfun(@plus, h0{3} * D.rec.W{3}, D.top.hbias'));
top = binornd(1, p_top);

stem(p_top);