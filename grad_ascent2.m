function [ X_ ] = grad_ascent2( D, rho, n)
%GRAD_ASCENT Summary of this function goes here
%   Detailed explanation goes here
    function [vh] = tarfun(X)
        vh = X;
        for l = 2:3
            vh = bsxfun(@plus, vh * D.rec.W{l-1}, D.rec.biases{l}');
%             vh = binornd(1, vh);
        end        
        vh = bsxfun(@plus, vh * D.top.W, D.top.hbias');
        vh = double(vh(n));
    end
    function [c ceq] = mycon(X)
        c = [];
        ceq = norm(X)-rho;
    end

    X_ = zeros(1,14*14);
    for i=1:100
        X = rand(14);
        X = X / norm(X)*rho;
        X = reshape(X, 14*14,1)';
        X_ = X_ + fmincon(@tarfun, X, [],[],[],[],[],[], @mycon);
    end
    X_ = X_/100;
%     W{1} = [[D.rec.W{1};D.rec.biases{2}'],[zeros(size(D.rec.W{1},1),1);1]];
%     W{2} = [[D.rec.W{2};D.rec.biases{3}'],[zeros(size(D.rec.W{2},1),1);1]];
%     W{3} = [[D.rec.W{3};D.top.hbias'],[zeros(size(D.rec.W{3},1),1);1]];
%     W_ = W{1}*W{2}*W{3};
%     W_ = W_(1:end-1,100);
%     X_ = norm(X)*W_/norm(W_);
    


end