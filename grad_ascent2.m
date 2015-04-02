function [ X_ ] = grad_ascent2( D, X, rho)
%GRAD_ASCENT Summary of this function goes here
%   Detailed explanation goes here
    function [o] = tarfun(X)
        o = -bsxfun(@plus, X * D.rec.W{1}, D.rec.biases{2}');
        o = double(o(1));
    end
    function [c ceq] = mycon(X)
        c = [];
        ceq = norm(X)-rho;
    end

%     X_ = fmincon(@tarfun, X, [],[],[],[],[],[], @mycon);
    W{1} = [[D.rec.W{1};D.rec.biases{2}'],[zeros(size(D.rec.W{1},1),1);1]];
    W{2} = [[D.rec.W{2};D.rec.biases{3}'],[zeros(size(D.rec.W{2},1),1);1]];
    W{3} = [[D.rec.W{3};D.top.hbias'],[zeros(size(D.rec.W{3},1),1);1]];
    W_ = W{1}*W{2}*W{3};
    W_ = W_(1:end-1,100);
    X_ = norm(X)*W_/norm(W_);
    


end