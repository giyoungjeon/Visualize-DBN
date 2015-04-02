function [ X_ ] = grad_ascent( D, X )
%GRAD_ASCENT Summary of this function goes here
%   Detailed explanation goes here
    function [vh] = tarfun(X)
        vh = X;
        for l = 2:3
            vh = sigmoid(bsxfun(@plus, vh * D.rec.W{l-1}, D.rec.biases{l}'));
            vh = binornd(1, vh);
        end        
        vh = sigmoid(bsxfun(@plus, vh * D.top.W, D.top.hbias'));
        vh = vh(1);
    end

    maxiter=100;
    learn_rate = 0.0001;
    delta = 1e-12;
    %delta = 0.0001;
    grad = zeros(size(X));    
    for iter=1:maxiter
        if  mod(iter,10)==0
            disp(sprintf('%dth iter', iter));
        end
        for i=1:size(X,2)
            X1 = X; X2 = X;
            X1(i) = X1(i)-delta;
            X2(i) = X2(i)+delta;
            
            y1 = tarfun(X1/norm(X1)*7);
            y2 = tarfun(X2/norm(X2)*7);
            grad(i) = (y2-y1)/(2*delta);
            X= X + grad*learn_rate;
            X=X/norm(X)*7;
        end
    end
    X_=X;
end