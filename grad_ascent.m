function [ X_ ] = grad_ascent( D, rho, n )
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

    maxiter=100;
    learn_rate = 0.01;
    delta = 14/1000;
    %delta = 0.0001;
    X_ = zeros(1,14*14);
    for i=1:100
        X0 = rand(14);
        X0 = X0 / norm(X0)*rho;
        X0 = reshape(X0, 14*14,1)';
        for iter=1:maxiter
            grad = zeros(size(X0));
            if  mod(iter,10)==0
%                 disp(sprintf('%dth iter', iter));
            end
            for i=1:size(X0,2)
                X = X0; X = X0;
                X(i) = X(i)-delta;
                y1 = tarfun(X);%/norm(X1)*rho);
                X(i) = X(i)+delta;
                y2 = tarfun(X);%/norm(X2)*rho);
                grad(i) = (y2-y1)/(2*delta);
                X0= X0 + grad*learn_rate;
                X0=X0/norm(X0)*rho;
            end
        end
        X_ = X_ + X0;
    end
    X_=X_/100;
end