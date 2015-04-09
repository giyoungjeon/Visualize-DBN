function [ X_ ] = grad_ascent( D, rho, n )
    function [vh] = tarfun(X)
        vh = X;
        for l = 2:3
            vh = bsxfun(@plus, vh * D.rec.W{l-1}, D.rec.biases{l}');
        end        
        vh = bsxfun(@plus, vh * D.top.W, D.top.hbias');
        vh = double(vh(n));
    end

    maxiter=100;
    learn_rate = 0.01;
    delta = 14/1000;
    sample_n = 1;
    X_ = zeros(1,14*14);
    for samp_i=1:sample_n
        X0 = rand(14);
        X0 = X0 / norm(X0)*rho;
        X0 = reshape(X0, 14*14,1)';
        for iter=1:maxiter
            grad = zeros(size(X0));
            for i=1:size(X0,2)
                X = X0;
                y1 = tarfun(X);
                X(i) = X(i)+delta;
                y2 = tarfun(X);
                grad(i) = (y2-y1)/(delta);
                X0= X0 + grad*learn_rate;
                X0=X0/norm(X0)*rho;
            end
            ex_grad = grad;
        end
        X_ = X_ + X0;
    end
    X_=X_/sample_n;
end