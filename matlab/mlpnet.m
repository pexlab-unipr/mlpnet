classdef mlpnet < handle
    % Multilayer perceptron (MLP) neural network

    properties
        sz; % layer sizes, including input and output layers
        nh; % number of hidden layers
        eta = 1e-3; % learning rate for stochastic gradient descent
        f = @(x) tanh(x); % activation function
        df = @(x) 1 - tanh(x).^2; % derivative of the activation function
        Y; % outputs for each layer
        B; % biases
        W; % weights
        gW; % gradient of the loss function w.r.t. the weights
        gB; % gradient of the loss function w.r.t. the biases
        % Parameters for Levenberg-Marquardt training algorithm
        itmax = 1000; % maximum number of iterations.
        tol = 1e-2; % relative termination tolerance
        rho = 1; % initial damping factor
        mu = 2; % damping adjustment factor
    end

    methods
        function net = mlpnet(sz)
            % Create an MLP neural network with the layer sizes specified
            % in the array sz = [n_in, n_1, n_2, ..., n_nh, n_out], where
            % n_in is the number of inputs, n_out is the number of outputs
            % and n_i is the number of neurons in the i-th hidden layer,
            % where 1 <= i <= nh.
            net.sz = sz;
            net.nh = length(sz) - 2;
            net.Y = cell(1,net.nh+1);
            net.B = cell(1,net.nh+1);
            net.W = cell(1,net.nh+1);
            net.gB = cell(1,net.nh+1);
            net.gW = cell(1,net.nh+1);
            for k = 1:net.nh+1
                net.Y{k} = zeros(1,net.sz(k+1));
                net.B{k} = zeros(1,net.sz(k+1));
                net.W{k} = randn(net.sz(k),net.sz(k+1))/net.sz(k);
                net.gB{k} = zeros(1,net.sz(k+1));
                net.gW{k} = zeros(net.sz(k),net.sz(k+1));
            end
        end
        function y = eval(net,x)
            % Evaluate the network at x and update Y. Return the network
            % output.
            net.Y{1} = x(:).'*net.W{1} + net.B{1};
            for k = 2:net.nh+1
                net.Y{k} = net.f(net.Y{k-1})*net.W{k} + net.B{k};
            end
            y = net.Y{end};
        end
        function loss = update(net,x,y)
            % Compute the gradient of the loss function w.r.t. the network
            % parameters by backpropagation at (x, y), then update W and B
            % by stochastic gradient descent. Return the loss value
            % computed before the update. The function also updates Y by
            % evaluating the network at x.
            t = net.eval(x) - y(:).';
            loss = 0.5*(t*t');
            for k = net.nh+1:-1:2
                net.gB{k} = t;
                net.gW{k} = t.*net.f(net.Y{k-1}).';
                t = (t*net.W{k}.').*net.df(net.Y{k-1});
            end
            net.gB{1} = t;
            net.gW{1} = t.*x(:);
            for k = 1:net.nh+1
                net.B{k} = net.B{k} - net.eta*net.gB{k};
                net.W{k} = net.W{k} - net.eta*net.gW{k};
            end
        end
        function J = jacobian(net,x)
            % Return the Jacobian matrix computed at x. The function also
            % updates Y by evaluating the network at x.
            if net.nh > 0
                net.eval(x);
                J = zeros(net.sz(end),net.sz(1));
                for o = 1:net.sz(end)
                    t = (net.W{net.nh+1}(:,o).').*net.df(net.Y{net.nh});
                    for k = net.nh:-1:2
                        t = (t*net.W{k}.').*net.df(net.Y{k-1});
                    end
                    J(o,:) = t*net.W{1}.';
                end
            else
                J = net.W{1}.';
            end
        end
        function [it,e_norm] = train(net,X,Y)
            % Train the MLP neural network by Levenberg-Marquardt
            % algorithm. X and Y are the array of training inputs and
            % outputs, respectively: each row must correspond to one
            % observation. The function returns the number of iterations
            % taken and the norm of the error vector.
            m = net.sz(end);
            n = size(Y,1);
            np = sum(net.sz(2:end));
            for k = 1:net.nh+1
                np = np + net.sz(k)*net.sz(k+1);
            end
            e_norm = 0;
            B_ = net.B;
            W_ = net.W;
            e = zeros(m,n);
            J = zeros(m*n,np);
            rho0 = net.rho;
            for it = 1:net.itmax
                if net.mu > 1 && it > 1
                    e_ = e;
                    J_ = J;
                end
                e_norm_ = e_norm;
                for k = 1:n
                    e(:,k) = (Y(k,:) - net.eval(X(k,:))).';
                    Jk = zeros(m,np);
                    for o = 1:m
                        idx0 = 1;
                        t = (1:m) == o;
                        for p = net.nh+1:-1:2
                            idx = idx0:idx0+numel(net.B{p})-1;
                            Jk(o,idx) = t;
                            idx0 = idx(end)+1;
                            idx = idx0:idx0+numel(net.W{p})-1;
                            tmp = t.*net.f(net.Y{p-1}).';
                            Jk(o,idx) = tmp(:).';
                            t = (t*net.W{p}.').*net.df(net.Y{p-1});
                            idx0 = idx(end)+1;
                        end
                        idx = idx0:idx0+numel(net.B{1})-1;
                        Jk(o,idx) = t;
                        idx0 = idx(end)+1;
                        idx = idx0:idx0+numel(net.W{1})-1;
                        tmp = t.*X(k,:).';
                        Jk(o,idx) = tmp(:).';
                    end
                    J((1:m)+(k-1)*m,:) = Jk;
                end
                e_norm = norm(e(:));
                if net.mu > 1 && it > 1
                    if e_norm > e_norm_
                        rho0 = rho0*net.mu*net.mu;
                        for k = 1:net.nh+1
                            net.B{k} = B_{k};
                            net.W{k} = W_{k};
                            e_norm = e_norm_;
                            e = e_;
                            J = J_;
                        end
                    else
                        rho0 = rho0/net.mu;
                    end
                end
                if norm(J.'*e(:)) < net.tol*(1 + norm(J(:)))
                    break
                end
                A = [J; sqrt(rho0)*eye(np)];
                b = [e(:); zeros(np,1)];
                dp = A\b;
                idx0 = 1;
                for k = net.nh+1:-1:1
                    B_{k} = net.B{k};
                    W_{k} = net.W{k};
                    idx = idx0:idx0+numel(net.B{k})-1;
                    net.B{k} = net.B{k} + reshape(dp(idx),size(net.B{k}));
                    idx0 = idx(end)+1;
                    idx = idx0:idx0+numel(net.W{k})-1;
                    net.W{k} = net.W{k} + reshape(dp(idx),size(net.W{k}));
                    idx0 = idx(end)+1;
                end
            end
        end
    end
end