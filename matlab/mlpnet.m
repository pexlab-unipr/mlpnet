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
        itermax = 1000; % maximum number of iterations.
        tol = 1e-2; % relative termination tolerance
        rho = 1e-3; % initial damping factor
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
        function [iter,resnorm] = train(net,X,Y)
            % Train the MLP neural network by Levenberg-Marquardt
            % algorithm. X and Y are the array of training inputs and
            % outputs, respectively: each row must correspond to one
            % observation. The function returns the number of iterations
            % taken and the norm of the residual vector.
            m = net.sz(end);
            n = size(Y,1);
            np = dot(net.sz(2:end), 1 + net.sz(1:end-1));
            resnorm = 0;
            B_ = net.B;
            W_ = net.W;
            rho_d = net.rho;
            r = zeros(m,n);
            J = zeros(m*n,np);
            for iter = 1:net.itermax
                if net.mu > 1 && iter > 1
                    r_ = r;
                    J_ = J;
                end
                resnorm_ = resnorm;
                row = 0;
                for k = 1:n
                    r(:,k) = (Y(k,:) - net.eval(X(k,:))).';
                    for o = 1:m
                        row = row + 1;
                        col = 0;
                        t = (0:m-1) == mod(row-1,m);
                        for p = net.nh+1:-1:2
                            col = col(end)+1:col(end)+numel(net.B{p});
                            J(row,col) = t;
                            col = col(end)+1:col(end)+numel(net.W{p});
                            tmp = t.*net.f(net.Y{p-1}).';
                            J(row,col) = tmp(:).';
                            t = (t*net.W{p}.').*net.df(net.Y{p-1});
                        end
                        col = col(end)+1:col(end)+numel(net.B{1});
                        J(row,col) = t;
                        col = col(end)+1:col(end)+numel(net.W{1});
                        tmp = t.*X(k,:).';
                        J(row,col) = tmp(:).';
                    end
                end
                resnorm = norm(r(:));
                if net.mu > 1 && iter > 1
                    if resnorm > resnorm_
                        rho_d = rho_d*net.mu*net.mu;
                        for k = 1:net.nh+1
                            net.B{k} = B_{k};
                            net.W{k} = W_{k};
                            resnorm = resnorm_;
                            r = r_;
                            J = J_;
                        end
                    else
                        rho_d = rho_d/net.mu;
                    end
                end
                if norm(J'*r(:)) < net.tol*(1 + norm(J(:)))
                    break
                end
                A = [J; sqrt(rho_d)*eye(np)];
                b = [r(:); zeros(np,1)];
                dp = A\b;
                row = 0;
                for k = net.nh+1:-1:1
                    B_{k} = net.B{k};
                    W_{k} = net.W{k};
                    row = row(end)+1:row(end)+numel(net.B{k});
                    net.B{k} = net.B{k} + reshape(dp(row),size(net.B{k}));
                    row = row(end)+1:row(end)+numel(net.W{k});
                    net.W{k} = net.W{k} + reshape(dp(row),size(net.W{k}));
                end
            end
        end
    end
end