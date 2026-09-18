classdef mlpnet < handle
    % Multilayer perceptron (MLP) neural network

    properties
        size; % layer sizes, including input and output layers
        nh; % number of hidden layers
        eta = 1e-3; % learning rate
        f = @(x) tanh(x); % activation function
        df = @(x) 1 - tanh(x).^2; % derivative of the activation function
        Y; % outputs for each layer
        W; % weights
        B; % biases
        dW; % gradient of the loss function w.r.t. the weights
        dB; % gradient of the loss function w.r.t. the biases
    end

    methods
        function net = mlpnet(net_size)
            % Object constructor.
            net.size = net_size;
            net.nh = length(net_size) - 2;
            net.Y = cell(1,net.nh+1);
            net.W = cell(1,net.nh+1);
            net.B = cell(1,net.nh+1);
            net.dW = cell(1,net.nh+1);
            net.dB = cell(1,net.nh+1);
            for k = 1:net.nh+1
                net.Y{k} = zeros(1,net.size(k+1));
                net.W{k} = randn(net.size(k),net.size(k+1))/net.size(k);
                net.B{k} = zeros(1,net.size(k+1));
                net.dW{k} = zeros(net.size(k),net.size(k+1));
                net.dB{k} = zeros(1,net.size(k+1));
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
                net.dB{k} = t;
                net.dW{k} = t.*net.f(net.Y{k-1}).';
                t = (t*net.W{k}.').*net.df(net.Y{k-1});
            end
            net.dB{1} = t;
            net.dW{1} = t.*x(:);
            for k = 1:net.nh+1
                net.W{k} = net.W{k} - net.eta*net.dW{k};
                net.B{k} = net.B{k} - net.eta*net.dB{k};
            end
        end
        function J = jacobian(net,x)
            % Return the Jacobian matrix computed at x. The function also
            % updates Y by evaluating the network at x.
            if net.nh > 0
                net.eval(x);
                J = zeros(net.size(end),net.size(1));
                for o = 1:net.size(end)
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
    end
end