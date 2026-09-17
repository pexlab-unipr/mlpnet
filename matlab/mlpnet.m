classdef mlpnet < handle % Multilayer perceptron (MLP) neural network

    properties
        size; % layer sizes, including input and output layers
        nh; % number of hidden layers
        eta = 1e-3; % learning rate
        f = @(x) tanh(x); % activation function
        df = @(x) 1 - tanh(x).^2; % derivative of the activation function
        X; % inputs of all layers
        Y; % outputs of all layers
        W; % weights
        B; % biases
        dW; % gradient of the loss function w.r.t. the weights
        dB; % gradient of the loss function w.r.t. the biases
    end

    methods
        function net = mlpnet(net_size)
            % Object constructor
            net.size = net_size;
            net.nh = length(net_size) - 2;
            net.X = cell(1,net.nh+1);
            net.Y = cell(1,net.nh+1);
            net.W = cell(1,net.nh+1);
            net.B = cell(1,net.nh+1);
            net.dW = cell(1,net.nh+1);
            net.dB = cell(1,net.nh+1);
            for k = 1:net.nh+1
                net.X{k} = zeros(1,net.size(k));
                net.Y{k} = zeros(1,net.size(k+1));
                net.W{k} = randn(net.size(k),net.size(k+1))/net.size(k);
                net.B{k} = zeros(1,net.size(k+1));
                net.dW{k} = zeros(net.size(k),net.size(k+1));
                net.dB{k} = zeros(1,net.size(k+1));
            end
        end

        function y = eval(net,x)
            % Evaluate the network at x
            net.X{1} = x(:).';
            for k = 1:net.nh+1
                net.Y{k} = net.X{k}*net.W{k} + net.B{k};
                if k < net.nh+1
                    net.X{k+1} = net.f(net.Y{k});
                end
            end
            y = net.Y{end};
        end

        function J = jacobian(net,x)
            % Compute the Jacobian matrix at x
            J = zeros(net.size(end),net.size(1));
            net.eval(x);
            for out = 1:net.size(end)
                t = (1:net.size(end)) == out;
                for k = net.nh+1:-1:2
                    t = (t*net.W{k}.').*net.df(net.Y{k-1});
                end
                J(out,:) = t*net.W{1}.';
            end
        end

        function loss = update(net,x,y)
            % Compute the gradient of the loss function w.r.t. the network
            % parameters by backpropagation at (x, y), then update the
            % parameters by stochastic gradient descent
            t = net.eval(x) - y(:).';
            loss = 0.5*(t*t');
            for k = net.nh+1:-1:1
                net.dW{k} = t.*net.X{k}.';
                net.dB{k} = t;
                if k > 1
                    t = (t*net.W{k}.').*net.df(net.Y{k-1});
                end
            end
            for k = 1:net.nh+1
                net.W{k} = net.W{k} - net.eta*net.dW{k};
                net.B{k} = net.B{k} - net.eta*net.dB{k};
            end
        end
    end
end