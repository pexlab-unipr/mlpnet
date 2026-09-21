clear
close all
clc

%% Generate the dataset
xt = linspace(-5,5,25);
yt = 5*sin(xt) + randn(size(xt));

%% Create the network
net = mlpnet([1,8,7,6,1]);
switch 0
    case 1 % ReLU
        net.f = @(x) max(0,x);
        net.df = @(x) x>0;
    case 2 % Softplus
        net.f = @(x) log(1+exp(x));
        net.df = @(x) 1./(1+exp(-x));
    case 3 % Softsign
        net.f = @(x) x./(1+abs(x));
        net.df = @(x) 1./(1+abs(x)).^2;
    case 4 % Sinusoid
        net.f = @(x) sin(x);
        net.df = @(x) cos(x);
    case 5 % SiLU
        net.f = @(x) x./(1+exp(-x));
        net.df = @(x) 1./(exp(-x) + 1) + (x.*exp(-x))./(exp(-x) + 1).^2;
    case 6 % Gaussian
        net.f = @(x) exp(-x.^2);
        net.df = @(x) -2*x.*exp(-x.^2);
end

%% Train the network
for epoch = 1:5000
    if ~mod(epoch,250)
        fprintf("Epoch %d\n",epoch);
    end
    for k = 1:length(xt)
        net.update(xt(k),yt(k));
    end
end

%% Evaluate the network
xg = linspace(min(xt),max(xt),1001);
yg = zeros(size(xg));
grad_net = zeros(size(xg));
for k = 1:numel(xg)
    grad_net(k) = net.jacobian(xg(k));
    yg(k) = net.Y{end};
end
grad_num = gradient(yg,xg(2)-xg(1));

%% Plot
figure
subplot(2,1,1)
plot(xt,yt,'*')
grid on
hold on
plot(xg,yg,'--')
xlabel('x')
ylabel('y')
ax(1) = gca;
legend('Data','Fitting')
subplot(2,1,2)
plot(xg,grad_num,'-.')
grid on
hold on
plot(xg,grad_net,'--')
xlabel('x')
ylabel('dy/dx')
ax(2) = gca;
legend('Numerical','Network')
linkaxes(ax,'x')