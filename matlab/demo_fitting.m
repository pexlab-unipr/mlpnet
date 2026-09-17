clear; close all; clc;

xt = linspace(-5,5,25);
yt = 5*sin(xt) + 1*randn(size(xt));

net = mlpnet([1,8,7,6,1]);
% net.f = @(x) max(0,x);
% net.df = @(x) x>0;

%% Train the network
for epoch = 1:5e3
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
linkaxes(ax,'x')
legend('Numerical','Network')