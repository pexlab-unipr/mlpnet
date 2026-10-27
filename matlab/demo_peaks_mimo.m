clear; close all; clc;

%% Generate the dataset
[X,Y,Z] = peaks(100);
idx = randi([1,numel(Z)],1e3,1);
xt = X(idx);
yt = Y(idx);
zt = Z(idx)/max(abs(X(idx)));

%% Create the network
nn = mlpnet([2,10,10,10,3]);

%% Train the network
train_env = 'matlab'; % training environment
train_alg = 'lma'; % algorithm used in MATLAB

switch train_env
    case 'matlab' % train here
        tic
        switch train_alg
            case 'sgd' % use stochastic gradient descent
                for epoch = 1:1000
                    if ~mod(epoch,500); disp(epoch); end
                    for k = 1:length(zt)
                        nn.update([xt(k),yt(k)],zt(k)*[1,-1,0.1]);
                    end
                end
            case 'lma' % use Levenberg-Marquardt algorithm
                [it,e_norm] = nn.train([xt,yt],zt.*[1,-1,0.1]);
        end
        toc
    case 'c' % train using C interface
        % Export the dataset
        folder = "../";

        fid = fopen(folder + "xt.bin",'w');
        fwrite(fid, xt, 'single');
        fclose(fid);
        fid = fopen(folder + "yt.bin",'w');
        fwrite(fid, yt, 'single');
        fclose(fid);
        fid = fopen(folder + "zt.bin",'w');
        fwrite(fid, zt, 'single');
        fclose(fid);

        % Train
        fprintf("Run 'main_mimo_training' and press enter...\n");
        pause()

        % Load the parameters
        fid = fopen(folder + "par.bin",'r');
        par = fread(fid, inf, 'single');
        fclose(fid);
        idx = 0;
        for k = 1:nn.nh+1
            idx = idx(end)+1:idx(end)+nn.sz(k+1);
            nn.B{k} = par(idx).';
            idx = idx(end)+1:idx(end)+nn.sz(k+1)*nn.sz(k);
            nn.W{k} = reshape(par(idx), nn.sz(k+1), nn.sz(k)).';
        end
end

%% Evaluate the network on a grid
[xg,yg] = ndgrid(linspace(min(xt),max(xt),50),linspace(min(yt),max(yt),51));
zg = zeros([numel(xg),nn.sz(end)]);
J1g = zeros([numel(xg),nn.sz(1)]);
J2g = zeros([numel(xg),nn.sz(1)]);
J3g = zeros([numel(xg),nn.sz(1)]);
for k = 1:numel(xg)
    zg(k,:) = nn.eval([xg(k),yg(k)]);
    J = nn.jacobian([xg(k),yg(k)]);
    J1g(k,:) = J(1,:);
    J2g(k,:) = J(2,:);
    J3g(k,:) = J(3,:);
end
zg = reshape(zg,[size(xg,1),size(xg,2),nn.sz(end)]);
J1g = reshape(J1g,[size(xg,1),size(xg,2),nn.sz(1)]);
J2g = reshape(J2g,[size(xg,1),size(xg,2),nn.sz(1)]);
J3g = reshape(J3g,[size(xg,1),size(xg,2),nn.sz(1)]);

%% Compute the gradients numerically
[dzg1dy,dzg1dx] = gradient(zg(:,:,1),yg(1,2)-yg(1,1),xg(2,1)-xg(1,1));
[dzg2dy,dzg2dx] = gradient(zg(:,:,2),yg(1,2)-yg(1,1),xg(2,1)-xg(1,1));
[dzg3dy,dzg3dx] = gradient(zg(:,:,3),yg(1,2)-yg(1,1),xg(2,1)-xg(1,1));

%% Plot
fontsize = 15;

figure

subplot(1,3,1)
surf(xg,yg,zg(:,:,1))
hold on; grid on
plot3(xt,yt,zt,'*r')
legend('Fitting','Data','Location','northwest')
xlabel('x','FontSize',fontsize);
ylabel('y','FontSize',fontsize);
zlabel('z_1','FontSize',fontsize);

subplot(1,3,2)
surf(xg,yg,zg(:,:,2))
hold on; grid on
plot3(xt,yt,-zt,'*r')
xlabel('x','FontSize',fontsize);
ylabel('y','FontSize',fontsize);
zlabel('z_2','FontSize',fontsize);

subplot(1,3,3)
surf(xg,yg,zg(:,:,3))
hold on; grid on
plot3(xt,yt,zt*0.1,'*r')
xlabel('x','FontSize',fontsize);
ylabel('y','FontSize',fontsize);
zlabel('z_3','FontSize',fontsize);

figure

subplot(3,2,1)
surf(xg,yg,J1g(:,:,1)); hold on;
surf(xg,yg,dzg1dx,'FaceColor','none','EdgeColor','red')
legend('Network','Numerical','Location','northwest')
xlabel('x','FontSize',fontsize);
ylabel('y','FontSize',fontsize);
zlabel('\partialz_1 / \partialx','FontSize',fontsize)

subplot(3,2,2)
surf(xg,yg,J1g(:,:,2)); hold on;
surf(xg,yg,dzg1dy,'FaceColor','none','EdgeColor','red')
xlabel('x','FontSize',fontsize);
ylabel('y','FontSize',fontsize);
zlabel('\partialz_1 / \partialy','FontSize',fontsize)

subplot(3,2,3)
surf(xg,yg,J2g(:,:,1)); hold on;
surf(xg,yg,dzg2dx,'FaceColor','none','EdgeColor','red')
xlabel('x','FontSize',fontsize);
ylabel('y','FontSize',fontsize);
zlabel('\partialz_2 / \partialx','FontSize',fontsize)

subplot(3,2,4)
surf(xg,yg,J2g(:,:,2)); hold on;
surf(xg,yg,dzg2dy,'FaceColor','none','EdgeColor','red')
xlabel('x','FontSize',fontsize);
ylabel('y','FontSize',fontsize);
zlabel('\partialz_2 / \partialy','FontSize',fontsize)

subplot(3,2,5)
surf(xg,yg,J3g(:,:,1)); hold on;
surf(xg,yg,dzg3dx,'FaceColor','none','EdgeColor','red')
xlabel('x','FontSize',fontsize);
ylabel('y','FontSize',fontsize);
zlabel('\partialz_3 / \partialx','FontSize',fontsize)

subplot(3,2,6)
surf(xg,yg,J3g(:,:,2)); hold on;
surf(xg,yg,dzg3dy,'FaceColor','none','EdgeColor','red')
xlabel('x','FontSize',fontsize);
ylabel('y','FontSize',fontsize);
zlabel('\partialz_3 / \partialy','FontSize',fontsize)