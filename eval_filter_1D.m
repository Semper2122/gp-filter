function [sqmaha, nllx, nlly, rmsex, nll_over_steps, random_seed] = eval_filter_1D(flag1, flag2, M, T, noTest)
% several filters (EKF, UKF, GP-UKF, GP-ADF) tested on a scalar function
%
% inputs arguments (number of arguments counts, not the value)
% flag1: indicates whether figures shall be drawn
% flag2: indicates whether figures shall be printed
%
% returns:
% sqmaha: square-root of Mahalanobis distance in x-space
% nllx:   Negative log-likelihood in x-space (point-wise)
% nlly:   Negative log-likelihood in y-space (point-wise)
% rmsex:  RMSE in x-space
%
% (C) Marc Deisenroth and Marco Huber, 2009-11-09

% close all; clear functions;
switch 0 %nargin
  case 0
    
    %clear all; 
    close all;
    fig = 32;
    printFig = 0;
    random_seed = randi(10000); %21 the best so far
    randn('seed',2);
    rand('twister',random_seed);
  case 1
    clear all; close all;
    fig = 0;
    printFig = 0;
  case 2
    clear all; close all;
    fig = 1;
    printFig = 1;
    randn('state',2);
    rand('twister',4);
end

% some defaults for the plots
set(0,'defaultaxesfontsize',30);
set(0,'defaultaxesfontunits', 'points')
set(0,'defaulttextfontsize',33);
set(0,'defaulttextfontunits','points')
set(0,'defaultaxeslinewidth',0.1);
set(0,'defaultlinelinewidth',2);
set(0,'DefaultAxesLineStyleOrder','-|--|:|-.');

% Parameters for UKF
alpha = 1; beta  = 0; kappa = 2;

%% SET DEFAULT VALUES

if nargin == 0
   flag1 = 1;
   flag2 = 1;
end
if nargin < 3
    random_seed = randi(10000); %2580; %21 the best so far
    randn('seed',2);
    rand('twister',random_seed);
    M = 2000;
    
    T =4;        % length of prediction horizon
    noTest = 10;%200;  %Before I used.. 201
end
%% Kitagawa-like model
c(1) = 0.5;
c(2) = 25;
c(3) = 5;
c(4) = 0;
c(5) = 2;

% system model
afun2 = @(x,u,n,t) c(1)*x + c(2)*x./(1+x.^2) + c(4)*cos(1.2);
A = @(x,u,n,t) c(1) - 2*c(2)*x.^2./(1 + x.^2)^2 + c(2)./(1 + x.^2);
B = @(x,u,n,t) 1;


% MBV: new system model 
%afun2 = @(x,u,n,t) c(1)*sin(pi*x/5+2) + c(2)*x./(1+x.^2) + c(4)*cos(1.2);
%A = @(x,u,n,t) c(1)*cos(pi*x/5+2)*pi/5 - 2*c(2)*x.^2./(1 + x.^2)^2 + c(2)./(1 + x.^2);
%B = @(x,u,n,t) 1;


% measurement model
hfun2 = @(x,u,n,t) c(3).*sin(c(5)*x);  %x.^2; %x.^2; %
H = @(x,u,n,t) c(3)*c(5).*cos(c(5)*x); %*2*x; 2*x; %

%%
if 0%nargin == 1 
  % sample some noise variances
  C =  (1*rand(1)+1e-04)^2;  % prior uncertainty
  Cw = (1*rand(1)+1e-04)^2;  % system noise
  Cv = (1*rand(1)+1e-04)^2; % measurement noise
else
  C = 0.5^2;
  Cw = 0.2^2;
  Cv = 0.01^2;
  if ~ flag2
      Cw = 1.5;
      Cv = 1;
  end
end
Cw
Cv
afun = @(x,u,n,t) afun2(x,u,n,t) + n;
hfun = @(x,u,n,t) hfun2(x,u,n,t) + n;


%% Learn Models
nd = 200; % size dynamics training set
nm = 50; % size of measurement training set  %TODO_M: changed to make it consistent..

% covariance function
covfunc={'covSum',{'covSEard','covNoise'}};


% learn dynamics model
xd = 40.*rand(nd,1)-20; %linspace(0,1,nd)'-20;  %
yd = afun(xd, [], 0, []) + sqrt(Cw).*randn(nd,1);
Xd = trainf(xd,yd);  disp(exp(Xd))

if 0
  % plot the dynamics model
  xx = linspace(-20,20,400)';
  [mxx sxx] = gpr(Xd,covfunc,xd,yd,xx);

  figure(1)
  hold on
  f = [mxx+2*sqrt(sxx);flipdim(mxx-2*sqrt(sxx),1)];
  fill([xx; flipdim(xx,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
  plot(xd,yd,'ro','markersize',2);
  plot(xx, mxx, 'k')
  plot(xx,afun(xx, [], 0, []),'b');
  xlabel('input state');
  ylabel('succ. state');
  axis tight
  if printFig; print_fig('dynModel'); end
  disp('this is the learned dynamics model'); 
  disp('press any key to continue');
  pause
end

% learn observation model
xm = 40.*linspace(0,1,nm)'-20; %%rand(nm,1)-20;
ym = hfun(xm, [], 0, []) + sqrt(Cv).*randn(nm,1);
Xm = trainf(xm,ym);  disp(exp(Xm))
%save('trained_GPs', 'xd','yd','Xd', 'xm','ym','Xm')


if 0
  % plot the observation model
  xx = linspace(-20,20,400)';
  [mxx sxx] = gpr(Xm,covfunc,xm,ym,xx);
  figure; plot(xx, sqrt(sxx))
  figure(2); clf
  hold on
  f = [mxx+2*sqrt(sxx);flipdim(mxx-2*sqrt(sxx),1)];
  fill([xx; flipdim(xx,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
  plot(xm,ym,'ro','markersize',2);
  plot(xx, mxx, 'k')
  plot(xx,hfun(xx, [], 0, []),'b');
  xlabel('input state');
  ylabel('observation');
  if printFig; print_fig('measModel'); end
  disp('this is the learned measurement model'); 
  disp('press any key to continue');
  pause
end
%{
data = load('trained_GPs.mat');
xd = data.xd; yd = data.yd; Xd = data.Xd;
xm = data.xm; ym = data.ym; Xm = data.Xm;
%}

%% some error measures
sfun = @(xt, x, C) (x-xt).^2; % squared distance
smfun = @(xt, x, C) (x-xt).^2./C./length(x); % squared Mahalanobis distance per point
mfun = @(xt, x, C) sqrt((x-xt).^2./C)./length(x); % Mahalanobis distance per point
nllfun = @(xt, x, C) (0.5*log(C) + 0.5*(x-xt).^2./C + 0.5.*log(2*pi))./length(x); % NLL per point


%% State estimation
x = linspace(-10, 10, noTest); % means of initial states
x = linspace(-1, 1, noTest); % means of initial states
y = zeros(1, noTest);  % observations
num_models = 6;
% Considered estimators: (1) ground truth, (2) ukf, (3) gpf, (4) ekf
xp = zeros(num_models, noTest,T+1); % predicted state
xe = zeros(num_models, noTest,T+1); % filtered state
xy = zeros(num_models, noTest,T+1); % predicted measurement (mean)
Cy = zeros(num_models, noTest,T+1); % predicted measurement (variance)
Cp = zeros(num_models, noTest,T+1); % predicted variance
Ce = zeros(num_models, noTest,T+1); % filtered variance

xe(:,:,1) = repmat(x,num_models,1);
xe(1,:,1) = chol(C)'*randn(noTest,1) + x';
Ce(:,:,1) = repmat(C,num_models,noTest);

%%%%% VARIABLES FOR GP-SUM %%%%%%%
weights = repmat(1/M,noTest,M, T+1);  %weight gaussians
mean_sum = repmat(x',1, M, T+1);  %weight gaussians
cov_sum = repmat(C,noTest,M, T+1);  %weight gaussians
mean_sum_obs = mean_sum;
cov_sum_obs = cov_sum;
mean_sum_y = mean_sum;
cov_sum_y = cov_sum;
y_old = zeros(1, noTest);
%%%%%%%%

%%%%%%% VARIABLES FOR TRUE DISTRIBUTION %%%%

num_x = 1000;
limit_x = 20;
pos_x = linspace(-limit_x,limit_x,num_x);
initial_x = zeros(length(x),num_x);
prop_x = zeros(length(x),num_x);
next_x = zeros(length(x),num_x);
for j = 1:length(x)
    initial_x(j,:) = normpdf(pos_x, x(j), sqrt(C)); 
    initial_x(j,:) = initial_x(j,:)/sum(initial_x(j,:));
end
[x_m, x_S] = gpr(Xd,covfunc, xd, yd,pos_x');
[z_m, z_S] = gpr(Xm,covfunc, xm, ym,pos_x');
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ww = sqrt(Cw)'*randn(100,2000);
vv = sqrt(Cv)'*randn(100,2000);


for t = 1:T
    t
    random_seed
  for i = 1:length(x)
          i
      
    %----------------------------- Ground Truth --------------------------
    w = ww(t,i);%sqrt(Cw)'*randn(1);
    v = vv(t,i);%sqrt(Cv)'*randn(1);
    %if i ~= [7], continue; end
    %tic
    xp(1,i,t+1) = afun(xe(1,i,t), [], w, []);
    xe(1,i,t+1) = xp(1,i,t+1);
    y(i) = hfun(xp(1,i,t+1), [], v, []);
    xy(1,i,t+1) = y(i);
    
    %----------------------------- Real Dist -----------------------------
    
    for j = 1:num_x
        prop_x(i,j) = sum(initial_x(i,:).*normpdf(pos_x(j), x_m, sqrt(x_S))');
    end
	prop_x(i,:) = prop_x(i,:)/sum(prop_x(i,:))';
    initial_x(i,:) = prop_x(i,:).*normpdf(y(i), z_m, sqrt(z_S))';
    initial_x(i,:) = initial_x(i,:)/sum(initial_x(i,:));
    %}
    %--------------------------------- UKF -------------------------------
    [xe(2,i,t+1), Ce(2,i,t+1), xp(2,i,t+1), Cp(2,i,t+1), xy(2,i,t+1), Cy(2,i,t+1)] = ...
      ukf_add(xe(2,i,t), Ce(2,i,t), [], Cw, afun, y(i), Cv, hfun, [], alpha, beta, kappa);
 
    %------------------------------ GP-ADF -------------------------------
    [xe(3,i,t+1), Ce(3,i,t+1), xp(3,i,t+1), Cp(3,i,t+1), xy(3,i,t+1), Cy(3,i,t+1)] = ...
      gpf(Xd, xd, yd, Xm, xm, ym, xe(3,i,t), Ce(3,i,t), y(i));
  
    %--------------------------------- EKF -------------------------------
    AA = A(xe(4,i,t), [], 0, []); % Jacobian of A
    BB = B(xe(4,i,t), [], 0, []); % Jacobian of B
    xp(4,i,t+1) = afun(xe(4,i,t), [], 0, []);   % predicted state mean
    Cp(4,i,t+1) = AA*Ce(4,i,t)*AA' + BB*Cw*BB'; % predicted state covariance

    HH = H(xp(4,i,t+1), [], 0, []); % Jacobian of H
    xy(4,i,t+1) = hfun(xp(4,i,t+1), [], 0, []); % predicted measurement mean
    Cy(4,i,t+1) = (Cv + HH*Cp(4,i,t+1)*HH'); % predicted measurmeent covariance

    K = Cp(4,i,t+1)*HH'/Cy(4,i,t+1); % Kalman gain
    xe(4,i,t+1) = xp(4,i,t+1) + K*(y(i) - xy(4,i,t+1)); % filter mean
    Ce(4,i,t+1) = Cp(4,i,t+1) - K*HH*Cp(4,i,t+1);       % filter covariance

    %--------------------------------- GP-UKF -------------------------------
    [xe(5,i,t+1), Ce(5,i,t+1), xp(5,i,t+1), Cp(5,i,t+1), xy(5,i,t+1), Cy(5,i,t+1)] = ...
      gpukf(xe(5,i,t), Ce(5,i,t), Xd, xd, yd, y(i), Xm, xm, ym, alpha, beta, kappa);

    %------------------------------ GP-SUM -------------------------------
    [xe(6,i,t+1), Ce(6,i,t+1), xp(6,i,t+1), Cp(6,i,t+1), xy(6,i,t+1), Cy(6,i,t+1), weights(i,:,t+1), mean_sum(i,:,t+1), cov_sum(i,:,t+1), mean_sum_obs(i,:,t+1), cov_sum_obs(i,:,t+1), mean_sum_y(i,:,t+1), cov_sum_y(i,:,t+1)] = ...
      gp_sum(Xd, xd, yd, Xm, xm, ym, y(i), M, weights(i,:,t), y_old(i), mean_sum(i,:,t), cov_sum(i,:,t), i);
    %toc
    %------------------------------ PLOT EVOLUTION -------------------------------
    if 0% i ==7 %i==2 %i == 94 %i == floor(length(x)/2)+1 && flag2 %Case where x = 0
        
        w
        v
        disp('afun')
        afun(xe(1,i,t), [], w, [])
        afun(xe(6,i,t), [], 0, [])
        afun(xe(1,i,t), [], 0, [])
        disp('hfun')
        hfun(xp(1,i,t+1), [], v, [])
        hfun(xp(6,i,t+1), [], 0, [])
        hfun(xp(1,i,t+1), [], 0, [])
        disp('mean_sum')
        mean_sum(i,:,t+1)
        weights(i,:,t+1)
        mean_sum_y(i,:,t+1)
        disp('y_old')
        y_old(i)
        y(i)
        mean(mean_sum(i,:,t))
        mean(mean_sum(i,:,t+1))
        xp(6,i,t+1)
        xe(6,i,t+1)
        disp('Cp,Ce')
        Ce(6,i,t+1)
        Cp(6,i,t+1)
        figure; hist(weights(i,:,t+1))
        figure; hist(mean_sum(i,:,t+1))
        %}
        xx = linspace(-20,20,250);
        yy = xx*0; yy_obs = yy;
        for j=1:M
           yy = yy +  weights(i,j,t+1)*normpdf(xx, mean_sum(i,j,t+1), sqrt(cov_sum(i,j,t+1)));
           yy_obs = yy_obs + weights(i,j,t+1)*normpdf(xx, mean_sum_obs(i,j,t+1), sqrt(cov_sum_obs(i,j,t+1)));
        end
%        figure(2); subplot(1,4, 2*t-1);
        figure; 
        hold on; plot(xx, yy); title('Propagated')
        plot(xx, normpdf(xx,xp(3,i,t+1), sqrt(Cp(3,i,t+1)))); 
        plot(xx, normpdf(xx,xp(5,i,t+1), sqrt(Cp(5,i,t+1)))); 
        plot(xx, normpdf(xx,xp(6,i,t+1), sqrt(Cp(6,i,t+1)))); 
        plot(xe(1,i,t+1), 0, 'o'); xlabel(num2str(t));
        plot(xe(1,i,t), 0, 'ok'); xlabel(num2str(t));
        plot(pos_x, prop_x(i,:)*num_x/(2*limit_x), '.');
        if t == 4, disp('hi')
        end
        %hold on; plot(xx,yy_obs); 
        %{
        disp('UFK')
        xe(5,i,t+1)
        sqrt(Ce(5,i,t+1))
        disp('ADF')
        xe(3,i,t+1)
        sqrt(Ce(3,i,t+1))
        %}
        if 1% t < 2
        %        figure(2); subplot(1,4, 2*t);
        figure; 
        hold on; plot(xx,yy_obs); title('Filtered')
        plot(xx, normpdf(xx,xe(3,i,t+1), sqrt(Ce(3,i,t+1)))); 
        plot(xx, normpdf(xx,xe(5,i,t+1), sqrt(Ce(5,i,t+1)))); 
        plot(xx, normpdf(xx,xe(6,i,t+1), sqrt(Ce(6,i,t+1)))); 
        plot(xe(1,i,t+1), 0, 'o'); xlabel(num2str(t))
        plot(xe(1,i,t), 0, 'ok'); xlabel(num2str(t))
        plot(pos_x, initial_x(i,:)*num_x/(2*limit_x),'.');
        end
    end
  end
  y_old = y;
end

%% Plot

if 0 %fig
  filternames = {'true','UKF', 'GP-ADF', 'EKF','GP-UKF', 'GP-SUM'};
  for i = 2:num_models
    figure;
    clf
    hold on

    errorbar(x,xe(i,:,T+1),2*sqrt(Ce(i,:,T+1))','rd');
    plot(x, xe(1, :,T+1), 'ko', 'LineWidth', 2);

    xlabel('\mu_0');
    ylabel('p(x_1|z_1,\mu_0,\sigma^2_0)');

    leg = legend(filternames{i},'true');
    set(leg, 'box', 'off', 'FontSize', 30, 'location', 'southeast');
    axis([-10 10 -30 30])
    if printFig; print_fig(['filterDistr_redraw' filternames{i}]); end
    disp(['this plot shows the filtered state distribution for the ' filternames{i}]);
    disp('press any key to continue');
    %pause
  end
end


%% some evaluations
disp('maha (x space)')
models_names = ['UKF           GP-ADF           EKF           GP-UKF       GP-SUM        GOOD-GP-SUM'];
disp(models_names)
for i=2:num_models
sqmaha(i-1) = sum(mfun(xe(1,:,T+1), xe(i,:,T+1), Ce(i,:,T+1)));
if i == num_models, sqmaha(i) = sqmaha(i-1); end   %No better way to do it
end
disp(num2str(sqmaha));

disp('pointwise NLL (x space):')
disp(models_names)
for i =2:num_models
nllx(i-1) = sum(nllfun(xe(1,:,T+1), xe(i,:,T+1), Ce(i,:,T+1)));
if i == num_models -1
    nllfun(xe(1,:,T+1), xe(i,:,T+1), Ce(i,:,T+1))
    -log(normpdf(xe(1,:,T+1), xe(i,:,T+1), sqrt(Ce(i,:,T+1))))/3
end
if i == num_models
    nll_sum_gp = zeros(length(x),1);
    for j=1:M, nll_sum_gp = nll_sum_gp+normpdf(xe(1,:,T+1), mean_sum_obs(:,j,T+1)', sqrt(cov_sum_obs(:,j,T+1)'))'.*weights(:,j,T+1); end
    -log(nll_sum_gp)./length(x)
    nllx(i) = sum(-log(nll_sum_gp)./length(x));
end
end
disp(num2str(nllx));

disp('RMSE (x space)')
disp(models_names)
for i =2:num_models
    rmsex(i-1) = sqrt(mean(sfun(xe(1,:,T+1), xe(i,:,T+1))));
    if i == num_models, rmsex(i) = rmsex(i-1); end
end
disp(num2str(rmsex));


disp('pointwise NLL (y space):')
disp(models_names)
for i =2:num_models
    nlly(i-1) = sum(nllfun(xy(1,:,T+1), xy(i,:,T+1), Cy(i,:,T+1)));
    if i == num_models
        nll_sum_gp = zeros(length(x),1);
        for j=1:M, nll_sum_gp = nll_sum_gp +normpdf(xy(1,:,T+1), mean_sum_y(:,j,T+1)', sqrt(cov_sum_y(:,j,T+1)'))'.*weights(:,j,T+1); end
        nlly(i) = sum(-log(nll_sum_gp)./length(x));
    end
end
disp(num2str(nlly));

%{
disp('KL divergence:')
disp(models_names)
for i =2:num_models
%nllx(i-1) = sum(KLfun(xe(1,:,T+1), xe(i,:,T+1), Ce(i,:,T+1)));
if i == num_models
    KL_sum_gp = zeros(length(x),1);
    for k=1:num_x
        if mod(k,10) ~= 0, continue; end
        q_x = initial_x(:,k)*num_x/(2*limit_x);
       for j=1:M
           p_x = normpdf(pos_x(:,k), mean_sum_obs(:,j,T+1)', sqrt(cov_sum_obs(:,j,T+1)'))'.*weights(:,j,T+1);
           KL_sum_gp = KL_sum_gp+log(p_x./q_x).*p_x; 
       end
    end
    KL(i) = sum(KL_sum_gp)./length(x);
end
end
%}
disp(num2str(nllx));


%% Evaluations overtime
nll_over_steps = zeros(num_models+1,T+1);
for i=2:num_models
    nll_over_steps(i,:) = sum(nllfun(xe(1,:,:), xe(i,:,:), Ce(i,:,:)));
end
% Computation for the GP-SUm
for t=1:T    
    nll_sum_gp = zeros(length(x),t);
    for j=1:M, nll_sum_gp(:,t) = nll_sum_gp(:,t)+normpdf(xe(1,:,t+1), mean_sum_obs(:,j,t+1)', sqrt(cov_sum_obs(:,j,t+1)'))'.*weights(:,j,t+1); end
    find(-log(nll_sum_gp(:,t))./length(x) > 1000)
    nll_over_steps(num_models+1,t) = sum(-log(nll_sum_gp(:,t))./length(x));
end

if 0 %flag1
    figure; plot(nll_over_steps(6,:)) %GP-SUM as one gaussian
    hold on; plot(nll_over_steps(3,:)) %GP-ADF
    hold on; plot(nll_over_steps(num_models+1,:)) %GP-SUM
    legend('GP-SUM','GP-ADF', 'GOOD-GP-SUM');
    %hold on; plot(nll_over_steps(5,:)) %GP-UKF
    %legend('GP-SUM','GP-ADF','GP-UKF ', 'GOOD-GP-SUM');
end


%% print figures
function print_fig(filename)
path = './figures/';
grid on
set(gcf,'PaperSize', [10 6]);
set(gcf,'PaperPosition',[0.1 0.1 10 6]);
print('-depsc2','-r300', [path filename '.eps']);
eps2pdf([path filename '.eps']);
delete([path filename '.eps']);
end
end
