function [m, S, m_t, S_t, m_y, S_y, w, mean_sum, cov_sum, mean_sum_obs, cov_sum_obs, mean_sum_y, cov_sum_y] = ...
  gp_sum(X_t, input_t, target_t, X_o, input_o, target_o, pm, pS, y, M, w, y_old, mean_sum, cov_sum, true_x)

% Bayesian filter using GP models for transition dynamics and observation
% (trained offline)
% assumes that the GP models are NOT learned on differences
%
% inputs:
% X_t:        (D+2)*E-by-1 vector of log-hyper-parameters (transition model)
% input_t:    n-by-D matrix of training inputs (transition model)
% target_t:   n-by-E matrix of training targets (transition model)
% X_o:        (E+2)*F-by-1 vector of log-hyper-parameters (observation model)
% input_o:    n-by-E matrix of training inputs (observation model)
% target_o:   n-by-F matrix of training targets (observation model)
% pm:         D-by-1 mean of current (hidden) state distribution
% pS:         D-by-D covariance matrix of current (hidden) state distribution
% y:          F-by-1 measurement at next time step
%
% outputs:
% m:        E-by-1 mean vector of filtered distribution
% S:        E-by-E covariance matrix of filtered distribution
% m_t:      E-by-1 mean vector of the predicted state distribution
% S_t:      E-by-E covariance matrix of the predicted state distribution
% m_y:      F-by-1 mean vector of predicted measurement distribution
% S_y:      F-by-F covariance matrix of predicted measurement distribution
% 
% (C) Marc Peter Deisenroth
% 2009-07-06

% predictive state distribution p(x_t|y_1,...,y_{t-1}), no incorporation of current
% measurement
%tic
[n, D] = size(input_t);          % number of examples and dimension of input space
[n, E] = size(target_t);                % number of examples and number of outputs
%X_t = reshape(X_t, D+2, E)';

%Previous: [m_t S_t] = gpPt(X_t, input_t, target_t, pm, pS); % call transition GP
%This has to be a sum of gaussians:
if size(pm,2) == 1 
    pm = repmat(pm, 1, M); %E-by-M
    pS = repmat(pS,1,1,M);
    %w = repmat(1/M,1,M); %1-by-M
end

selected_gaussians = randsample(M, M, true, w);
new_points = mvnrnd(mean_sum(selected_gaussians),sqrt(cov_sum(selected_gaussians))); %wont generalize to multiple dimensions... TODO

%%% Predict mean and variance for new_points
% covariance function
covfunc={'covSum',{'covSEard','covNoise'}};
%tic
[m_t, S_t] = gpr(X_t,covfunc,input_t,target_t,new_points');
[myy, syy] = gpr(X_o,covfunc,input_o,target_o,new_points');
%time_gpr = toc; disp('time_gpr'); disp(time_gpr);
mean_sum = m_t;
cov_sum = S_t;
%Create new gaussians:
m = zeros(D,M); %Won't scale high dimension x...
w = zeros(1,M); %weight of each gaussian
S = zeros(D, D, M);
% compute measurement distribution
m_y = zeros(D,M);
S_y = zeros(D,M);
Cxy = zeros(D,M);
%tic

%{
for i=1:M
    [m_y(i), S_y(i), Cxy(i)] = gpPo(X_o, input_o, target_o, m_t(i), S_t(i)); % call observation GP
    %{
    L = chol(S_y(i))'; B = L\(Cxy(i)');  m(i) = m_t(i) +
    Cxy(i)*(S_y(i)\(y-m_y(i))); S(i) = S_t(i) - B'*B; 
    %}
end
%}
tic
[m_y, S_y, Cxy] = gpPoSum(X_o, input_o, target_o, m_t, S_t); % call observation GP
time_gpPo = toc; disp('time_gpPo_1'); disp(time_gpPo)
%time_gpPo = toc; disp('time_gpPo'); disp(time_gpPo)
m = m_t' + Cxy.*(y-m_y)./S_y;
S = S_t' - Cxy.^2./S_y;
if y_old == 0, w = repmat(1/M, 1, M);  %TODO_M: hack, find a better way...
else, w = normpdf(y_old, myy, sqrt(syy))'; end
mean_sum_obs = m;
cov_sum_obs = S;
mean_sum_y = m_y;
cov_sum_y = S_y;
w = w/sum(w);
S_t = mean(S_t+m_t.^2)-mean(m_t).^2;
m_t = mean(m_t);
S = sum((S(:)+m(:).^2)'.*w) - sum(m.*w).^2;
m = sum(m.*w);
S_y = sum((S_y(:)+m_y(:).^2)'.*w) - sum(m_y.*w).^2;
m_y = sum(m_y.*w);
if sum(w) == 0
    disp('stooooooooooooooooooop')
end
%time_gpPo = toc; disp('time_gpPo_3'); disp(time_gpPo)
%w = (w+0.00000000000000000000000001);  %TODO_M: wtf?
%w = w/sum(w);
end