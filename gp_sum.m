function [m, S, m_t, S_t, m_y, S_y, w, mean_sum, cov_sum, mean_sum_obs, cov_sum_obs, mean_sum_y, cov_sum_y] = ...
  gp_sum(X_t, input_t, target_t, X_o, input_o, target_o, y, M, w_old, y_old, mean_sum_old, cov_sum_old, it_x)
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

selected_gaussians = randsample(length(w_old), M, true, w_old);
new_points = normrnd(mean_sum_old(selected_gaussians),sqrt(cov_sum_old(selected_gaussians))); %wont generalize to multiple dimensions... TODO

%%% Predict mean and variance for new_points
% covariance function
covfunc={'covSum',{'covSEard','covNoise'}};
[m_t, S_t] = gpr(X_t,covfunc,input_t,target_t,new_points');
[myy, syy] = gpr(X_o,covfunc,input_o,target_o,new_points');

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


for i=1:M
    [m_y(i), S_y(i), Cxy(i)] = gpPo(X_o, input_o, target_o, m_t(i), S_t(i)); % call observation GP
    % filter step: combine prediction and measurement
    L = chol(S_y(i))'; B = L\(Cxy(i)');
    m(i) = m_t(i)+ Cxy(i)*(S_y(i)\(y-m_y(i)));
    S(i) = S_t(i) - B'*B;
end

%{  
%%% Trying to make it fast...
for i=1:M
    [m_y(i), S_y(i), Cxy(i)] = gpPo(X_o, input_o, target_o, m_t(i), S_t(i)); % call observation GP
    %{
    L = chol(S_y(i))'; B = L\(Cxy(i)');  m(i) = m_t(i) +
    Cxy(i)*(S_y(i)\(y-m_y(i))); S(i) = S_t(i) - B'*B; 
    %}
end

time_gpPo = toc; disp('time_gpPo_1'); disp(time_gpPo)
tic
[m_y, S_y, Cxy] = gpPoSum(X_o, input_o, target_o, m_t, S_t); % call observation GP
time_gpPo = toc; disp('time_gpPo_2'); disp(time_gpPo)
%time_gpPo = toc; disp('time_gpPo'); disp(time_gpPo)
m = m_t' + Cxy.*(y-m_y)./S_y;
S = S_t' - Cxy.^2./S_y;
%}
if y_old == 0, w = repmat(1/M, 1, M);  %TODO_M: hack, find a better way...
else w = normpdf(y_old, myy, sqrt(syy))'; end

something_went_wrong = 0;
threshold = normpdf(exp(X_o(3)),0,exp(X_o(3)));
if max(w) < threshold & y_old ~= 0

max_w = max(w);
%{
max_w
threshold
it_x
%}
something_went_wrong = 1;
end
mean_sum_obs = m;
cov_sum_obs = S;
mean_sum_y = m_y;
cov_sum_y = S_y;

if ~(sum(w) > 0) || ~all(w>=0)  
    w = (w+0.00000000000000000000000001);  %TODO_M: wtf?
    if i == 94, disp('stooooooooooooooooooop_special_case'); end
    disp('stooooooooooooooooooop')
   something_went_wrong = 1; 
end
w = w/sum(w);
S_t = sum((S_t(:)+m_t(:).^2)'.*w)-sum(m_t'.*w).^2;
m_t = sum(m_t'.*w);
S = sum((S(:)+m(:).^2)'.*w) - sum(m.*w).^2;
m = sum(m.*w);
S_y = sum((S_y(:)+m_y(:).^2)'.*w) - sum(m_y.*w).^2;
m_y = sum(m_y.*w);

if 0 %something_went_wrong
   new_M = 2*M;
   disp('new_M:'); disp(new_M);
   [m, S, m_t, S_t, m_y, S_y, w, mean_sum, cov_sum, mean_sum_obs, cov_sum_obs, mean_sum_y, cov_sum_y] = ...
  gp_sum(X_t, input_t, target_t, X_o, input_o, target_o, y, new_M, w_old, y_old, mean_sum_old, cov_sum_old, it_x);

[~, index_w] = sort(w,'descend');
index_w = index_w(1:M);
w = w(index_w);
mean_sum = mean_sum(index_w);
cov_sum = cov_sum(index_w);
mean_sum_obs = mean_sum_obs(index_w);
cov_sum_obs = cov_sum_obs(index_w);
mean_sum_y = mean_sum_y(index_w);
cov_sum_y = cov_sum_y(index_w);
end


end