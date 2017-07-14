function [m, S, m_t, S_t, m_y, S_y, w, mean_sum, cov_sum, mean_sum_obs, cov_sum_obs] = ...
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

w_cum = cumsum(w);
new_points = zeros(D, M);
for i=1:M  %this should be x.... TODO_M
    selected_gaussian = find(w_cum >= rand, 1);
    new_points(:,i) = mvnrnd(mean_sum(:,selected_gaussian),cov_sum(:,selected_gaussian));
end
%{
inital_dist = gmdistribution(pm',pS,w);
if isempty(new_points)
    new_points = random(inital_dist, M); %sampling: gives points M-by-E 
    % todo_m Do it in another way.... !
end
%}

%figure; hist(new_points)
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

for i=1:M
    %[m_t S_t] = gpPt(X_t, input_t, target_t, pm(1), pS(1)); % call transition GP
    [m_y, S_y, Cxy] = gpPo(X_o, input_o, target_o, m_t(i), S_t(i)); % call observation GP
    
    % filter step: combine prediction and measurement
    L = chol(S_y)'; B = L\(Cxy');
    m(i) = m_t(i) + Cxy*(S_y\(y-m_y));
    S(i) = S_t(i) - B'*B;
    if y_old == 0 %TODO_M: hack
        w(i) = 1/M;
    else
        w(i) = mvnpdf(y_old, myy(i), syy(i));
    end
end
mean_sum_obs = m;
cov_sum_obs = S;
wa = w/sum(w);
if isnan(wa(1))
    disp('hi')
end
w = wa;
S_t = mean(S_t+m_t.^2)-mean(m_t).^2;
m_t = mean(m_t);
S = sum((S(:)+m(:).^2)'.*w) - sum(m.*w).^2;
m = sum(m.*w);
w = (w+0.00000000000000000000000001);  %TODO_M: wtf?
w = w/sum(w);
end