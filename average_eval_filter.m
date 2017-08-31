
clear all; close all;
M = 2000;
T = 5;
num_models = 6;
num_avg = 20;
noTest = 200; %Be carefull!
random_seeds = zeros(num_avg,1);
sqmaha = zeros(num_avg,num_models);
nllx = zeros(num_avg,num_models);
nlly = zeros(num_avg,num_models);
rmsex = zeros(num_avg,num_models);
nll_over_steps = zeros(num_models+1,T+1,num_avg);
flag1= 1; low_noise = 1; %Be carefull!
experiment_name = 'Experiment_num_avg=20_noTest=200_T=5_M=2000_low_prior_noise';
for i=1:num_avg
    i
   [sqmaha(i,:), nllx(i,:), nlly(i,:), rmsex(i,:), nll_over_steps(:,:,i), random_seeds(i)] = eval_filter_1D(flag1, low_noise, M, T, noTest);
   save(experiment_name)
end

save(experiment_name)
sqmaha = mean(sqmaha,1);
nllx = mean(nllx,1);
nlly = mean(nlly,1);
rmsex = mean(rmsex,1);
nll_over_steps = mean(nll_over_steps, 3);

%% some evaluations
disp(' ');
disp(' ');
disp(' --------------- ');
disp(' ');
disp(' ');
models_names = ['UKF           GP-ADF           EKF           GP-UKF       GP-SUM        GOOD-GP-SUM'];
disp('maha (x space)')
disp(models_names)
disp(num2str(sqmaha));

disp('pointwise NLL (x space):')
disp(models_names)
disp(num2str(nllx));


disp('RMSE (x space)')
disp(models_names)
disp(num2str(rmsex));


disp('pointwise NLL (y space):')
disp(models_names)
disp(num2str(nlly));

if flag1
    %figure; plot(nll_over_steps(6,:)) %GP-SUM as one gaussian
    figure; plot(nll_over_steps(5,:)) %GP-UKF as one gaussian
    hold on; plot(nll_over_steps(3,:)) %GP-ADF
    hold on; plot(nll_over_steps(num_models+1,:)) %GP-SUM
    %legend('GP-SUM','GP-ADF', 'GOOD-GP-SUM');
    legend('GP-UKF','GP-ADF', 'GP-SUM');
    %hold on; plot(nll_over_steps(5,:)) %GP-UKF
    %legend('GP-SUM','GP-ADF','GP-UKF ', 'GOOD-GP-SUM');
end
