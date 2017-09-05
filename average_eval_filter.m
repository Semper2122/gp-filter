
clear all; close all;
M = 1000;
T = 10;
num_models = 6;
num_avg = 100;
noTest = 200; %Be carefull!
random_seeds = zeros(num_avg,1);
sqmaha = zeros(num_avg,num_models);
nllx = zeros(num_avg,num_models);
nlly = zeros(num_avg,num_models);
rmsex = zeros(num_avg,num_models);
KL = zeros(num_avg,num_models);
KS = zeros(num_avg,num_models);
Norm = zeros(num_avg,num_models);
KL_prop = zeros(num_avg,num_models);
KS_prop = zeros(num_avg,num_models);
Norm_prop = zeros(num_avg,num_models);
nll_over_steps = zeros(num_models+1,T+1,num_avg);
flag1= 1; low_noise = 1; %Be carefull!
experiment_name = 'Experiment_num_avg=100_noTest=200_T=10_M=1000_redo_if_stoop_with_distances_corrected_v1';
for i=1:num_avg
    i
   [sqmaha(i,:), nllx(i,:), nlly(i,:), rmsex(i,:), nll_over_steps(:,:,i), random_seeds(i),KL(i,:), KS(i,:), Norm(i,:), KL_prop(i,:), KS_prop(i,:), Norm_prop(i,:)] = eval_filter_1D(flag1, low_noise, M, T, noTest);
   disp('nllx so far:')
   disp(mean(nllx(1:i,:),1))
   save(experiment_name)
end


save(experiment_name)

good_until = 99;
sqmaha_avg = mean(sqmaha(1:good_until,:),1);
nllx_avg = mean(nllx(1:good_until,:),1);
nlly_avg = mean(nlly(1:good_until,:),1);
rmsex_avg = mean(rmsex(1:good_until,:),1);
nll_over_steps_avg = mean(nll_over_steps(:,:,1:good_until), 3);
KL_avg = mean(KL(1:good_until,:),1);
KS_avg = mean(KS(1:good_until,:),1);
Norm_avg = mean(Norm(1:good_until,:),1);
KL_prop_avg = mean(KL_prop(1:good_until,:),1);
KS_prop_avg = mean(KS_prop(1:good_until,:),1);
Norm_prop_avg = mean(Norm_prop(1:good_until,:),1);


%% some evaluations
disp(' ');
disp(' ');
disp(' --------------- ');
disp(' ');
disp(' ');
models_names = ['UKF           GP-ADF           EKF           GP-UKF       GP-SUM        GOOD-GP-SUM'];
disp('maha (x space)')
disp(models_names)
disp(num2str(sqmaha_avg));

disp('pointwise NLL (x space):')
disp(models_names)
disp(num2str(nllx_avg));


disp('RMSE (x space)')
disp(models_names)
disp(num2str(rmsex_avg));


disp('pointwise NLL (y space):')
disp(models_names)
disp(num2str(nlly_avg));

disp('KL divergence:')
disp(models_names)
disp(num2str(KL_avg));
disp('Norm of the diference:')
disp(models_names)
disp(num2str(Norm_avg));
disp('KS ditance:')
disp(models_names)
disp(num2str(KS_avg));


disp('KL divergence propagation:')
disp(models_names)
disp(num2str(KL_prop_avg));
disp('Norm of the diference propagation:')
disp(models_names)
disp(num2str(Norm_prop_avg));
disp('KS ditance propagation:')
disp(models_names)
disp(num2str(KS_prop_avg));

if flag1
    figure; plot(nll_over_steps_avg(6,:)) %GP-SUM as one gaussian
    %figure; plot(nll_over_steps_avg(5,:)) %GP-UKF as one gaussian
    hold on; plot(nll_over_steps_avg(3,:)) %GP-ADF
    hold on; plot(nll_over_steps_avg(num_models+1,:)) %GP-SUM
    %legend('GP-SUM','GP-ADF', 'GOOD-GP-SUM');
    legend('GP-SUM-MEAN','GP-ADF', 'GP-SUM');
    %hold on; plot(nll_over_steps(5,:)) %GP-UKF
    %legend('GP-SUM','GP-ADF','GP-UKF ', 'GOOD-GP-SUM');
end
