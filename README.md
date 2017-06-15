# gp-adf
% These matlab functions contain the core of the GP-ADF-algorithm as described in
% Deisenroth, Huber, Hanebeck: Analytic Moment-based Gaussian Process Filtering, 
% International Conference on Machine Learning (ICML), 2009.
% 
% functions included:
% 
% eps2pdf.m:   plot/print utility
% gpf.m:       the function that does the high-level filtering (one time step)
% gpPt.m:      GP predictions with uncertain inputs (transition dynamics)
% gpPo.m:      GP predicitons with uncertain inputs (observation function)
% gpukf.m:     GP-UKF implementation
% maha.m:      computes the pairwise squared Mahalanobis distance between to
%              sets of vectors
% scaledSymmetricSigmaPoints.m:   compute sigma points for UKF
% sim_scalar.m scalar toy example to compare filters
% trainf.m:    wrapper to train multiple-target GPs
% ukf_add.m:   UKF for additive Gaussian noise
%
%
% You should be able to run the GP-ADF
%
%
% example call:
% sim_scalar
%
% 
% (C) Copyright 2009-2016, Marc Peter Deisenroth
% 
% http://wp.doc.ic.ac.uk/sml
% 2016-07-19
