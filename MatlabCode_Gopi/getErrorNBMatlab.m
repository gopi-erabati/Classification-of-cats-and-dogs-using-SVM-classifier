function getErrorNBMatlab()
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this function is used to get train and test errors using Naive Bayes
% classifier of MATLAB
% 
% datais read from the current working directory
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;

% to load train data
data_train = load('./spamTrain.txt');
data_train_lab = load('./spamTrainLabels.txt');

% to load test data
data_test = load('./spamTest.txt');
data_test_lab = load('./spamTestLabels.txt');

%% 1.3.a Standardize the columns so they all have mean 0 and unit variance
[data_stand, data_mean_train, data_std_train] = getPreProcess( data_train , 1); % type 1 for standardise

% to preprocess test data using TEST data mean and standard variation
data_test_stand = getPreProcess( data_test , 1); % to preproces the test data 


%% 1.3.b Transform the features using log(xij + 0.1)
data_trans = getPreProcess( data_train , 2); % type 2 for transform features
data_test_trans = getPreProcess( data_test , 2); % to preproces the test data

%% 1.3.c Binarize the features using I(xij > 0), i.e.
% make every feature vector a binary vector

data_bin = getPreProcess( data_train , 3); %type 3 for binarize
data_test_bin = getPreProcess( data_test , 3); % to preproces the test data


%% get the probabilities of two classes Spam and Notspam
prob_spam = nnz(data_train_lab)/numel(data_train_lab);
prob_notspam = 1 - prob_spam;

%% Using standardise data

% Build classifier
prior = [prob_spam prob_notspam];
classNames = [1 0];
Model_stand = fitcnb(data_stand, data_train_lab, 'DistributionNames', 'normal', 'ClassNames' , classNames, 'Prior' , prior);

%predict test class
test_lab_NBMatlab = predict(Model_stand, data_test_stand);

% get error
%to compute error of Test data
error_stand_naive = getOnlyError( test_lab_NBMatlab, data_test_lab );
disp('*******************************************************************');
disp('Naive Bayes Classifier(Matlab) Result');
disp(['Error in test set using standardised data is ',num2str(error_stand_naive)]);
disp('*******************************************************************');


%% Using Transformed data

% Build classifier
prior = [prob_spam prob_notspam];
classNames = [1 0];
Model_trans = fitcnb(data_trans, data_train_lab, 'DistributionNames', 'normal', 'ClassNames' , classNames, 'Prior' , prior);

%predict test class
test_lab_NBMatlab = predict(Model_trans, data_test_trans);

% get error
%to compute error of Test data
error_trans_naive = getOnlyError( test_lab_NBMatlab, data_test_lab );
disp('*******************************************************************');
disp('Naive Bayes Classifier(Matlab) Result');
disp(['Error in test set using Transformed data is ',num2str(error_trans_naive)]);
disp('*******************************************************************');

% %% Using Binarized data
% 
% % Build classifier
% prior = [prob_spam prob_notspam];
% classNames = [1 0];
% Model_bin = fitcnb(double(data_bin), data_train_lab, 'ClassNames' , classNames, 'Prior' , prior);
% 
% %predict test class
% test_lab_NBMatlab = predict(Model_bin, data_test_bin);
% 
% % get error
% %to compute error of Test data
% error_bin_naive = getOnlyError( test_lab_NBMatlab, data_test_lab );
% disp('*******************************************************************');
% disp('Naive Bayes Classifier(Matlab) Result');
% disp(['Error in test set using Binarized data is ',num2str(error_bin_naive)]);
% disp('*******************************************************************');
