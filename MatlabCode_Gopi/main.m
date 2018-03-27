% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the main file of the Homework2 which contains problem 1 and problem 2
% 
% Problem1 (Logistic Regression)
% The code line to get lambda is commented and you can uncomment it to 
% find lambda, which was found already and used in the code.
% As, there are optimisation function involved it takes some time to 
% get the reuslts on the command line.
% 
% I also implemented naive Bayes, writing my own implementation function.
% 
% Problem2 (SVM )
% You can choose feature type to select what feature to use to classify data.
% And rest of things the program does for you.
% 
% Cheers!
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all; close all;

% to load train data
data_train = load('./spamTrain.txt');
data_train_lab = load('./spamTrainLabels.txt');

% to load test data
data_test = load('./spamTest.txt');
data_test_lab = load('./spamTestLabels.txt');

%% 1.1 Max and mean of the average length of uninterrupted sequences
% of capital letters in the training set

[ data_max, data_mean ] = getMaxMean( data_train, 55 );

disp('********************************************************************************************************');
disp(['The max of the average length of uninterrupted sequences ' ...
    'of capital letters in the training set is ',num2str(data_max)]);
disp(['The mean of the average length of uninterrupted sequences ' ...
    'of capital letters in the training set is ',num2str(data_mean)]);
disp('*******************************************************************************************************');

%% 1.2 max and mean of the lengths of the longest uninterrupted sequences
%of capital letters in the training set

[ data_max, data_mean ] = getMaxMean( data_train, 56 );

disp('*******************************************************************************************************');
disp(['The max of the lengths of the longest uninterrupted sequences '...
    'of capital letters in the training set is ',num2str(data_max)]);
disp(['The mean of the lengths of the longest uninterrupted sequences ' ...
    'of capital letters in the training set is ',num2str(data_mean)]);
disp('*******************************************************************************************************');

%% 1.3.a Standardize the columns so they all have mean 0 and unit variance
[data_stand1, data_mean_train, data_std_train] = getPreProcess( [data_train;data_test] , 1); % type 1 for standardise

% to preprocess test data using TEST data mean and standard variation
% data_test_stand = getPreProcess( data_test , 1); % to preproces the test data 
data_stand = data_stand1(1:size(data_train,1),:);
data_test_stand = data_stand1(size(data_train,1)+1:end, :);

%% 1.3.b Transform the features using log(xij + 0.1)
data_trans = getPreProcess( data_train , 2); % type 2 for transform features
data_test_trans = getPreProcess( data_test , 2); % to preproces the test data

%% 1.3.c Binarize the features using I(xij > 0), i.e.
% make every feature vector a binary vector

data_bin = getPreProcess( data_train , 3); %type 3 for binarize
data_test_bin = getPreProcess( data_test , 3); % to preproces the test data

%% logistic regression with standardised data

data_stand_logis = [ones(size(data_stand,1),1) data_stand];% to add ones to first column of data

% add row of ones
% data_test_stand = [ones(size(data_test_stand,1),1) data_test_stand];

% to preprocess test data using TRAIN data mean and standard variation

% data_mean_train = repmat(data_mean_train, size(data_test,1),1);
% data_std_train = repmat(data_std_train,size(data_test,1),1);
% data_test_stand_logis = (data_test - data_mean_train)./data_std_train;

data_test_stand_logis = [ones(size(data_test_stand,1),1) data_test_stand];

% to get lambda basing on minimum validation error
%lambda  = getlambda( data_stand_logis, data_train_lab )
lambda = 0.01; % i got lambda = 0.01 after minimizing the validation error

%to get training set error
regs_para = getRegPara( data_stand_logis, data_train_lab, lambda );
error_train_stand = getError( data_stand_logis, data_train_lab, regs_para);
disp('*******************************************************************');
disp(['./Error in training set using standardised data is ',num2str(error_train_stand)]);

% to get test set error
error_test_stand =  getError( data_test_stand_logis, data_test_lab, regs_para);
disp(['./Error in test set is ',num2str(error_test_stand)]);
disp('*******************************************************************');


%% Logistic regression with Transformed feature data

data_trans_logis = [ones(size(data_trans,1),1) data_trans];% to add ones to first column of data

% add rows on ones
data_test_trans_logis = [ones(size(data_test_trans,1),1) data_test_trans];

% to get lambda basing on minimum validation error
%lambda  = getlambda( data_trans_logis, data_train_lab )
lambda = 0.005; % i got lambda = 0.005 after minimizing the validation error

%to get training set error
regs_para = getRegPara( data_trans_logis, data_train_lab, lambda );
error_train_trans = getError( data_trans_logis, data_train_lab, regs_para);
disp('*******************************************************************');
disp(['Error in training set using transformed data is ',num2str(error_train_trans)]);

% to get test set error
error_test_trans =  getError( data_test_trans_logis, data_test_lab, regs_para);
disp(['Error in test set is ',num2str(error_test_trans)]);
disp('*******************************************************************');

%% Logistic regression with Binarized feature data

data_bin_logis = [ones(size(data_bin,1),1) data_bin];% to add ones to first column of data

% add rows of ones
data_test_bin_logis = [ones(size(data_test_bin,1),1) data_test_bin];

% to get lambda basing on minimum validation error
% lambda  = getlambda( data_bin_logis, data_train_lab )
lambda = 0.005; % i got lambda = 0.005 after minimizing the validation error

%to get training set error
regs_para = getRegPara( data_bin_logis, data_train_lab, lambda );
error_train_trans = getError( data_bin_logis, data_train_lab, regs_para);
disp('*******************************************************************');
disp(['Error in training set using binarized data is ',num2str(error_train_trans)]);

% to get test set error
error_test_trans =  getError( data_test_bin_logis, data_test_lab, regs_para);
disp(['Error in test set is ',num2str(error_test_trans)]);
disp('*******************************************************************');

%% Naive Bayes

%% Naive Byes using standardised preprocessing data

% get the test labels of Naive classifier
test_lab_naive = getNaiveBayes( data_stand, data_train_lab, data_test_stand );

%to compute error of Test data
error_stand_naive = getOnlyError( test_lab_naive, data_test_lab );
disp('*******************************************************************');
disp('Naive Bayes Classifier Result');
disp(['Error in test set using standardised data is ',num2str(error_stand_naive)]);

%% Naive Byes using transformed preprocessing data

% get the test labels of Naive classifier
test_lab_naive = getNaiveBayes( data_trans, data_train_lab, data_test_trans );

%to compute error of Test data
error_trans_naive = getOnlyError( test_lab_naive, data_test_lab );
disp('*******************************************************************');
disp('Naive Bayes Classifier Result');
disp(['Error in test set using transformed data is ',num2str(error_trans_naive)]);

%% Naive Byes using Binarized preprocessing data

% get the test labels of Naive classifier
test_lab_naive = getNaiveBayes( data_bin, data_train_lab, data_test_bin );

%to compute error of Test data
error_bin_naive = getOnlyError( test_lab_naive, data_test_lab );
disp('*******************************************************************');
disp('Naive Bayes Classifier Result');
disp(['Error in test set using binarized data is ',num2str(error_bin_naive)]);


%% CATS AND DOGS CLASSIFICATION

%load cat and dog data
cat = load('catData.mat');
dog = load('dogData.mat');

catData = cat.cat;
dogData = dog.dog;

%get features from cat and dog data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FEATURE                  featureType
% 
% Intensity of pixels         1
% HarrisCorners               2
% HOG Features                3 (more accuracy)
% FAST Features               4
% SURF Features               5
% SIFT Features               6
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
featureType = 3;
[ catFeatures, dogFeatures ] = getFeatures(catData, dogData, featureType);

% after getting eartures train SVM and get cross validation error
[ error_train, error_valid ] = trainSVMAndGetError( catFeatures, dogFeatures );

disp('*******************************************************************');
disp('SVM Classifier');
switch featureType
    case 1
        disp('Using intenisty of pixels as features');
    case 2
        disp('Using corners as features');
    case 3
        disp('Using HOG Features');
    case 4
        disp('Uisng FAST Features');
    case 5
        disp('Using SURF Features');
    case 6
        disp('Using SIFT Features');
end

disp(['The training error is ', num2str(error_train)]);
disp(['The validation error is ', num2str(error_valid)]);
disp('*******************************************************************');




