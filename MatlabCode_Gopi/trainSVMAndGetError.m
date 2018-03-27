function [ error_train, error_valid ] = trainSVMAndGetError( cat, dog )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % this function is to train SVM and get error of cross validation
% input
%     cat             cat data features
%     dog             dog data features
%
% output
%     error_train      training error
%     error_valid      validation error
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% select 80% of data to train and 20% to test for cross validation
index80 = ceil(size(cat, 1) * 0.8);
remainData = size(cat, 1) - index80; %remaining data

% train data and labels
data_train = [cat(1:index80, :); dog(1:index80, :)];
data_train_lab = [ones(index80,1); zeros(index80,1)];

% test data and labels
data_test = [cat(index80+1:end,:) ; dog(index80+1:end, :)];
data_test_lab = [ones(remainData, 1); zeros(remainData, 1)];

% train SVM
classNames = [ 1 0];
kernelFunction = 'gaussian';
kernelScale = 'auto';

modelSVM = fitcsvm(double(data_train), data_train_lab, 'KernelFunction' , kernelFunction, 'KernelScale' , kernelScale, 'ClassNames' , classNames, 'Standardize' , true);

% predict the validation test set and get error
test_lab_svm = predict(modelSVM, double(data_test));
error_valid = getOnlyError(test_lab_svm, data_test_lab);

%predict train set and get error
train_lab_svm = predict(modelSVM, double(data_train));
error_train = getOnlyError(train_lab_svm, data_train_lab);



end

