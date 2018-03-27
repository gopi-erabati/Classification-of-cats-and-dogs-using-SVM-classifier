function test_lab_naive = getNaiveBayes( data_train, data_train_lab, data_test )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this function is used to get Naive Bayes classification when given a test
% data sample 
% %%%%%%%%%%%%%%%%%%%%%%%% MY IMPLEMENTATION %%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% input data_train        - training data
%       data_train_lab    - training data labels
%       data_test         - test data to classify
%       
% output      
%       label             - laebl of test data after classification
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get the probabilities of two classes Spam and Notspam
prob_spam = nnz(data_train_lab)/numel(data_train_lab);
prob_notspam = 1 - prob_spam;


% get the mean and standrad deviation of training fetures for class Spam
spam_indices = find(data_train_lab);  % get spam indices
mean_spam = mean(data_train(spam_indices,:)); % do mean and std for spam data
std_spam = std(data_train(spam_indices,:));

notspam_indices  =find(~data_train_lab); % get notspam indices
mean_notspam = mean(data_train(notspam_indices,:)); % do mean and std for spam data
std_notspam = std(data_train(notspam_indices,:));

% get the probabilities of each feature for class 1 Spam of test data
prob_feature_spam_final = [];
for ndata = 1 : size(data_test,1)
    prob_feature_spam = getProbabilityFeature( data_test(ndata, :), mean_spam, std_spam );
    prob_feature_spam_final = [prob_feature_spam_final ; prod(prob_feature_spam) * prob_spam];
end

% get the probabilities of each feature for class 2 NotSpam of test data
prob_feature_notspam_final = [];
for ndata = 1 : size(data_test,1)
    prob_feature_notspam = getProbabilityFeature( data_test(ndata,:), mean_notspam, std_notspam );
    prob_feature_notspam_final = [prob_feature_notspam_final ; prod(prob_feature_notspam) * prob_notspam];
end

% check for higher probabilities to asiign it to a class
test_lab_naive =  prob_feature_spam_final > prob_feature_notspam_final;

end

