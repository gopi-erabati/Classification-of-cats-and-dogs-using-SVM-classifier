function  lambda  = getlambda( data_train, data_train_lab )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %this function is used to get lambda using cross validation
% cross validation data division of training data
% 80% for training and 20% for validation
% 
% input
%     data_train          training data
%     data_train_lab      training data labels
%
%                         
% output
%     lambda              optimal lambda
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_train = ceil(0.8 * size(data_train,1)); % to select 80% of data
data_train_train = data_train(1:num_train,:); % first 80% as training data
data_train_valid = data_train(num_train+1:end,:); % last 20% as validate data
data_train_lab_train = data_train_lab(1:num_train,:); % first 80% as training data labels
data_train_lab_valid = data_train_lab(num_train+1:end,:);% last 20% as validate data labels

error_per = zeros(1,11); % to allocate error values
error_idx = 1; % index used to store error of validation
% lambda_init= linspace(0,0.05,11); % to parametr sweep lambda
% lambda_init = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 50];
lambda_init = linspace(0,0.1,11);

% to parametr sweep lambda to check for minimum validation error
for lambda = lambda_init
    
    disp(['getting error for lambda = ',num2str(lambda)])
    
    % to get regression parameters using optimisation
    regs_para = getRegPara( data_train_train, data_train_lab_train, lambda );
    
    % to predict the output of validation data using estimated parameters
    % above
    [y_pred] = predict(data_train_valid, regs_para);
    
    y_pred = y_pred > 0.5; % to check and assign a class for data
    
    
    % to predict the train set
    [train_pred] = predict(data_train_train, regs_para);
    
    train_pred = train_pred > 0.5;  % to check and assign a class for data
    
    % get train and valid errors
    train_error(error_idx) = getOnlyError(train_pred, data_train_lab_train);
    
    valid_error(error_idx) = getOnlyError(y_pred, data_train_lab_valid);
    

    error_idx = error_idx + 1;
end
% to check for minimum error and corresponding lambda wil be output
train_error
valid_error
[~,idx] = min(valid_error)
lambda = lambda_init(idx);
figure
plot(lambda_init,train_error,'r',lambda_init, valid_error,'b');
xlabel('lambda');ylabel('error (in %)');title('training/validation error vs lambda');
legend('train error','validation error');
end
