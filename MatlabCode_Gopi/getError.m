function error_per = getError( data, labels, regs_para)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % this function is used to get error when data and labels are given
% 
% input
%     data                data with feaures
%     labels              labels of data
%     regs_para           regression parameters
%                         
% output
%     error_per           percentage of error 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% to predict the output of validation data using estimated parameters
[y_pred] = predict(data, regs_para);

y_pred = y_pred > 0.5; % to check and assign a class for data

%to compute error of validation data
error = 0;
for nsamp = 1:length(labels)
    if labels(nsamp) ~= y_pred(nsamp)
        error = error + 1;
    end
end
error_per = error/length(labels)*100;
end

