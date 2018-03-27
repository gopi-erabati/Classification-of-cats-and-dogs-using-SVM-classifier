function [data_process, data_mean, data_std] = getPreProcess( data , type)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % this function does the preprocessing required for data
% 
% input
%     data                data to be preprocessed
%     type                preproscessing type
%                         1- standardized
%                         2 - tranformed
%                         3 - binarize
%                         
% output
%     data_process        processed data
%     data_mean            mean of data for type1
%     data_std            std of data for type1
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


switch type
    case 1 % to standardise the data
        
        % calculate xi - xbar/sigmai
        
        % to calculate mean of data
        data_mean = mean(data);
        data_mean1 = repmat(data_mean,size(data,1),1);
        
        % to calculate standard deviation of data
        data_std = std(data);
        data_std1 = repmat(data_std,size(data,1),1);
        
        % to subtract mean and divide by std. dev. to standardise data
        data_process = (data - data_mean1)./data_std1;
        
    case 2 % to Transform the features using log(xij + 0.1)
        
        data_process = log(data + 0.1);
        
    case 3 % to Binarize the features using I(xij > 0), i.e.
        % make every feature vector a binary vector
        data_process = data > 0;
end

end

