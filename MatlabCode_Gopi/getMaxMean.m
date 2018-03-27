function [ data_max, data_mean ] = getMaxMean( data, feature )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % This function calculates the max and mean of data of a given feature
% 
% input
%     data                data with feuture
%     feature             to get mean and max of which feature
%                         
% output
%     data_max            maximum data
%     data_mean           mean data
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% for max 
data_max = max(data(:,feature));

% for mean
data_mean = mean(data(:,feature));

end

