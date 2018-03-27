function prob_feature = getProbabilityFeature( feature, mean1, std1 )
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this function gives the probability of a feature of a given class
% 
% input
%     feature             feature to find propbability
%     mean1               mean of train data
%     std1                std of train data
%                         
% output
%     prob_feature           Probability of feature 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

prob_feature = exp((-(feature - mean1).^2)./(2.*std1.^2))./(sqrt(2 * pi) .* std1);


end

