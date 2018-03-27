function error_per = getOnlyError( labels_predicted, labels_truth )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this function is used to compute the error using predicted lables and
% truth labels
% 
% input
%     labels_predicted    predicted labels of data
%     labeks_truth        truth labels of data
%                         
% output
%     error_per           percentage of error 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%to compute error 
error = 0;
for nsamp = 1:length(labels_predicted)
    if labels_predicted(nsamp) ~= labels_truth(nsamp)
        error = error + 1;
    end
end
error_per = error/length(labels_predicted)*100;


end

