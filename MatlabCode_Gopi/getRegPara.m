function regs_para = getRegPara( data, labels, lambda )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % this function is used to get parameters by optmising the cost function
% 
% input
%     data                data with feaures
%     labels              labels of data
%     lambda              regularization parameters
%                         
% output
%     regs_para          regression parameters 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


regs_para = zeros(size(data,2),1); % to intialise the parameters
%     options = optimoptions('fminunc','Display','iter','GradObj','on','MaxIter',400);
options = optimoptions('fminunc','GradObj','on','MaxIter',400);
% function to minimise cost function of logistic regression and find
% parameters

[regs_para, ~] = fminunc(@(regs_para)costFunction_regu(data, labels, regs_para, lambda), regs_para, options);



end

