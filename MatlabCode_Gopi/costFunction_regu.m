function [e grad] = costFunction_regu(X, y, w, lambda)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% this function computes error and gradient given features, output and parameters
% 
% input
%     X               data with features
%     y               labels of data
%     w               regression parameters
%     lambda          regularization parameter
%                         
% output
%     e               Error
%     grad            gradient
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

E = 0;
grad = zeros(size(X)) ;
rowCounter = 1;

% run it for all samples
for i = 1:size(X,1)
    phi = X(i,:);
    phi_n = sigmoid(w'*phi');
    E = E + (y(i)*log(phi_n)+(1-y(i))*log(1-phi_n)); 
    grad(rowCounter,:) = [(phi_n-y(i))*X(i,:)+lambda.*[0 w(2:end)']];
    rowCounter = rowCounter + 1; % increment row counter
end

grad = sum(grad)/length(y);
e = -1 * E/length(y) ;
e = e + lambda/2*sum(w(2:end).^2);
end

