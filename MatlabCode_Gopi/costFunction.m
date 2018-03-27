function [e grad] = costFunction(X,y,w)
%% this function computes error and gradient given features, output and parameters

E = [];
grad = [];
X = [ones(size(X,1),1) X];

% run it for all samples
for i = 1:size(X,1)
    phi = X(i,:);
    phi_n = sigmoid(w'*phi');
    E = [E y(i)*log(phi_n)+(1-y(i))*log(1-phi_n)]; 
    grad = [grad; (phi_n-y(i))*X(i,:)];
end

grad = sum(grad)/length(y);
e = -1 * sum(E)/length(y);
end
    
    