function [ y ] = predict( X,w )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % test function to predict whether a candidate can be admitted or not
% 
% input
%     X                   data with feaures
%     w                   regression parameters
%                         
% output
%     y                   predicted label 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y = sigmoid(w'*X');

end

