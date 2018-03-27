function imageMatrix = getImageMatrix( data )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this function is used to get image matrix for the data
% 
% input
%     data                images matrix data
%                         
% output
%     lambda              optimal lambda
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for nData = 1 : size(data,2)
    imageMatrix1 = vec2mat(data(:, nData), 64);
    imageMatrix(:, :, nData) = imageMatrix1';

end

