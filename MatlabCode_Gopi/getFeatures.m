function [ catFeatures, dogFeatures ] = getFeatures(cat, dog, featureType)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % this function is ised to compute differnet features of cats and dogs data
% input
%     cat             cat data
%     dog             dog data
%     featureType     type of feature to extract
%
% output
%     catFeatures     features of cat
%     dogFeatures     features of dog
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch featureType
    case 1 % intenisty features
        catFeatures = cat';
        dogFeatures = dog';
        
    case 2 % corner features
        %cat features
        catImageMatrix = getImageMatrix( cat ); % get the image matrices of cat stored ina stack
        
        catFeatures = zeros(size(catImageMatrix(:,:,1))); % intialise cat features matrix
        catRowCounter  = 1; % intialise cat row counter to stack cat features
        
        for nCat = 1 : size(cat, 2)
            corners = detectHarrisFeatures(catImageMatrix(:,:,nCat)); %detect harris corners
            [features, ~] = extractFeatures(catImageMatrix(:,:,nCat), corners, 'Method', 'FREAK'); % extract features at corners
            features = mean(features.Features); % mean the features for one image
            catFeatures(catRowCounter, :) = features; % store features in a stack
            catRowCounter = catRowCounter + 1; % increment row counter
        end
        
        %dog features
        dogImageMatrix = getImageMatrix( dog );
        
        dogFeatures = zeros(size(dogImageMatrix(:,:,1))); % intialise dog features matrix
        dogRowCounter  = 1; % intialise dog row counter to stack cat features
        
        for nDog = 1 : size(dog, 2)
            corners = detectHarrisFeatures(dogImageMatrix(:,:,nDog)); %detect harris corners
            [features, ~] = extractFeatures(dogImageMatrix(:,:,nDog), corners, 'Method', 'FREAK'); % extract features at corners
            features = mean(features.Features); % mean the features for one image
            dogFeatures(dogRowCounter, :) = features; % store features in a stack
            dogRowCounter = dogRowCounter + 1; % increment row counter
        end
        
    case 3 % HOG FEatures
        %cat features
        catImageMatrix = getImageMatrix( cat ); % get the image matrices of cat stored ina stack
        
        catFeatures = []; % intialise cat features matrix
        %         catRowCounter  = 1; % intialise cat row counter to stack cat features
        
        for nCat = 1 : size(cat, 2)
            features = extractHOGFeatures(catImageMatrix(:,:,nCat)); %detect HOG features
            catFeatures = [catFeatures; features]; % store features in a stack
        end
        
        %dog features
        dogImageMatrix = getImageMatrix( dog );
        
        dogFeatures = []; % intialise dog features matrix
        %         dogRowCounter  = 1; % intialise dog row counter to stack cat features
        
        for nDog = 1 : size(dog, 2)
            features = extractHOGFeatures(dogImageMatrix(:,:,nDog)); %detect HOG features
            dogFeatures = [dogFeatures; features]; % store features in a stack
        end
        
    case 4 % using FAST features
        %cat features
        catImageMatrix = getImageMatrix( cat ); % get the image matrices of cat stored ina stack
        
        catFeatures = zeros(size(catImageMatrix(:,:,1))); % intialise cat features matrix
        catRowCounter  = 1; % intialise cat row counter to stack cat features
        
        for nCat = 1 : size(cat, 2)
            corners = detectFASTFeatures(catImageMatrix(:,:,nCat)); %detect harris corners
            [features, ~] = extractFeatures(catImageMatrix(:,:,nCat), corners); % extract features at corners
            features = mean(features.Features); % mean the features for one image
            catFeatures(catRowCounter, :) = features; % store features in a stack
            catRowCounter = catRowCounter + 1; % increment row counter
        end
        
        %dog features
        dogImageMatrix = getImageMatrix( dog );
        
        dogFeatures = zeros(size(dogImageMatrix(:,:,1))); % intialise dog features matrix
        dogRowCounter  = 1; % intialise dog row counter to stack cat features
        
        for nDog = 1 : size(dog, 2)
            corners = detectFASTFeatures(dogImageMatrix(:,:,nDog)); %detect harris corners
            [features, ~] = extractFeatures(dogImageMatrix(:,:,nDog), corners); % extract features at corners
            features = mean(features.Features); % mean the features for one image
            dogFeatures(dogRowCounter, :) = features; % store features in a stack
            dogRowCounter = dogRowCounter + 1; % increment row counter
        end
        
    case 5 % using SURF features
        %cat features
        catImageMatrix = getImageMatrix( cat ); % get the image matrices of cat stored ina stack
        
        catFeatures = zeros(size(catImageMatrix(:,:,1))); % intialise cat features matrix
        catRowCounter  = 1; % intialise cat row counter to stack cat features
        
        for nCat = 1 : size(cat, 2)
            corners = detectSURFFeatures(catImageMatrix(:,:,nCat)); %detect harris corners
            [features, ~] = extractFeatures(catImageMatrix(:,:,nCat), corners, 'Method', 'SURF'); % extract features at corners
            features = mean(features); % mean the features for one image
            catFeatures(catRowCounter, :) = features; % store features in a stack
            catRowCounter = catRowCounter + 1; % increment row counter
        end
        
        %dog features
        dogImageMatrix = getImageMatrix( dog );
        
        dogFeatures = zeros(size(dogImageMatrix(:,:,1))); % intialise dog features matrix
        dogRowCounter  = 1; % intialise dog row counter to stack cat features
        
        for nDog = 1 : size(dog, 2)
            corners = detectSURFFeatures(dogImageMatrix(:,:,nDog)); %detect harris corners
            [features, ~] = extractFeatures(dogImageMatrix(:,:,nDog), corners, 'Method', 'SURF'); % extract features at corners
            features = mean(features); % mean the features for one image
            dogFeatures(dogRowCounter, :) = features; % store features in a stack
            dogRowCounter = dogRowCounter + 1; % increment row counter
        end
        
    case 6 % Using SIFT FEatures
        %cat features
        catImageMatrix = getImageMatrix( cat ); % get the image matrices of cat stored ina stack
        
        catFeatures = zeros(size(catImageMatrix(:,:,1),1),128); % intialise cat features matrix
        catRowCounter  = 1; % intialise cat row counter to stack cat features
        
        for nCat = 1 : size(cat, 2)
            features = SIFT(catImageMatrix(:,:,nCat)); %detect SIFT features
            features = mean(features,2); % mean the features for one image
            catFeatures(catRowCounter, :) = features'; % store features in a stack
            catRowCounter = catRowCounter + 1; % increment row counter
        end
        
        %dog features
        dogImageMatrix = getImageMatrix( dog );
        
        dogFeatures = zeros(size(dogImageMatrix(:,:,1),1),128); % intialise dog features matrix
        dogRowCounter  = 1; % intialise dog row counter to stack cat features
        
        for nDog = 1 : size(dog, 2)
            features = SIFT(dogImageMatrix(:,:,nDog)); %detect SIFT features
            features = mean(features,2); % mean the features for one image
            dogFeatures(dogRowCounter,:) = features'; % store features in a stack
            dogRowCounter = dogRowCounter + 1; % increment row counter
        end
        
        
        
end

