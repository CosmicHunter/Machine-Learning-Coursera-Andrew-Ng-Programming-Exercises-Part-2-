function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

res = ones(64,3);
ptr = 0;
for Ctest = [0.01 0.03 0.1 0.3 1 3 10 30]
   for sigval = [0.01 0.03 0.1 0.3 1 3 10 30]
     ptr = ptr+1;
     trainedSvmModel = svmTrain(X,y,Ctest,@(x1,x2)gaussianKernel(x1,x2,sigval));
     prediction = svmPredict(trainedSvmModel,Xval);
     perr = mean(double(prediction~=yval));
     
     res(ptr,:) = [Ctest,sigval,perr];
   endfor
endfor

%resSort = sortrows(res,3);
%C = resSort(1,1);
%sigma = resSort(1,2);
[val ind]=min(res(:,3));
C = res(ind,1);
sigma = res(ind,2);




% =========================================================================

end
