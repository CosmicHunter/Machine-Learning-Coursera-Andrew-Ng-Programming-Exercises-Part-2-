function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% Theta1 is of 25 x 401
% Theta2 is of 10 x 26
   X = [ ones(m,1),X];     % X is of 5000x400
   a1 = X;                 %a1 is of 5000x401
   z2 = a1 * Theta1';      % z2 is of 5000 x 25     
   a2 = sigmoid(z2);    
   a2 = [ones(m,1),a2];    % a2 is of 5000 x 26
   z3 = a2 * Theta2';      % z3 is of 5000 x 10
   hth = sigmoid(z3);      % hth is a vector of 5000 x 10 dimension
   
   yvec = zeros(m,num_labels);  % dimension of 5000 x 10
   for i = 1:m
       yvec(i,y(i))=1;
       
   endfor
 
 J = (1/m)*sum(sum(((-yvec.*log(hth))-((1-yvec).*log(1-hth))),2));
 
 temp1 = Theta1(:,2:size(Theta1,2));
 temp2 = Theta2(:,2:size(Theta2,2));
 regularisedTerm = (lambda/(2*m)) *( sum(sum(temp1.^2))+sum(sum(temp2.^2)) );
 J = J + regularisedTerm;

  A1 = X; % 5000 x 401
  
  Z2 = A1 * Theta1';  % m x hidden_layer_size == 5000 x 25
  A2 = sigmoid(Z2); % m x hidden_layer_size == 5000 x 25
  A2 = [ones(size(A2,1),1), A2]; % Adding 1 as first column in z = (Adding bias unit) % m x (hidden_layer_size + 1) == 5000 x 26
  
  Z3 = A2 * Theta2';  % m x num_labels == 5000 x 10
  A3 = sigmoid(Z3); % m x num_labels == 5000 x 10
  
  % h_x = a3; % m x num_labels == 5000 x 10
  
  yVec = zeros(m,num_labels);  % dimension of 5000 x 10
   for i = 1:m
       yVec(i,y(i))=1;
       
   endfor % m x num_labels == 5000 x 10
  
  DELTA3 = A3 - yVec; % 5000 x 10
  DELTA2 = (DELTA3 * Theta2) .* [ones(size(Z2,1),1) sigmoidGradient(Z2)]; % 5000 x 26
  DELTA2 = DELTA2(:,2:end); % 5000 x 25 %Removing delta2 for bias node
  
  Theta1_grad = (1/m) * (DELTA2' * A1); % 25 x 401
  Theta2_grad = (1/m) * (DELTA3' * A2); % 10 x 26
  

 

Theta1_grad_reg_term = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; % 25 x 401
  Theta2_grad_reg_term = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)]; % 10 x 26
  
  %Adding regularization term to earlier calculated Theta_grad
  Theta1_grad = Theta1_grad + Theta1_grad_reg_term;
  Theta2_grad = Theta2_grad + Theta2_grad_reg_term;
  













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
