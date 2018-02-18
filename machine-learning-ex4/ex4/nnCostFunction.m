function [J grad a2 a3] = nnCostFunction(nn_params, ...
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

%%% Part 1

y_matrix = eye(num_labels)(y,:); % 1 - Expand the 'y' output values into a matrix of single values (see ex4.pdf Page 5). This is most easily done using an eye() matrix of size num_labels, with vectorized indexing by 'y'. 


a1 = [ones(m,1), X];						%	a1 equals the X input matrix with a column of 1's added (bias units) as the first column

a2 = [ones(m,1), sigmoid(a1 * Theta1')];	%	z2 equals the product of a1 and Θ1 ...
											%	a2 is the result of passing z2 through g()
											%	Then add a column of bias units to a2 (as the first column).
											%	NOTE: Be sure you DON'T add the bias units as a new row of Theta.	


a3 = sigmoid(a2 * Theta2');					%	z3 equals the product of a2 and Θ2
											%	a3 is the result of passing z3 through g()


J = 1/m * sum(sum(-y_matrix.*log(a3)-(1-y_matrix).*log(1-a3)));


%	Compute the unregularized cost according to ex4.pdf (top of Page 5), using a3, your y_matrix, and m (the number of training examples). Note that the 'h'
%	argument inside the log() function is exactly a3. Cost should be a scalar value. Since y_matrix and a3 are both matrices, you need to compute the double-sum.
%	Remember to use element-wise multiplication with the log() function.


regterms = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));

J = J + regterms;

%	4 - Compute the regularized component of the cost according to ex4.pdf Page 6, using Θ1 and Θ2 (excluding the Theta columns for the bias units), 
%	along with λ, and m. The easiest method to do this is to compute the regularization terms separately, then add them to the unregularized cost from Step 3. 

%	I am working with matrix but operating with
%	1) All rows
%	2) But only columns from second to last (without first which is bias unit)



%%% Part 2: 

for t = 1:m							%	We recommend implementing backpropagation using a for-loop
									%	over the training examples if you are implementing it for the 
									%	first time

	% l=1, for the input layer :
	a1 = [1; X(t,:)'];				%	Set the input layer’s values (a(1)) to the t-th training example x(t)
									%	adding bias unit to a1, and transpose it for later multiplication

	% l=2, for the hidden layers :
	z2 = Theta1 * a1;				%	computing the activations (z(2); a(2); z(3); a(3))
									%	for layers 2 and 3.
	a2 = [1; sigmoid(z2)];			%	adding bias unit to a2

	z3 = Theta2 * a2;				%	
	a3 = sigmoid(z3);				%	

	ylog = (1:num_labels==y(t))';	%	obtain 10×1 logical array from row vector by transposition 
									%	with 1 at position with the same value (5 => 1 at 5 position)
									%	(0 at 10 position)
									%	compare array from 1 to 10 to current y at t position (which from first to m row)
									%	
	%	size(yy)					%	check dimensions
	
	% For the delta values:
	delta_3 = a3 - ylog;

	%delta_2 = (Theta2(:,2:end)' * delta_3) .* sigmoidGradient(z2);
	delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)];
									%	d2 is the product of d3 and Theta2(no bias), then element-wise scaled by sigmoid gradient of z2
									
	delta_2 = delta_2(2:end);		%	removing delta_0
	
									%	delta_1 is not calculated because input doesnt had error    

	% Big delta update
	Theta1_grad = Theta1_grad + delta_2 * a1';
	Theta2_grad = Theta2_grad + delta_3 * a2';
end

Theta1_grad = (1/m) * Theta1_grad;	%	Theta1_grad and Theta2_grad are the same size as their respective Deltas, just scaled by 1/m.
Theta2_grad = (1/m) * Theta2_grad;	%	Now you have the unregularized gradients. 


Theta1wOutBias = Theta1;
Theta2wOutBias = Theta2;
Theta1wOutBias(:,1)=0;						%	set the first column of Theta1 and Theta2 to all-zeros
Theta1wOutBias(:,1)=0;						%
Theta1reg=(lambda/m)*Theta1wOutBias;		%	Scale each Theta matrix by λ/m
Theta2reg=(lambda/m)*Theta2wOutBias;		%	

Theta1_grad = Theta1_grad + Theta1reg;	%	Add each of these modified-and-scaled Theta matrices to the un-regularized Theta gradients that you computed earlier.
Theta2_grad = Theta2_grad + Theta2reg;	%	




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
