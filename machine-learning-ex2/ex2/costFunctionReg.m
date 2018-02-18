function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

predict = sigmoid(X*theta);				%	our hypothesis with using of sigmoid

red = transpose(-y)*log(predict);		%	The red-circled term 

blue = transpose(1-y)*log(1-predict);	%	The blue-circled term 

substract = red - blue;					%	Subtract the right-side term from the left-side term

J = (1/m) * substract;					%	Scale the result by 1/m. This is the unregularized cost.

grad = transpose(X)*(predict-y)*1/m;	%	gradient calculation; vector product also includes the required summation.


% Regularized part

% regularized part for J

theta(1)=0;								%	set theta(1) to zero

theta_sum = transpose(theta)*theta;		%	calculate the sum of the squares of theta

reg_part=(lambda/(2*m))*theta_sum;		%	scale the cost regularization term by (lambda / (2 * m))

J = J + reg_part;						%	add your unregularized and regularized cost terms together

% regularized part for gradient

grad_reg=theta*(lambda/m);				%	calculate the regularized gradient term as theta scaled by (lambda / m)

grad = grad + grad_reg;

% =============================================================

end
