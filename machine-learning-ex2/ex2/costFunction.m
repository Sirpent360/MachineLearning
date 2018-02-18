function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% z=transpose(theta)*X;			%	z for g(z)

predict = sigmoid(X*theta);			% our hypothesis with using of sigmoid

% hypothesis = 1./(1+exp(-z));	%	hypothesis with sigmoid function

red = transpose(-y)*log(predict);		%	The red-circled term 

blue = transpose(1-y)*log(1-predict);%	The blue-circled term 

substract = red - blue;			%	Subtract the right-side term from the left-side term

J = (1/m) * substract;			%	Scale the result by 1/m. This is the unregularized cost.

grad = transpose(X)*(predict-y)*1/m;	%	gradient calculation; vector product also includes the required summation.






% =============================================================

end
