function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% J
hx = X * theta;
left = hx - y;
leftsquare = left .* left;
right = theta(2:end);
rightsquare = right .* right;
J = (0.5/m)*(sum(leftsquare))+(lambda/(2*m))*(sum(rightsquare));

% gradient

gradleft = left' * X;
gradright = lambda * theta(2:end);
grad = (1/m)*(gradleft' + [0;gradright(:)]);










% =========================================================================

grad = grad(:);

end
