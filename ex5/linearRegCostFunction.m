function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
yPredicted = X * theta;
temp = yPredicted - y;

grad = sum(temp .* X) ./ m;
Theta = theta;
Theta(1) = 0;
grad = grad' + lambda/m .* Theta;
temp = temp .^ 2;
J = sum(temp)/(2*m);
J = J + lambda/(2*m) * sum(theta(2:end) .^ 2); 














grad = grad(:);

end
