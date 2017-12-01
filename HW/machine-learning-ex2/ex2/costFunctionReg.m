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

[cost, grad_wo] = costFunction(theta, X, y)

J = cost + (sum(theta.^2)-theta(1)*theta(1))*lambda*1.0/(2*m)

grad(1)=grad_wo(1)

for i=2:length(grad)
	grad(i)=grad_wo(i) + lambda*theta(i)*1.0/m
endfor


% =============================================================

end
