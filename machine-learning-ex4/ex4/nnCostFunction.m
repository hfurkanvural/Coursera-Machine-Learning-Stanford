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

X = [ones(m,1), X];  
a1 = X; 
  
z2 = a1 * Theta1'; 
a2 = sigmoid(z2); 
a2 = [ones(size(a2,1),1), a2]; 

z3 = a2 * Theta2';
a3 = sigmoid(z3); 

h_x = a3; % m x num_labels == 5000 x 10
y_Vec = (1:num_labels)==y;
J = (1/m) * sum(sum((-y_Vec.*log(h_x))-((1-y_Vec).*log(1-h_x))));



for t=1:m
    a1 = X(t,:)'; % (n+1) x 1 == 401 x 1
    
    z2 = Theta1 * a1;  % hidden_layer_size x 1 == 25 x 1
    a2 = [1; sigmoid(z2)]; % (hidden_layer_size+1) x 1 == 26 x 1
    z3 = Theta2 * a2; % num_labels x 1 == 10 x 1    
    a3 = sigmoid(z3); % num_labels x 1 == 10 x 1    

    yVector = (1:num_labels)'==y(t); % num_labels x 1 == 10 x 1    
    
    delta3 = a3 - yVector; % num_labels x 1 == 10 x 1    
    delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)]; % (hidden_layer_size+1) x 1 == 26 x 1
    delta2 = delta2(2:end); % hidden_layer_size x 1 == 25 x 1 %Removing delta2 for bias node  
    
    Theta1_grad = Theta1_grad + (delta2 * a1'); % 25 x 401
    Theta2_grad = Theta2_grad + (delta3 * a2'); % 10 x 26
 
end

Theta1_grad = (1/m) * Theta1_grad; % 25 x 401
Theta2_grad = (1/m) * Theta2_grad; % 10 x 26



reg_term = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))); %scalar

J = J + reg_term; %scalar

Theta1_grad_reg_term = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; % 25 x 401
Theta2_grad_reg_term = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)]; % 10 x 26

Theta1_grad = Theta1_grad + Theta1_grad_reg_term;
Theta2_grad = Theta2_grad + Theta2_grad_reg_term;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
