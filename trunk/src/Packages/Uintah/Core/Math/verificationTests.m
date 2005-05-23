% This file is a Matlab script that contains
% verification tests to compare against
% for the different math functions in this directory.
%________________________________________
%  M A T R I X   T E S T S
%  2 x 2 matrix  solve the ill conditioned Hilbert matrix
%  If you use the 
fprintf('_________________________2 X 2 \n\n');
A = hilb(2)
X = ones(2,1)
B = A * X

fprintf(' Back out X  all entries of X should be 1.0\n')
A_inverse = inv(A)
x = A_inverse * B
%________________________________________
%  3 x 3 matrix
fprintf('_________________________3 X 3 \n\n');
A = hilb(3)
X = ones(3,1)
B = A * X

fprintf(' Back out X  all entries of X should be 1.0\n')
A_inverse = inv(A)
x = A_inverse * B
%________________________________________
%  4 x 4 matrix
fprintf('_________________________4 X 4 \n\n');
A = hilb(4)
X = ones(4,1)
B = A * X

fprintf(' Back out X  all entries of X should be 1.0\n')
A_inverse = inv(A)
x = A_inverse * B
%________________________________________
%  5 x 5 matrix
fprintf('_________________________5 X 5 \n\n');
A = hilb(5)
X = ones(5,1)
B = A * X

fprintf(' Back out X  all entries of X should be 1.0\n')
A_inverse = inv(A)
x = A_inverse * B
%________________________________________
%    6 X 6 matrixs
fprintf('_________________________6 X 6 \n\n');
A = hilb(6)
X = ones(6,1)
B = A * X

fprintf(' Back out X  all entries should of X should be 1.0\n')
A_inverse = inv(A)
x = A_inverse * B