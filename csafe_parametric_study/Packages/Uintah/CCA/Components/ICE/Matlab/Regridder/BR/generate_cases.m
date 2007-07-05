%GENERATE_CASES Generate test cases for the clustering algorithm.
%   This script generates several test cases (binary matrices of flagged cells)
%   for the clustering algorithm CREATE_CLUSTER. See the directory ../TESTCASES
%   (w.r.t. clustering algorithm code directory) for various test cases' input files.
%
%   See also CREATE_CLUSTER.
 
% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% HARD TEST CASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Test Case H1: The one appearing in Fig. 18, [BG91] paper
%save hard1 flag -ascii

%%%% Test Case H2: two blocks in a diagonal constellation
flag = zeros(18,12);
flag(1:9,1:6) = 1
flag(10:18,7:12) = 1
flag
save hard4 flag -ascii

%%%% Test Case H3: red-black grid 11x11
flag = zeros(11);
flag
flag(1:2:prod(size(flag))) = 1;
flag
help save
save hard3 flag -ascii

%%%% Test Case H4: randomly (Gaussian) distributed cells with sigma=3, around mid-point of the array

flag = zeros(30);
sigma = 3;
center = [15 15];
random_tests = 100;
x = round(sigma*randn(random_tests,1) + center(1));
y = round(sigma*randn(random_tests,1) + center(2));

for i = 1:random_tests,
    flag(x(i),y(i)) = 1;
end
