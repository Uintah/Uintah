function b = cascade(a)
%CASCSDE Dilate a binary image of flagged cells into a "cascaded form".
%   B = CASCADE(A) returns a "cascaded dilated image" of A (a convlution with a filter
%   that determines the local connections. We use a 5-point filter in 2D, 7-point in 3D).
%   Instead of 1's in A, B contains in those cells the number of A-neighbours of such an A-cell.
%   (CASCADE(A)>0) is the standard dilation operator (one "safety layer") on the binary A.
%
%   See also CONV2, TEST_CASE, TEST_MOVEMENT.
 
% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments.

[m,n]   = size(a);                                              % Sizes of a
z       = find(a == 0);                                         % Location of 0's in a
%f       = [[1 1 1];[1 1 1];[1 1 1]];                            % Filter (9-point)
f       = [[0 1 0];[1 1 1];[0 1 0]];                            % Filter (5-point)
bbig    = conv2(a,f);                                           % Convolve A with the filter, so bbig is replaced by the sum of neighouring 1's, neighbours specified by f
b       = bbig(2:m+1,2:n+1);                                    % The convolution makes a bigger b than we need
%b(z)    = 0;                                                    % All the non-zeros are the info in bbig, just set the zeros of A back to their place.
