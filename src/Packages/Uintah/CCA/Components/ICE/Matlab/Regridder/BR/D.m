function b = D(a)
%D Display an array with the "normal alignment" as it is written on a page.
%
%   B = D(A) will print the m x n two-dimensional array A flipped-up down and transposed, so that
%   the lowest-left corner of B (when printed in MATLAB) is A(1,1), and the upper-right corner of
%   B is A(m,n).
%
%   See also FLIPUD.
  
% Author: Oren Livne
% Date  : 05/27/2004    Version 1: created and added comments.
 

Display an array with the "normal alignment" as it is written on a page.
b = flipud(a');
