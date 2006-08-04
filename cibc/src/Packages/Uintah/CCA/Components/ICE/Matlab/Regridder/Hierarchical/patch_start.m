function start = patch_start(k,j)
%PATCH_START Return patch "first" cell coordinate.
%   FINISH = PATCH_START(K,J) returns the cell coordinate of the "lower-left"
%   corner (in d-D) of patch number J at level K.
%   
%   See also PATCH_FIND, PATCH_FINISH.

% Author: Oren Livne
%         06/28/2004    Version 1: Created

global_params;

n = size(j,1);
if (n == 0)
    start = zeros(size(j));
else
    start   = (j-o  ).*repmat(L{k}.patch_size,n,1) + o;
end
