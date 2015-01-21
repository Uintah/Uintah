function finish = patch_finish(k,j)
%PATCH_FINISH Return patch "last" cell coordinate.
%   FINISH = PATCH_FINISH(K,J) returns the cell coordinate of the "upper-right"
%   corner (in d-D) of patch number J at level K.
%   
%   See also PATCH_FIND, PATCH_START.

% Author: Oren Livne
%         06/28/2004    Version 1: Created

global_params;

finish          = patch_start(k,j+1) - 1;
last            = find(j == L{k}.patch_num);
finish(last)    = L{k}.cell_num(last);
