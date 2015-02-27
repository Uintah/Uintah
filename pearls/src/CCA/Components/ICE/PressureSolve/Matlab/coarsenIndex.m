function indexC = coarsenIndex(grid,k,indexF)
%COARSENINDEX  Find coarse level index from fine level index.
%   C = COARSENINDEX(GRID,K,F) returns the coarse level index of the coarse
%   level cell containing the index F. F is at level K, C is at level k-1.
%   GRID represents the grid hierarchy, where K=1 is the coarsest and
%   increasing K mean increasingly finer levels.
%
%   See also: REFINEINDEX, SETPATCHINTERFACE.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

if (k == 1)
    error('Cannot coarsen index from coarsest level');
end

r = grid.level{k}.refRatio;
indexC = ceil(indexF./repmat(r,size(indexF)./size(r)));
