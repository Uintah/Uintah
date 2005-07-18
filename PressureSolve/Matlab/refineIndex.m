function indexF = refineIndex(grid,k,indexC)
%REFINEINDEX  Find fine level index from coarse level index.
%   F = COARSENINDEX(GRID,K,C) returns the bottom-left fine level cell index
%   of the coarse level cell containing the index C. C is at level K,
%   F is at level k+1.
%   GRID represents the grid hierarchy, where K=1 is the coarsest and
%   increasing K mean increasingly finer levels.
%
%   See also: COARSENINDEX, SETPATCHINTERFACE.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

if (k == grid.numLevels)
    error('Cannot refine index from finest level');
end

r = grid.level{k+1}.refRatio;
indexF = repmat(r,size(indexC)./size(r)).*(indexC - 1) + 1;

