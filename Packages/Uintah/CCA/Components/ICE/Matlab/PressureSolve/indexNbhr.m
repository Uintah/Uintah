function indNbhr = indexNbhr(P,indCell,n,ind)
%INDEXNBHR  Index of neighbouring cells.
%   INDNBHR = INDEXNBHR(P,INDCELL,N) returns the cell indices of the
%   neighbours of indices INDCELL in patch P, in the normal direction N.
%   For instance, if N=(1,0), INDNBHR are the x-right neighbors of INDCELL.
%
%   See also: INDEXBOX, SETPATCHINTERIOR.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

if (nargin < 4)
    ind                 = P.cellIndex;                                          % Global indices of patch cells
end
numDims             = length(P.ilower);

%=====================================================================
% Convert indCell to patch subscripts
%=====================================================================
subCell             = cell(numDims,1);
[subCell{:}]        = myind2sub(P.size,indCell - P.offsetInd);

%=====================================================================
% Compute nbhr patch subscripts
%=====================================================================
subNbhr             = cell(numDims,1);
for dim = 1:numDims
    subNbhr{dim}    = subCell{dim} + n(dim);
end

%=====================================================================
% Convert nbhr patch subs to indices
%=====================================================================
indNbhr             = ind(mysub2ind(P.size,subNbhr{:}));
indNbhr             = indNbhr(:);
