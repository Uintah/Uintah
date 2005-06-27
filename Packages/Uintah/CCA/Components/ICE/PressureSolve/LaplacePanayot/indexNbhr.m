function indNbhr = indexNbhr(P,ind,d,s)
%INDEXNBHR  Index of neighbouring cells.
%   INDNBHR = INDEXNBHR(P,IND,D,S) returns the cell indices of the
%   neighbours of indices IND in patch P, in dimension D (between 1 and
%   numDims) and direction S (-1 if left nbhrs, +1 if right nbhrs).
%
%   See also: SETPATCHINTERIOR.

ind                 = P.cellIndex;                                          % Global indices of patch cells
numDims             = length(size(ind));

%=====================================================================
% Convert ind to patch subscripts
%=====================================================================
sub                 = cell(numDims,1);
[sub{:}]            = ind2sub(P.size,ind - P.offsetInd);

%=====================================================================
% Compute normal direction
%=====================================================================
% Outward normal direction
nbhrOffset          = zeros(1,numDims);
nbhrOffset(d)       = s;

%=====================================================================
% Compute nbhr patch subscripts
%=====================================================================
subNbhr             = cell(numDims,1);
for dim = 1:numDims
    subNbhr{dim}    = subNbhr{dim} + nbhrOffset(d);
end

%=====================================================================
% Convert nbhr patch subs to indices
%=====================================================================
indNbhr             = ind(subNbhr{:});
