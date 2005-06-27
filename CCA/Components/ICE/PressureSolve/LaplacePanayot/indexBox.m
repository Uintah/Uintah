function [indCell,range] = indexBox(P,ilower,iupper)
%INDEXBOX  Index of a box of cells.
%   INDCELL = INDEXBOX(P,ILOWER,IUPPER) returns the cell indices of the
%   D-dimensional box [ILOWER(1):IUPPER(1)] x ... x [ILOWER(D),IUPPER(D)]
%   in patch P.
%
%   See also: INDEXNBHR, SETPATCHINTERIOR.

ind                 = P.cellIndex;                                          % Global indices of patch cells
numDims             = length(size(ind));

%=====================================================================
% Prepare cell array of ranges of patch subscripts of the box
%=====================================================================
range               = cell(numDims,1);
for dim = 1:numDims
    range{dim}      = [ilower(dim):iupper(dim)] + P.offsetSub(dim);     % Patch-based cell indices including ghosts
end

%=====================================================================
% Prepare ndgrid version of range
%=====================================================================
matRange            = cell(numDims,1);
[matRange{:}]       = ndgrid(range{:});

%=====================================================================
% Convert range patch subs to indices
%=====================================================================
indCell             = ind(sub2ind(P.size,matRange{:}));
indCell             = indCell(:);
