function [indCell,range,matRange] = indexBox(P,ilower,iupper,type)
%INDEXBOX  Index of a box of cells.
%   [INDCELL,R,M] = INDEXBOX(P,ILOWER,IUPPER,TYPE) returns the cell indices of the
%   D-dimensional box [ILOWER(1):IUPPER(1)] x ... x [ILOWER(D),IUPPER(D)]
%   in patch P. R is a cell array of the box's ranges, and M is the same as
%   R, but in "ndgrid format" where all ranges are d-dimensional arrays.
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
