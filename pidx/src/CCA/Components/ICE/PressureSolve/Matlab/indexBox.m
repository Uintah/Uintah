function [indCell,range,matRange] = indexBox(P,ilower,iupper,ind)
%INDEXBOX  Index of a box of cells.
%   [INDCELL,R,M] = INDEXBOX(P,ILOWER,IUPPER,TYPE) returns the cell indices of the
%   D-dimensional box [ILOWER(1):IUPPER(1)] x ... x [ILOWER(D),IUPPER(D)]
%   in patch P. R is a cell array of the box's ranges, and M is the same as
%   R, but in "ndgrid format" where all ranges are d-dimensional arrays.
%   Both R and M subscripts are PATCH-BASED.
%
%   See also: INDEXNBHR, SETPATCHINTERIOR.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

if (nargin < 4)
    ind                 = P.cellIndex;                                          % Global indices of patch cells
end
numDims             = length(ilower);

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
[matRange{:}]       = myndgrid(range{:});

%=====================================================================
% Convert range patch subs to indices
%=====================================================================
indCell             = ind(mysub2ind(P.size,matRange{:}));
indCell             = indCell(:);
