function nbhr = nbhrIndex(P,c,d,s)
%NBHRINDEX  Cell index of neighbor in the d-dimension.
%   NBHR = NBHRINDEX(P,C,D,S) returns the global variable index of the
%   neighbor of C in patch P in the D-dimension and direction S. P is a
%   rectangular patch structure, CELL can be a vector of length Nx1, D is
%   between 1 and numDims (=number of dimensions), and S can be -1 (left
%   neighbor) or +1 (right neighbor). NBHR is returned as an Nx1 vector.
%
%   See also: TESTDFM, SETOPERATORPATCH, DELETEUNDERLYINGDATA.
global verboseLevel

if (verboseLevel >= 1)
    fprintf('--- nbhrIndex ---\n');
end

map             = P.cellIndex;
dim             = length(size(map));

% Convert variable indices c to map subscripts
cSub            = cell(dim,1);
[cSub{:}]       = find(map);

% Offset to dimension d, direction s
nbhrOffset      = zeros(1,dim);
nbhrOffset(d)   = s;
[face{:}]       = find(matInterior{d} == edgePatch{side}(d));          % Interior cell indices near PATCH boundary
for dim = 1:dim                                    % Translate face to patch-based indices (from index in the INTERIOR matInterior)
    face{dim} = face{dim} + 1;
end

% Compute subscripts of nbhr
nbhrSub = cell(dim,1);
for dim = 1:dim
    nbhrSub{dim} = cSub{dim} + nbhrOffset(dim);
end

% Compute variable indices of nbhr
matNbhr         = cell(dim,1);
[matNbhr{:}]    = ndgrid(nbhr{:});
pindexNbhr      = sub2ind(boxSize,matNbhr{:});                      % Patch-based cell indices - list
pindexNbhr      = pindexNbhr(:);
nbhr            = map(pindexNbhr);
