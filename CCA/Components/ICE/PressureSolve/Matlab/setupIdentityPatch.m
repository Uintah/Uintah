function Alist = setupIdentityPatch(grid,k,q,ilower,iupper)
%SETUPIDENTITYPATCH  Set an identity operator at a patch.
%   The usage Alist = setupIdentityPatch(grid,k,q,ilower,iupper)
%   sets the discrete equations at all interior cells of patch q at
%   level k to the identity operator (normally with zero RHS).
%   The ghost cells are either ignored (if they are on an interior
%   interface) or assigned a boundary condition (if against the domain
%   boundary). We set the equations within the box [ilower,iupper] in the
%   patch. On output we return Alist (list of [i j aij]) and the part of b
%   that should be added to a global list Alist to be converted to sparse
%   format. If ghostConnections=1, Alist also
%   includes the connections from ghost cells to interior cells, otherwise
%   does not include that.
%
%   See also: SETOPERATORPATCH, DELETEUNDERLYINGDATA.

fprintf('--- setupIdentityPatch(k = %d, q = %d) ---\n',k,q);
level       = grid.level{k};
numPatches  = length(level.numPatches);
h           = level.h;
P           = grid.level{k}.patch{q};
Alist       = zeros(0,3);
boxSize     = iupper-ilower+3;
boxOffset   = P.offset + P.ilower - ilower;

% Find local map of the box (including ghost cells)
all         = cell(grid.dim,1);
for dim     = 1:grid.dim
    all{dim} = [ilower(dim)-1:iupper(dim)+1] + P.offset(dim);   % Patch-based cell indices including ghosts
end
matAll      = cell(grid.dim,1);
[matAll{:}] = ndgrid(all{:});
pindexAll   = sub2ind(P.size,matAll{:});                        % Patch-based cell indices - list
pindexAll   = pindexAll(:);
map         = P.cellIndex(pindexAll);                           % Cell global indices for the box [ilower-1,iupper+1]
boxBase     = map(1);
b           = zeros(prod(boxSize),1);

% Prepare a list of all cell indices in this box including ghosts
all    = cell(grid.dim,1);
for dim = 1:grid.dim
    all{dim} = [ilower(dim)-1:iupper(dim)+1] + boxOffset(dim);  % Box-based cell indices including ghosts
end
matAll      = cell(grid.dim,1);
[matAll{:}] = ndgrid(all{:});
pindexAll   = sub2ind(boxSize,matAll{:});                       % Box-based cell indices - list
pindexAll   = pindexAll(:);
pindexRemaining = pindexAll;

% Prepare a list of all interior cells
interior    = cell(grid.dim,1);
for dim = 1:grid.dim
    interior{dim} = [ilower(dim):iupper(dim)] + boxOffset(dim);  % Patch-based cell indices including ghosts
end
matInterior      = cell(grid.dim,1);
[matInterior{:}] = ndgrid(interior{:});
pindexInterior   = sub2ind(boxSize,matInterior{:});                      % Patch-based cell indices - list
pindexInterior  = pindexInterior(:);
mapInterior     = map(pindexInterior);
pindexRemaining = setdiff(pindexRemaining,pindexInterior);

% Domain edges
edgeDomain        = cell(2,1);
edgeDomain{1}     = level.minCell + boxOffset;             % First cell next to left domain boundary - patch-based index
edgeDomain{2}     = level.maxCell + boxOffset;             % Last Cell next to right domain boundary - patch-based index
% Patch edges
edgePatch          = cell(2,1);
edgePatch{1}       = ilower + boxOffset;             % First cell next to left domain boundary - patch-based index
edgePatch{2}       = iupper + boxOffset;             % Last Cell next to right domain boundary - patch-based index

%=====================================================================
% Set corresponding ghost cell eqns near domain boundaries to identity.
% (Boundary conditions are now defined at finer level ghost cells.)
%=====================================================================

face = cell(2,1);
for d = 1:grid.dim,                                             % Loop over dimensions of patch
    for side = 1:2                                              % side=1 (left) and side=2 (right) in dimension d
        if (side == 1)
            direction = -1;
        else
            direction = 1;
        end

        [face{:}] = find(matInterior{d} == edgeDomain{side}(d));          % Interior cell indices near DOMAIN boundary
        % Process only if this face is non-empty.
        if (isempty(face{1}))
            continue;
        end

        for dim = 1:grid.dim                                    % Translate face to patch-based indices (from index in the INTERIOR matInterior)
            face{dim} = face{dim} + 1;
        end

        % Construct corresponding ghost cell equations = boundary conditions
        ghost           = face;
        ghost{d}        = face{d} + direction;                      % Ghost cell indices
        ghost{d}        = ghost{d}(1);                              % Because face is an ndgrid-like structure, it repeats the same value in ghost{d}, lengthghostnbhr{d}) times. So shrink it back to a scalar so that ndgrid and flux(...) return the correct size vectors. We need to do that only for dimension d as we are on a face which is grid.dim-1 dimensional object.
        %[ghost{:}]
        matGhost        = cell(grid.dim,1);
        [matGhost{:}]   = ndgrid(ghost{:});
        pindexGhost     = sub2ind(boxSize,matGhost{:});
        pindexGhost     = pindexGhost(:);
        mapGhost        = map(pindexGhost);
        %mapGhost
        pindexRemaining = setdiff(pindexRemaining,pindexGhost);
        % Identity operator
        Alist = [Alist; ...
            [mapGhost mapGhost repmat(1.0,size(mapGhost))]; ...
            ];
    end
end

%=====================================================================
% Set identity at all interior cells. Ignore ghost cells that are not
% near domain boundaries. They retain their own equations,
% defined elsewhere.
%=====================================================================

Alist = [Alist; ...
    [mapInterior mapInterior repmat(1.0,size(mapInterior))]; ...
    ];
