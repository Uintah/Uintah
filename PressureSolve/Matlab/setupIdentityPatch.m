function [Alist,b] = setupIdentityPatch(grid,k,q,ilower,iupper)
%SETUPIDENTITY  Set the discrete operator at a patch.
%   The usage [Alist,b] = setupOperatorPatch(grid,k,q,ilower,iupper)
%   Set the discrete equations at all interior cells of patch q at
%   level k. The ghost cells are either ignored (if they are on an interior
%   interface) or assigned a boundary condition (if against the domain
%   boundary). We set the equations within the box [ilower,upper] in the
%   patch. On output we return Alist (list of [i j aij]) and the part of b
%   that should be added to a global list Alist to be converted to sparse
%   format, and to a global RHS vector b.
%
%   See also: TESTDFM, SETOPERATOR.

fprintf('--- setupOperatorPatch(k = %d, q = %d) ---\n',k,q);
level       = grid.level{k};
numPatches  = length(level.numPatches);
s           = level.stencilOffsets;
numEntries  = size(s,1);
h           = level.h;
P           = grid.level{k}.patch{q};
map         = P.cellIndex;
Alist       = zeros(0,3);
b           = zeros(prod(P.size),1);

% Prepare a list of all cell indices in this patch including ghosts
all    = cell(grid.dim,1);
for dim = 1:grid.dim
    all{dim} = [ilower(dim)-1:upper(dim)+1] + P.offset(dim);  % Patch-based cell indices including ghosts
end
matAll      = cell(grid.dim,1);
[matAll{:}] = ndgrid(all{:});
pindexAll   = sub2ind(P.size,matAll{:});                      % Patch-based cell indices - list
pindexAll   = pindexAll(:);
pindexRemaining = pindexAll;

% Prepare a list of all interior cells
interior    = cell(grid.dim,1);
for dim = 1:grid.dim
    interior{dim} = [ilower(dim):upper(dim)] + P.offset(dim);  % Patch-based cell indices including ghosts
end
matInterior      = cell(grid.dim,1);
[matInterior{:}] = ndgrid(interior{:});
pindexInterior   = sub2ind(P.size,matInterior{:});                      % Patch-based cell indices - list
pindexInterior  = pindexInterior(:);
mapInterior     = map(pindexInterior);
pindexRemaining = setdiff(pindexRemaining,pindexInterior);

% Domain edges
edgeDomain        = cell(2,1);
edgeDomain{1}     = level.minCell + P.offset;             % First cell next to left domain boundary - patch-based index
edgeDomain{2}     = level.maxCell + P.offset;             % Last Cell next to right domain boundary - patch-based index
% Patch edges
edgePatch          = cell(2,1);
edgePatch{1}       = ilower + P.offset;             % First cell next to left domain boundary - patch-based index
edgePatch{2}       = iupper + P.offset;             % Last Cell next to right domain boundary - patch-based index

%=====================================================================
% Set corresponding ghost cell boundary conditions to identity.
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
        %[ghost{:}]
        matGhost        = cell(grid.dim,1);
        [matGhost{:}]   = ndgrid(ghost{:});
        pindexGhost     = sub2ind(P.size,ghost{:});
        pindexGhost     = pindexGhost(:);
        mapGhost        = map(pindexGhost);
        %mapGhost
        pindexRemaining = setdiff(pindexRemaining,pindexGhost);
        % B.C.: Dirichlet, u=0
        Alist = [Alist; ...
            [mapGhost mapGhost repmat(1.0,size(mapGhost))]; ...
            ];
        b(mapGhost-P.baseIndex+1) = 0.0;                            % This inserts the RHS data into the "chunk" of b that we output from this routine, hence we translate map indices to patch-based 1D indices.

    end
end

%=====================================================================
% Set identity at all interior cells. Ignore ghost cells that are not
% near domain boundaries. They will have their own equations,
% defined elsewhere.
%=====================================================================

Alist = [Alist; ...
    [mapInterior mapInterior repmat(1.0,size(mapInterior))]; ...
    ];
b(mapInterior-P.baseIndex+1) = 0.0;                            % This inserts the RHS data into the "chunk" of b that we output from this routine, hence we translate map indices to patch-based 1D indices.
