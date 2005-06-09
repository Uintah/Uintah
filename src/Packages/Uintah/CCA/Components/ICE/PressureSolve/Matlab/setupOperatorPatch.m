function [Alist,b] = setupOperatorPatch(grid,k,q)
% Set the discrete equations at all interior cells of patch q at
% level k. The ghost cells are either "ghost" or boundary values.

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
    all{dim} = [P.ilower(dim)-1:P.iupper(dim)+1] + P.offset(dim);  % Patch-based cell indices including ghosts
end
matAll      = cell(grid.dim,1);
[matAll{:}] = ndgrid(all{:});
pindexAll   = sub2ind(P.size,matAll{:});                      % Patch-based cell indices - list
pindexAll   = pindexAll(:);
pindexRemaining = pindexAll;

% Prepare a list of all interior cells
interior    = cell(grid.dim,1);
for dim = 1:grid.dim
    interior{dim} = [P.ilower(dim):P.iupper(dim)] + P.offset(dim);  % Patch-based cell indices including ghosts
end
matInterior      = cell(grid.dim,1);
[matInterior{:}] = ndgrid(interior{:});
pindexInterior   = sub2ind(P.size,matInterior{:});                      % Patch-based cell indices - list
pindexInterior  = pindexInterior(:);
mapInterior     = map(pindexInterior);
pindexRemaining = setdiff(pindexRemaining,pindexInterior);

edge        = cell(2,1);
edge{1}     = level.minCell + P.offset;             % First cell next to left domain boundary - patch-based index
edge{2}     = level.maxCell + P.offset;             % Last Cell next to right domain boundary - patch-based index

%=====================================================================
% Construct interior fluxes
%=====================================================================
% Flux vector of -Laplace operator.
% Format: [west east north south] (west=2.0 means
% 2.0*(uij-u_{i,j-1}), for instance)
% rhsFactor multiplies the RHS average of the cell in the
% discretization.

% Fluxes for interior cells not near boundaries: [1 1 1 1]
flux = zeros([P.size 2*grid.dim]);
for direction = 1:2*grid.dim
    flux(interior{:},direction) = 1;
end
rhsValues = zeros(P.size);
rhsValues(interior{:}) = prod(h) * ...
    rhs(([P.ilower(1):P.iupper(1)]-0.5)*level.h(1),([P.ilower(2):P.iupper(2)]-0.5)*level.h(2));

%=====================================================================
% Correct fluxes near DOMAIN boundaries (not necessarily patch
% boundaries)
%=====================================================================

face = cell(2,1);
for d = 1:grid.dim,                                             % Loop over dimensions of patch
    for side = 1:2                                              % side=1 (left) and side=2 (right) in dimension d
        if (side == 1)
            direction = -1;
        else
            direction = 1;
        end

        [face{:}] = find(matInterior{d} == edge{side}(d));          % Interior cell indices near domain boundary
        for dim = 1:grid.dim                                    % Translate face to patch-based indices (from index in the INTERIOR matInterior)
            face{dim} = face{dim} + 1;
        end

        % Correct flux at face
        flux(face{:},2*d+side-2) = 2*flux(face{:},2*d+side-2);                    % twice smaller spacing for flux in this direction
        otherDim = setdiff([1:2*grid.dim],[2*d-1,2*d]);
        for dim = otherDim
            flux(face{:},dim) = 0.75*flux(face{:},dim);     % 3/4 the boundary of CV in the perpendicular directions
        end
        rhsValues(face{:}) = 0.75*rhsValues(face{:});           % 3/4 less volume of CV ==> correct RHS average
    end
end

%=====================================================================
% Construct stencil coefficients from fluxes flux-based, for
% interior cells and ghost cells (boundary conditions)
%=====================================================================

for d = 1:grid.dim,                                             % Loop over dimensions of patch
    %d
    for side = 1:2                                              % side=1 (left) and side=2 (right) in dimension d
        if (side == 1)
            direction = -1;
        else
            direction = 1;
        end
        %direction
        % Outward normal direction
        nbhrOffset = zeros(1,grid.dim);
        nbhrOffset(d) = direction;
        
        [face{:}] = find(matInterior{d} == edge{side}(d));          % Interior cell indices near domain boundary
        for dim = 1:grid.dim                                    % Translate face to patch-based indices (from index in the INTERIOR matInterior)
            face{dim} = face{dim} + 1;
        end

        % Add fluxes to list of non-zeros Alist and to b
        nbhr = cell(grid.dim,1);
        for dim = 1:grid.dim
            nbhr{dim} = interior{dim} + nbhrOffset(dim);
        end
        matNbhr         = cell(grid.dim,1);
        [matNbhr{:}]    = ndgrid(nbhr{:});
        pindexNbhr      = sub2ind(P.size,matNbhr{:});                      % Patch-based cell indices - list
        pindexNbhr      = pindexNbhr(:);
        mapNbhr         = map(pindexNbhr);
        f = flux(interior{:},2*d+side-2);
        f = f(:);
        Alist = [Alist; ...
            [mapInterior mapInterior  f]; ...
            [mapInterior mapNbhr     -f] ...
            ];

        % Construct ghost cell equations = boundary conditions
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
        b(mapGhost) = 0.0;

    end
end

b(mapInterior) = rhsValues(interior{:});

% All remaining points are unused ghost points; set them to 0.

pindexRemaining = setdiff(pindexRemaining,pindexGhost);
mapRemaining = map(pindexRemaining);
Alist = [Alist; ...
    [mapRemaining mapRemaining repmat(1.0,size(mapRemaining))]; ...
    ];
b(mapRemaining) = 0.0;
