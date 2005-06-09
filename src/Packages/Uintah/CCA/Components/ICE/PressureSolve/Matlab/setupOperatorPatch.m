function [Alist,b] = setupOperatorPatch(grid,k,q,ilower,iupper)
%SETUPOPERATORPATCH  Set the discrete operator at a patch.
%   The usage [Alist,b] = setupOperatorPatch(grid,k,q,ilower,iupper)
%   Set the discrete equations at all interior cells of patch q at
%   level k. The ghost cells are either ignored (if they are on an interior
%   interface) or assigned a boundary condition (if against the domain
%   boundary). We set the equations within the box [ilower,iupper] in the
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
    all{dim} = [ilower(dim)-1:iupper(dim)+1] + P.offset(dim);  % Patch-based cell indices including ghosts
end
matAll      = cell(grid.dim,1);
[matAll{:}] = ndgrid(all{:});
pindexAll   = sub2ind(P.size,matAll{:});                      % Patch-based cell indices - list
pindexAll   = pindexAll(:);
pindexRemaining = pindexAll;

% Prepare a list of all interior cells
interior    = cell(grid.dim,1);
for dim = 1:grid.dim
    interior{dim} = [ilower(dim):iupper(dim)] + P.offset(dim);  % Patch-based cell indices including ghosts
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
    rhs(([ilower(1):iupper(1)]-0.5)*level.h(1),([ilower(2):iupper(2)]-0.5)*level.h(2));

%=====================================================================
% Correct fluxes near DOMAIN boundaries (not necessarily patch
% boundaries). Set corresponding ghost cell boundary conditions.
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

        % Correct flux at face
        flux(face{:},2*d+side-2) = 2*flux(face{:},2*d+side-2);                    % twice smaller spacing for flux in this direction
        otherDim = setdiff([1:2*grid.dim],[2*d-1,2*d]);
        for dim = otherDim
            flux(face{:},dim) = 0.75*flux(face{:},dim);     % 3/4 the boundary of CV in the perpendicular directions
        end
        rhsValues(face{:}) = 0.75*rhsValues(face{:});           % 3/4 less volume of CV ==> correct RHS average

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
% Construct stencil coefficients from fluxes flux-based, for
% interior cells. Ignore ghost cells that are not near domain
% boundaries. They will have an interpolation stencil later.
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
        
        [face{:}] = find(matInterior{d} == edgePatch{side}(d));          % Interior cell indices near PATCH boundary
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
        
        % Ghost cell equations of C/F interfaces; they are ignored here and
        % will have an interpolation stencil later.
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
        %         % B.C.: Dirichlet, u=0
        %         Alist = [Alist; ...
        %             [mapGhost mapGhost repmat(1.0,size(mapGhost))]; ...
        %             ];
        %         b(mapGhost-P.baseIndex+1) = 0.0;                            % This inserts the RHS data into the "chunk" of b that we output from this routine, hence we translate map indices to patch-based 1D indices.

    end
end

b(mapInterior-P.baseIndex+1) = rhsValues(interior{:});                            % This inserts the RHS data into the "chunk" of b that we output from this routine, hence we translate map indices to patch-based 1D indices.

% All remaining points are unused ghost points; set them to 0.
pindexRemaining = setdiff(pindexRemaining,pindexGhost);
fprintf('Number of unused points in this patch = %d\n',length(pindexRemaining));
mapRemaining = map(pindexRemaining);
Alist = [Alist; ...
    [mapRemaining mapRemaining repmat(1.0,size(mapRemaining))]; ...
    ];
b(mapRemaining-P.baseIndex+1) = 0.0;                            % This inserts the RHS data into the "chunk" of b that we output from this routine, hence we translate map indices to patch-based 1D indices.
