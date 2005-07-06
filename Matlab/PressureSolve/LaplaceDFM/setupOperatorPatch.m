function [Alist,b] = setupOperatorPatch(grid,k,q,ilower,iupper,interiorConnections,ghostConnections)
%SETUPOPERATORPATCH  Set the discrete operator at a patch.
%   The usage [Alist,b] = setupOperatorPatch(grid,k,q,ilower,iupper,interiorConnections,ghostConnections)
%   Set the discrete equations at all interior cells of patch q at
%   level k. The ghost cells are either ignored (if they are on an interior
%   interface) or assigned a boundary condition (if against the domain
%   boundary). We set the equations within the box [ilower,iupper] in the
%   patch. On output we return Alist (list of [i j aij]) and the part of b
%   that should be added to a global list Alist to be converted to sparse
%   format, and to a global RHS vector b. If ghostConnections=1, Alist also
%   includes the connections from ghost cells to interior cells, otherwise
%   does not include that. If interiorConnections=1, Alist includes the
%   equations at interior nodes and boundary conditions at ghost cells at
%   domain boundaries. If not, only ghostConnections to interior
%   nodes will be in Alist, if ghostConnections=1; otherwise, Alist is
%   empty.
%
%   See also: TESTDFM, SETOPERATOR, DELETEUNDERLYINGDATA.
global verboseLevel

if (verboseLevel >= 1)
    fprintf('--- setupOperatorPatch(k = %d, q = %d) ---\n',k,q);
end
%=====================================================================
% Initialize; set patch "pointers" (in matlab: we actually copy P)
%=====================================================================
level       = grid.level{k};
numPatches  = length(level.numPatches);
h           = level.h;
P           = grid.level{k}.patch{q};
Alist       = zeros(0,3);
boxSize     = iupper-ilower+3;
boxOffset   = P.offset + P.ilower - ilower;
b           = zeros(prod(boxSize),1);

%=====================================================================
% Compute local map of the box (including ghost cells)
%=====================================================================
all         = cell(grid.dim,1);
for dim     = 1:grid.dim
    all{dim} = [ilower(dim)-1:iupper(dim)+1] + P.offset(dim);       % Patch-based cell indices including ghosts
end
matAll      = cell(grid.dim,1);
[matAll{:}] = ndgrid(all{:});
pindexAll   = sub2ind(P.size,matAll{:});                            % Patch-based cell indices - list
pindexAll   = pindexAll(:);
map         = P.cellIndex(pindexAll);                               % Cell global indices for the box [ilower-1,iupper+1]

%=====================================================================
% Prepare a list of all cell indices in this box including ghosts
%=====================================================================
all    = cell(grid.dim,1);
for dim = 1:grid.dim
    all{dim} = [ilower(dim)-1:iupper(dim)+1] + boxOffset(dim);      % Box-based cell indices including ghosts
end
matAll      = cell(grid.dim,1);
[matAll{:}] = ndgrid(all{:});
pindexAll   = sub2ind(boxSize,matAll{:});                           % Box-based cell indices - list
pindexAll   = pindexAll(:);
pindexRemaining = pindexAll;

%=====================================================================
% Prepare a list of all interior cells
%=====================================================================
interior    = cell(grid.dim,1);
for dim = 1:grid.dim
    interior{dim} = [ilower(dim):iupper(dim)] + boxOffset(dim);     % Patch-based cell indices including ghosts
end
matInterior      = cell(grid.dim,1);
[matInterior{:}] = ndgrid(interior{:});
pindexInterior   = sub2ind(boxSize,matInterior{:});                 % Patch-based cell indices - list
pindexInterior  = pindexInterior(:);
mapInterior     = map(pindexInterior);
pindexRemaining = setdiff(pindexRemaining,pindexInterior);

%=====================================================================
% Compute edge indices
%=====================================================================
% Domain edges
edgeDomain        = cell(2,1);
edgeDomain{1}     = level.minCell + boxOffset;                      % First cell next to left domain boundary - patch-based index
edgeDomain{2}     = level.maxCell + boxOffset;                      % Last Cell next to right domain boundary - patch-based index
% Patch edges
edgePatch          = cell(2,1);
edgePatch{1}       = ilower + boxOffset;                            % First cell next to left domain boundary - patch-based index
edgePatch{2}       = iupper + boxOffset;                            % Last Cell next to right domain boundary - patch-based index

%=====================================================================
% Construct interior fluxes
%=====================================================================
% Flux vector of -Laplace operator.
% Format: [west east north south] (west=2.0 means
% 2.0*(uij-u_{i,j-1}), for instance)
% rhsFactor multiplies the RHS average of the cell in the
% discretization.
flux = zeros([boxSize 2*grid.dim]);
for direction = 1:2*grid.dim
    flux(interior{:},direction) = 1;                                % Fluxes for interior cells not near boundaries: [1 1 1 1]
end
rhsValues = zeros(boxSize);
rhsValues(interior{:}) = prod(h) * ...
    rhs(([ilower(1):iupper(1)]-0.5)*level.h(1),([ilower(2):iupper(2)]-0.5)*level.h(2));

%=====================================================================
% Correct fluxes near DOMAIN boundaries (not necessarily patch
% boundaries). Set corresponding ghost cell boundary conditions.
%=====================================================================
face = cell(grid.dim,1);
for d = 1:grid.dim,                                                 % Loop over dimensions of patch
    for side = 1:2                                                  % side=1 (left) and side=2 (right) in dimension d
        if (side == 1)
            direction = -1;
        else
            direction = 1;
        end

        [face{:}] = find(matInterior{d} == edgeDomain{side}(d));    % Interior cell indices near DOMAIN boundary
        % Process only if this face is non-empty.
        if (isempty(face{1}))
            continue;
        end

        for dim = 1:grid.dim                                        % Translate face to patch-based indices (from index in the INTERIOR matInterior)
            face{dim} = face{dim} + 1;
        end

        % Correct flux at face
        flux(face{:},2*d+side-2) = 2*flux(face{:},2*d+side-2);      % twice smaller spacing for flux in this direction
        otherDim = setdiff([1:2*grid.dim],[2*d-1,2*d]);
        for dim = otherDim
            flux(face{:},dim) = 0.75*flux(face{:},dim);             % 3/4 the boundary of CV in the perpendicular directions
        end
        rhsValues(face{:}) = 0.75*rhsValues(face{:});               % 3/4 less volume of CV ==> correct RHS average

        % Construct corresponding ghost cell equations = boundary conditions
        ghost           = face;
        ghost{d}        = face{d} + direction;                      % Ghost cell indices
        ghost{d}        = ghost{d}(1);                              % Because face is an ndgrid-like structure, it repeats the same value in ghost{d}, lengthghostnbhr{d}) times. So shrink it back to a scalar so that ndgrid and flux(...) return the correct size vectors. We need to do that only for dimension d as we are on a face which is grid.dim-1 dimensional object.
        matGhost        = cell(grid.dim,1);
        [matGhost{:}]   = ndgrid(ghost{:});
        pindexGhost     = sub2ind(boxSize,matGhost{:});
        pindexGhost     = pindexGhost(:);
        mapGhost        = map(pindexGhost);
        pindexRemaining = setdiff(pindexRemaining,pindexGhost);
        
        %======== B.C.: Dirichlet, u=rhsBC =============
        if (interiorConnections == 1)
            inwardNormal = zeros(1,grid.dim);
            inwardNormal(d) = -direction;

            % Ghost point physical locations
            ghostLocation   = cell(grid.dim,1);
            for dim = 1:grid.dim                                                % Translate face to patch-based indices (from index in the INTERIOR matInterior)
                ghostLocation{dim} = (ghost{dim} - boxOffset(dim) - 0.5 + 0.5*inwardNormal(dim))*h(dim);
            end
            matGhostLocation = cell(grid.dim,1);
            [matGhostLocation{:}]   = ndgrid(ghostLocation{:});
            %rhsBCValues = zeros(boxSize);
            rhsBCValues = rhsBC(ghostLocation{:});
            Alist = [Alist; ...
                [mapGhost mapGhost repmat(1.0,size(mapGhost))]; ...
                ];
            b(pindexGhost) = rhsBCValues(:);                                   % This inserts the RHS data into the "chunk" of b that we output from this routine, hence we translate map indices to patch-based 1D indices.
        end
    end
end

%=====================================================================
% Construct stencil coefficients from fluxes flux-based, for
% interior cells. Ignore ghost cells that are not near domain
% boundaries. They will have an interpolation stencil later.
%=====================================================================
for d = 1:grid.dim,                                                 % Loop over dimensions of patch
    %d
    for side = 1:2                                                  % side=1 (left) and side=2 (right) in dimension d
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
        pindexNbhr      = sub2ind(boxSize,matNbhr{:});                      % Patch-based cell indices - list
        pindexNbhr      = pindexNbhr(:);
        mapNbhr         = map(pindexNbhr);
        f = flux(interior{:},2*d+side-2);
        f = f(:);
        if (interiorConnections == 1)
            Alist = [Alist; ...
                [mapInterior mapInterior  f]; ...
                [mapInterior mapNbhr     -f] ...
                ];
        end

        % Ghost cell equations of C/F interfaces; they are normally ignored here and
        % will have an interpolation stencil later. However, for
        % deleteUnderlyingData(), we also need the coefficients of interior
        % patch nodes in the surrounding cell equations (which are denoted for
        % notation unifomity "ghost cells" w.r.t the box [ilower,iupper]).
        ghost           = face;
        ghost{d}        = face{d} + direction;                              % Ghost cell indices
        ghost{d}        = ghost{d}(1);                                      % Because face is an ndgrid-like structure, it repeats the same value in ghost{d}, lengthghostnbhr{d}) times. So shrink it back to a scalar so that ndgrid and flux(...) return the correct size vectors. We need to do that only for dimension d as we are on a face which is grid.dim-1 dimensional object.
        %[ghost{:}]
        matGhost        = cell(grid.dim,1);        
        [matGhost{:}]   = ndgrid(ghost{:});
        pindexGhost     = sub2ind(boxSize,matGhost{:});
        pindexGhost     = pindexGhost(:);
        mapGhost        = map(pindexGhost);
        %mapGhost
        pindexRemaining = setdiff(pindexRemaining,pindexGhost);

        % Inward normal direction
        nbhrOffset = zeros(1,grid.dim);
        nbhrOffset(d) = -direction;
        % Nbhrs of all ghost cells in this direction
        nbhr = cell(grid.dim,1);
        for dim = 1:grid.dim
            nbhr{dim} = ghost{dim} + nbhrOffset(dim);
        end
        matNbhr         = cell(grid.dim,1);
        [matNbhr{:}]    = ndgrid(nbhr{:});
        pindexNbhr      = sub2ind(boxSize,matNbhr{:});                      % Patch-based cell indices - list
        pindexNbhr      = pindexNbhr(:);
        mapNbhr         = map(pindexNbhr);
        f = flux(nbhr{:},2*d+side-2);
        f = f(:);
        
        if (ghostConnections & (edgePatch{side}(d) ~= edgeDomain{side}(d)))
            % Add the following connections only if we are not near a
            % domain boundary where there are B.C. at the ghost cells.
            % Add fluxes of the reverse direction (inward normal to this face)
            % to list of non-zeros Alist and to b

            if (ghostConnections == 2)      % Return interpolation coefficient of point in the patch to ghost point
                r = grid.level{k}.refRatio;
                alpha = (r(d)-1)/(r(d)+1);
                Alist = [Alist; ...
                    [mapGhost mapNbhr repmat(alpha,size(mapGhost))]; ...
                    ];
            else                            % Normal flux connections of ghost points to points in the patch
                Alist = [Alist; ...
                    [mapGhost mapGhost  f]; ...
                    [mapGhost mapNbhr  -f] ...
                    ];
            end
        end
        
        if ((interiorConnections == 2) & (edgePatch{side}(d) == edgeDomain{side}(d)))
            % Return a list of Dirichlet B.C. nodes and their nbhrs inside
            % the patch. For diagonally scaling the matrix to be symmetric
            % there.
            Alist = [Alist; ...
                [mapGhost mapNbhr f];
                ];
        end
        
    end
end

%=====================================================================
% Set dummy (unused) nodes to zero (LHS=identity, RHS=0)
%=====================================================================
if (interiorConnections == 1)
    b(pindexInterior) = rhsValues(interior{:});                            % This inserts the RHS data into the "chunk" of b that we output from this routine, hence we translate map indices to patch-based 1D indices.
    if (~ghostConnections)
        % All remaining points are unused ghost points; set them to 0.
        pindexRemaining = setdiff(pindexRemaining,pindexGhost);
        if (verboseLevel >= 2)
            fprintf('Number of unused points in this patch = %d\n',length(pindexRemaining));
        end
        mapRemaining = map(pindexRemaining);
        Alist = [Alist; ...
            [mapRemaining mapRemaining repmat(1.0,size(mapRemaining))]; ...
            ];
        b(pindexRemaining) = 0.0;                            % This inserts the RHS data into the "chunk" of b that we output from this routine, hence we translate map indices to patch-based 1D indices.
    end
end
