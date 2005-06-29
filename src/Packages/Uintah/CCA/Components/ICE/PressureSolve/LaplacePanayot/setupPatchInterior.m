function [A,b] = setupPatchInterior(grid,k,q,A,b,alpha,ilower,iupper,reallyUpdate)
%SETUPPATCCHINTERIOR  Set the discrete operator at a patch's edge.
%   [A,B] = SETUPPATCHINTERIOR(GRID,K,Q,A,B,ALPHA,ILOWER,IUPPER,REALLYUPDATE) updates the LHS
%   matrix A and the RHS matrix B, adding to them all the equations at
%   edge nodes (near patch boundaries) near all faces of the patches.
%   ALPHA is a parameter vector,
%   where each of its entries ALPHA(DIR), DIR=1,...,2*numDims is in (0,1)
%   that specifies the size of the face cells in the DIR direction.
%   ALPHA(DIR)=0.5 is a regular cell, 0.25 is used in Dirichlet B.C.
%   In the modified Panayot method we use ALPHA(DIR) smaller than 0.5
%   (maybe even smaller than 0.25) at C/F interfaces.
%
%   See also: TESTDISC, ADDGRIDPATCH, SETUPPATCHINTERIOR.

globalParams;

if (param.verboseLevel >= 1)
    fprintf('--- setupPatchInterior(k = %d, q = %d) ---\n',k,q);
end

if (nargin < 5)
    error('Too few input arguments (need at least grid,k,q,A,b)\n');
end

if (nargin < 6)
    % Standard alphas near boundaries = 0.25 (assuming Dirichlet BC
    % at all domain boundaries)
    alpha                   = zeros(1,2*grid.dim);
    for d = 1:grid.dim,
        for s = [-1 1],
            alpha(2*d-1)    = 0.25;                                 % Dirichlet boundary on the left in dimension d
            alpha(2*d)      = 0.25;                                 % Dirichlet boundary on the right in dimension d
        end
    end
end
if (nargin < 9)
    reallyUpdate = 1;
end

%=====================================================================
% Initialize; set patch "pointers" (in matlab: we actually copy P).
%=====================================================================
level                   = grid.level{k};
numPatches              = length(level.numPatches);
h                       = level.h;
P                       = grid.level{k}.patch{q};
ind                     = P.cellIndex;                              % Global 1D indices of cells
edgeDomain              = cell(2,1);                                % Domain edges
edgeDomain{1}           = level.minCell + P.offsetSub;              % First domain cell - next to left domain boundary - patch-based sub
edgeDomain{2}           = level.maxCell + P.offsetSub;              % Last domain cell - next to right domain boundary - patch-based sub

%=====================================================================
% Prepare a list of all cell indices whose equations are created below.
%=====================================================================

if (nargin < 7)
    ilower              = P.ilower;
end
if (nargin < 8)
    iupper              = P.iupper;
end
boxSize                 = iupper-ilower+1;

if (param.verboseLevel >= 3)
    ilower
    iupper
end
[indBox,box,matBox]     = indexBox(P,ilower,iupper);                % Indices whose equations are created and added to Alist below
x                       = cell(grid.dim,1);
for dim = 1:grid.dim,
    x{dim}              = (box{dim} - P.offsetSub(dim) - 0.5) * h(dim);
end
[x{:}]                  = ndgrid(x{:});

%=====================================================================
% Compute cell lengths and volumes.
%=====================================================================
cellLength              = cell(2*grid.dim,1);
near                    = cell(2*grid.dim,1);
far                     = cell(2*grid.dim,1);

% Standard interior cells. Cell's extents are
% [x{1}-cellLength{1},x{1}+cellLength{2}] x ... x
% [x{dim}-cellLength{2*dim-1},x{dim}+cellLength{2*dim-1}].
for dim = 1:grid.dim,                                                       % Loop over dimensions of face
    for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
        sideNum                 = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
        fluxNum                 = 2*dim+sideNum-2;
        cellLength{fluxNum}     = zeros(boxSize);
        cellLength{fluxNum}(:)  = 0.5*h(dim);
    end
end

% Adjust cells near domain boundaries
for dim = 1:grid.dim,                                                       % Loop over dimensions of face
    for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
        sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
        fluxNum         = 2*dim+sideNum-2;
        nearBoundary    = cell(1,grid.dim);
        [nearBoundary{:}] = find(matBox{dim} == edgeDomain{sideNum}(dim)); % Interior cell subs near DOMAIN boundary
        near{fluxNum}   = find(matBox{dim} == edgeDomain{sideNum}(dim));     % Interior cell indices near DOMAIN boundary (actually Dirichlet boundaries only)
        far{fluxNum}    = find(~(matBox{dim} == edgeDomain{sideNum}(dim)));   % The rest of the cells
        cellLength{fluxNum}(nearBoundary{:}) = alpha(fluxNum)*h(dim);       % Shrink cell in this direction
    end
end

% Compute cell volumes and centroids
volume                  = ones(boxSize);
centroid                = cell(grid.dim,1);
for dim = 1:grid.dim,                                                       % Loop over dimensions of face
    volume              = volume .* (cellLength{2*dim-1} + cellLength{2*dim});
    centroid{dim}		= x{dim} + 0.5*(-cellLength{2*dim-1} + cellLength{2*dim});
end

if (param.verboseLevel >= 3)
    fprintf('Cell Lengths = \n');
    cellLength{:}
    fprintf('Cell volumes = \n');
    volume
end

%=====================================================================
% Create fluxes for each cell (with corrections near boundaries).
%=====================================================================
flux                    = cell(2*grid.dim,1);
for dim = 1:grid.dim,                                                       % Loop over dimensions of face
    for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
        sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
        fluxNum         = 2*dim+sideNum-2;
        faceArea        = volume ./ (cellLength{2*dim-1} + cellLength{2*dim});
        flux{fluxNum}   = faceArea ./ (2*cellLength{fluxNum});              % Flux_coef = avg_diffusion_coef * face_area / (2*cellLength_of_this_direction)
    end
end
if (param.verboseLevel >= 3)
    fprintf('Fluxes = \n');
    flux{:}
end

%=====================================================================
% Create a list of non-zeros to be added to A, consisting of the stencil
% coefficients of all cells.
%=====================================================================
Alist                   = zeros(0,3);

for dim = 1:grid.dim,                                                       % Loop over dimensions of patch
    for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
        if (param.verboseLevel >= 2)
            fprintf('  ==> Adding fluxes to Alist (Face d = %d, s = %+d) ---\n',dim,side);
        end

        % Direction vector ("normal") from cell to its nbhr
        nbhrNormal      = zeros(1,grid.dim);
        nbhrNormal(dim) = side;
        indNbhr         = indexNbhr(P,indBox,nbhrNormal);

        % Add fluxes in dimension=dim, direction=side to list of non-zeros Alist and to b
        sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
        fluxNum         = 2*dim+sideNum-2;
        indFlux         = flux{fluxNum}(:);
        thisNear        = near{fluxNum};
        thisFar         = far{fluxNum};
        Alist = [Alist; ...                                                 % Far from Dirichlet boundaries, nbhrs are cell-center values
            [indBox(thisFar)    indBox(thisFar)         indFlux(thisFar)]; ...
            [indBox(thisFar)    indNbhr(thisFar)        -indFlux(thisFar)] ...
            ];

        Alist = [Alist; ...                                                 % near Dirichlet boundaries, nbhrs are fluxes
            [indBox(thisNear)   indNbhr(thisNear)       -indFlux(thisNear)] ...
            ];
    end
end

if (reallyUpdate)
    %=====================================================================
    % Set the equations of A of indices in the box [ilower,iupper] to
    % those specified by Alist.
    %=====================================================================
    Anew                    = spconvert([Alist; [grid.totalVars grid.totalVars 0]]);
    A(indBox,:)             = Anew(indBox,:);

    %=====================================================================
    % Update LHS vector b.
    %=====================================================================
    %rhsValues = rhs(x);
    rhsValues = rhs(centroid);

    b(indBox)               = volume(:) .* rhsValues(:);
end
