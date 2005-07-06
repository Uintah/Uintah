function [A,b,T,Alist,Tlist] = setupPatchInterior(grid,k,q,A,b,T,ilower,iupper,reallyUpdate)
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
    fprintf('--- setupPatchInterior(k = %d, q = %d) BEGIN ---\n',k,q);
end

if (nargin < 6)
    error('Too few input arguments (need at least grid,k,q,A,b)\n');
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
edgePatch               = cell(2,1);                                % Domain edges
edgePatch{1}            = ilower + P.offsetSub;                     % First patch cell - next to left domain boundary - patch-based sub
edgePatch{2}            = iupper + P.offsetSub;                     % Last patch cell - next to right domain boundary - patch-based sub
[indBox,box,matBox]     = indexBox(P,ilower,iupper);                % Indices whose equations are created and added to Alist below
x                       = cell(grid.dim,1);
for dim = 1:grid.dim,
    x{dim}              = (box{dim} - P.offsetSub(dim) - 0.5) * h(dim);
end
[x{:}]                  = ndgrid(x{:});

%=====================================================================
% Compute cell lengths and volumes.
%=====================================================================
% Cell extent is 
% [x{1}-0.5*h(1),x{1}+0.5*h(1)] x ... x [x{dim}-0.5*h(dim),x{dim}+0.5*h(dim)].
% Cell volume is thus prod(h).
diffLength              = cell(2*grid.dim,1);                               % Distance along which we approximate the flux by finite difference
volume                  = prod(h).*ones(boxSize);                           % Cell volume
centroid                = x;                                                % Cell centroid
near                    = cell(2*grid.dim,1);                               % Cells near domain boundaries in the fluxNum direction
far                     = cell(2*grid.dim,1);                               % Cells far from domain boundaries in the fluxNum direction

for dim = 1:grid.dim,                                                       % Loop over dimensions of face
    for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
        sideNum                 = (side+3)/2;                               % side=-1 ==> 1; side=1 ==> 2
        fluxNum                 = 2*dim+sideNum-2;
        diffLength{fluxNum}     = h(dim)*ones(boxSize);                     % Standard FD is over distance h        

        % Adjust ind map near a nbhring fine patch (override ghost point
        % indices with interior indices from nbhring patch PN)
        qn                      = P.nbhrPatch(dim,(side+3)/2);
        if (qn > 0)
            if (param.verboseLevel >= 2)
                fprintf('Found nbhring patch qn=%d for dim=%d, side=%d, modifying ind near this face\n',qn,dim,side);
            end
            % indices of this face in P.cellIndex
            flower                  = P.ilower;
            fupper                  = P.iupper;
            if (side == -1)
                flower(dim)         = flower(dim)-1;                        % Ghost points on the PN-face
                fupper(dim)         = flower(dim);
            else
                fupper(dim)         = fupper(dim)+1;                        % Ghost points of the PN-face
                flower(dim)         = fupper(dim);
            end
            [indFace,face,matFace]  = indexBox(P,flower,fupper);
            
            % indices of the interior points in PN.cellIndex replacing the
            % ghost points of this face in P
            PN                      = grid.level{k}.patch{qn};
            flower                  = PN.ilower;
            fupper                  = PN.iupper;
            if (side == 1)                                                  % Refering to face(dim,-side) in PN
                fupper(dim)         = flower(dim);
            else
                flower(dim)         = fupper(dim);
            end
            [indNFace,NFace,matNFace] = indexBox(PN,flower,fupper);
            
            % Override ghost cell indices in ind (but not in P.cellIndex!)
            ind(matFace{:})         = PN.cellIndex(matNFace{:});            
        end

        % Adjust distances for FD near domain boundaries        
        nearBoundary            = cell(1,grid.dim);
        [nearBoundary{:}]       = find(matBox{dim} == edgeDomain{sideNum}(dim));  % Interior cell subs near DOMAIN boundary
        near{fluxNum}   = find(matBox{dim} == edgeDomain{sideNum}(dim));    % Interior cell indices near DOMAIN boundary (actually Dirichlet boundaries only)
        far{fluxNum}    = find(~(matBox{dim} == edgeDomain{sideNum}(dim))); % The rest of the cells
        diffLength{fluxNum}(nearBoundary{:}) = 0.5*h(dim);                  % Twice smaller distance in this direction
    end
end

%=====================================================================
% Create fluxes for each cell (with corrections near boundaries).
%=====================================================================
flux                    = cell(2*grid.dim,1);
for dim = 1:grid.dim,                                                       % Loop over dimensions of face
    for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
        sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
        fluxNum         = 2*dim+sideNum-2;
        faceArea        = volume ./ h(dim);
        flux{fluxNum}   = faceArea ./ diffLength{fluxNum};                  % Flux_coef = avg_diffusion_coef (=1 here) * face_area / (FD length)
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
Tlist                   = zeros(0,3);
indAll                  = indBox;
indTransformed          = [];

for dim = 1:grid.dim,                                                       % Loop over dimensions of patch
    for side = [-1 1]                                                       % side=-1 (left) and side=+1 (right) directions in dimension d
        if (param.verboseLevel >= 2)
            fprintf('  ==> Adding fluxes to Alist (Face d = %d, s = %+d) ---\n',dim,side);
        end

        % Direction vector ("normal") from cell to its nbhr
        nbhrNormal      = zeros(1,grid.dim);
        nbhrNormal(dim) = side;
        indNbhr         = indexNbhr(P,indBox,nbhrNormal,ind);

        % Add fluxes in dimension=dim, direction=side to list of non-zeros Alist and to b
        sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
        fluxNum         = 2*dim+sideNum-2;
        indFlux         = flux{fluxNum}(:);
        thisNear        = near{fluxNum};
        thisFar         = far{fluxNum};
        
        % Contribution of flux to interior equation at indBox
        Alist = [Alist; ...                                                 % Far from Dirichlet boundaries, nbhrs are cell-center values
            [indBox(thisFar)    indBox(thisFar)         indFlux(thisFar)]; ...
            [indBox(thisFar)    indNbhr(thisFar)        -indFlux(thisFar)] ...
            ];
        Alist = [Alist; ...                                                 % near Dirichlet boundaries, nbhrs are fluxes
            [indBox(thisNear)   indNbhr(thisNear)       -indFlux(thisNear)] ...
            ];

        % Contribution of flux to BC equation (Dirichlet B.C.) at indNbhr
        % BC vars are fluxes so that A will be symmetric.
        if (~isempty(thisNear))
            indBC = indNbhr(thisNear);
            Alist = [Alist; ...                                                 % BC vars (= nbhr vars) are fluxes
                [indBC   indBC                   -indFlux(thisNear)]; ...
                [indBC   indBox(thisNear)        -indFlux(thisNear)] ...
                ];
            Tlist = [Tlist; ...                                                 % BC vars (= nbhr vars) are fluxes
                [indBC   indBC                   repmat(1.0,size(indBC))]; ...
                [indBC   indBox(thisNear)        repmat(-1.0,size(indBC))] ...
                ];
            indAll = union(indAll,indBC);
            indTransformed = union(indTransformed,indBC);
            
            if (reallyUpdate)
                %=====================================================================
                % Update LHS vector b with boundary equations.
                %=====================================================================
                xBC                 = cell(grid.dim,1);
                for d = 1:grid.dim
                    xBC{d}          = x{d}(thisNear);                                 % BC RHS location (at faces, not cell centers) - to be corrected below
                end
                xBC{dim}            = xBC{dim} + side*0.5*h(dim);                      % Move from cell center to cell face
                rhsBCValues         = rhsBC(xBC);
                b(indBC)            = -indFlux(thisNear) .* rhsBCValues(:);
            end
        end
    end    
end

if (reallyUpdate)
    %=====================================================================
    % Set the equations of A of indices in the box [ilower,iupper] to
    % those specified by Alist, and the BC equations of the BC vars
    % neighboring the box.
    %=====================================================================
    Anew                    = spconvert([Alist; [grid.totalVars grid.totalVars 0]]);
    A(indAll,:)             = Anew(indAll,:);

    %=====================================================================
    % Update LHS vector b with interior equations.
    %=====================================================================
    rhsValues               = rhs(centroid);
    b(indBox)               = volume(:) .* rhsValues(:);        

    %=====================================================================
    % Update transformation matrix T.
    %=====================================================================
    Tnew                    = spconvert([Tlist; [grid.totalVars grid.totalVars 0]]);
    T(indTransformed,:)     = Tnew(indTransformed,:);
end

if (param.verboseLevel >= 1)
    fprintf('--- setupPatchInterior(k = %d, q = %d) END ---\n',k,q);
end
