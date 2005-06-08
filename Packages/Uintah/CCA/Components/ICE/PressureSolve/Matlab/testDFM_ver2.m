%TESTDFM  Test Distributive Flux Matching (DFM) discretization.
%   We test DFM discretization error for a simple 2D Poisson problem with a
%   known solution. We prepare an AMR grid with two levels: a global level
%   1, and a local level 2 patch around the center of the domain, where the
%   solution u has more variations. DFM is a cell-centered, finite volume,
%   symmetric discretization on the composite AMR grid.
%
%   See also: ?.

%-------------------------------------------------------------------------
% Set up grid (AMR levels, patches)
%-------------------------------------------------------------------------

numLevels = 1;
grid = cell(numLevels,1);

%--------------- Level 1: global coarse grid -----------------

k = 1;                                          % Level ID
grid{k}.istart = [1 1];                         % Lower left corner of physical index space of this level
grid{k}.iend = [256 256];                           % Upper right corner of physical index space of this level
grid{k}.h = [1.0 1.0]./grid{k}.iend;            % Meshsize in x,y

% Patch 1 over the entire domain
p = 1;                                          % Patch ID
P.ilower = grid{k}.istart; ;                    % Index of lower-left corner of patch
P.iupper = grid{k}.iend;                        % Index of upper-right corner of patch
grid{k}.patch{p} = P;

%--------------- Level 2: local fine grid around center of domain -----------------

% k = 2;                                          % Level ID
% grid{k}.iBase = 65;                             % Base index of sparse-matrix index space of this level
% grid{k}.istart = [1 1];                         % Lower left corner of physical index space of this level
% grid{k}.iend = grid{k-1}.iend.*[2 2];           % Upper right corner of physical index space of this level
% grid{k}.h = grid{k-1}.h./[2 2];                 % Meshsize in x,y: refinement of 2 in each direction
% 
% % Patch 2 over central part
% p = 1;                                          % Patch ID
% grid{k}.patch{p}.ilower = [6 6];                % Index of lower-left corner of patch
% grid{k}.patch{p}.iupper = [13 13];              % Index of upper-right corner of patch

%-------------------------------------------------------------------------
% Set up stencils
%-------------------------------------------------------------------------

for k = 1:numLevels,
    grid{k}.stencilOffsets = [...                      % -Laplacian stencil non-zero structure
        [ 0 0 ]; ...
        [-1  0]; ...
        [ 1  0]; ...
        [ 0 -1]; ...
        [ 0  1]...
        ];
end

%-------------------------------------------------------------------------
% Set up the structured part of the matrix
%-------------------------------------------------------------------------
A = cell(numLevels,1);                                 % [i j aij] list for sparse matrix to be created
b = cell(numLevels,1);                                 % RHS
baseIndex = 1;

for k = 1:numLevels,
    %k
    g = grid{k};
    s = g.stencilOffsets;
    numPatches = length(g.patch);
    numEntries = size(s,1);
    
    % Add stencil entries to sparse matrix over each entire patch of each
    % AMR level. Analogous to HYPRE_SStructMatrixSetBoxValues call in HYPRE
    % FAC interface.
    
    A{k} = cell(numPatches,1);
    b{k} = cell(numPatches,1);
    
    for p = 1:numPatches,
        %p
        P = g.patch{p};
        PSize = P.iupper-P.ilower+1;
        %PSize
        LHS = zeros([PSize+2 numEntries]);          % Ghost cells on either side
        RHS = zeros(PSize+2);                       % Ghost cells on either side

        %============== Construct stencil coefficients - flux-based =================
        
        POffset     = -P.ilower+2;                   % Add to physical cell index to get patch cell index
        leftSide    = 2*ones(size(P.ilower));
        rightSide   = PSize+1;        
        rhsValues   = rhs(([P.ilower(1):P.iupper(1)]-0.5)*g.h(1),([P.ilower(2):P.iupper(2)]-0.5)*g.h(2));

        % Loop over (interior) patch cells
        for i1 = P.ilower(1):P.iupper(1)
            for i2 = P.ilower(2):P.iupper(2)
                j1 = i1 + POffset(1);
                j2 = i2 + POffset(2);

                % Flux vector of -Laplace operator.
                % Format: [west east north south] (west=2.0 means
                % 2.0*(uij-u_{i,j-1}), for instance)
                % rhsFactor multiplies the RHS average of the cell in the
                % discretization.
                
                flux = [1 1 1 1];
                rhsFactor = 1.0;

                % Change fluxes near boundaries
                if (j1 == leftSide(1))
                    flux(1) = flux(1)*2;
                    flux(3:4) = flux(3:4)*0.75;
                    rhsFactor = rhsFactor*0.75;
                end
                if (j1 == rightSide(1))
                    flux(2) = flux(2)*2;
                    flux(3:4) = flux(3:4)*0.75;
                    rhsFactor  = rhsFactor*0.75;
                end
                if (j2 == leftSide(2))
                    flux(3) = flux(3)*2;
                    flux(1:2) = flux(1:2)*0.75;
                    rhsFactor  = rhsFactor*0.75;
                end
                if (j2 == rightSide(2))
                    flux(4) = flux(4)*2;
                    flux(1:2) = flux(1:2)*0.75;
                    rhsFactor  = rhsFactor*0.75;
                end

                % Assemble fluxes into a stencil
                for i = 1:4
                    LHS(j1,j2,1) = LHS(j1,j2,1) + flux(i);
                    LHS(j1,j2,i+1) = LHS(j1,j2,i+1) - flux(i);
                end
                RHS(j1,j2) = rhsFactor * rhsValues(j1-1,j2-1) * prod(g.h);

            end
        end

        %baseIndex
        P.baseIndex = baseIndex;
        baseIndex = baseIndex + prod(PSize+2);
        
        % Save patch info in global data structures
        grid{k}.patch{p} = P;
        A{k}{p} = LHS;
        b{k}{p} = RHS;

    end
end

[As,bs] = AMRToSparse(A,b,grid);

%-------------------------------------------------------------------------
% Solve the linear system
%-------------------------------------------------------------------------

xs = As\bs;                         % Direct solver
u = sparseToAMR(xs,grid);           % Translate the solution vector to patch-based

%-------------------------------------------------------------------------
% Computed exact solution vector, patch-based
%-------------------------------------------------------------------------

for k = 1:numLevels,
    %k
    g = grid{k};

    for p = 1:length(g.patch),
        %p
        P = g.patch{p};        
        PSize = P.iupper-P.ilower+1;
        POffset = -P.ilower+2;                   % Add to physical cell index to get patch cell index                
        ue = zeros(PSize);
        for i1 = P.ilower(1):P.iupper(1)
            for i2 = P.ilower(2):P.iupper(2)
                ue(i1+POffset(1)-1,i2+POffset(2)-1) = exactSolution((i1-0.5)*g.h(1),(i2-0.5)*g.h(2));
            end
        end
        uExact{k}{p} = ue;
    end
end

k=1;
p=1;
figure(1);
clf;
surf(u{k}{p}-uExact{k}{p});
shg;
Lpnorm(u{k}{p}-uExact{k}{p})





