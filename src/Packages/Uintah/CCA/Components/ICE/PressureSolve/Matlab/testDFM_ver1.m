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

numLevels = 2;
grid = cell(numLevels,1);

%--------------- Level 1: global coarse grid -----------------

k = 1;                                          % Level ID
grid{k}.iBase = 1;                              % Base index of sparse-matrix index space of this level
grid{k}.istart = [1 1];                         % Lower left corner of physical index space of this level
grid{k}.iend = [16 16];                           % Upper right corner of physical index space of this level
grid{k}.h = [1.0 1.0]./grid{k}.iend;                      % Meshsize in x,y

% Patch 1 over the entire domain
p = 1;                                          % Patch ID
grid{k}.patch{p}.ilower = grid{k}.istart; %[1 1];                % Index of lower-left corner of patch
grid{k}.patch{p}.iupper = grid{k}.iend; %[4 4];                % Index of upper-right corner of patch

%--------------- Level 2: local fine grid around center of domain -----------------

k = 2;                                          % Level ID
grid{k}.iBase = 65;                             % Base index of sparse-matrix index space of this level
grid{k}.istart = [1 1];                         % Lower left corner of physical index space of this level
grid{k}.iend = grid{k-1}.iend.*[2 2];           % Upper right corner of physical index space of this level
grid{k}.h = grid{k-1}.h./[2 2];                 % Meshsize in x,y: refinement of 2 in each direction

% Patch 2 over central part
p = 1;                                          % Patch ID
grid{k}.patch{p}.ilower = [6 6];                % Index of lower-left corner of patch
grid{k}.patch{p}.iupper = [13 13];              % Index of upper-right corner of patch

numLevels = 1;          % Fake # levels to be only 1 for now

%-------------------------------------------------------------------------
% Set up stencils
%-------------------------------------------------------------------------

for k = 1:numLevels,
    %    grid{k}.stencilOffsets = zeros(5,2);
    grid{k}.stencilOffsets = [...                      % -Laplacian stencil non-zero structure
        [ 0 0 ]; ...
        [-1  0]; ...
        [ 1  0]; ...
        [ 0 -1]; ...
        [ 0  1]...
        ];
    grid{k}.stencilValues = [...                      % -Laplacian stencil entries
        4.0; ...
        -1.0; ...
        -1.0; ...
        -1.0; ...
        -1.0; ...
        ];
end

%-------------------------------------------------------------------------
% Set up the structured part of the matrix
%-------------------------------------------------------------------------
A = zeros(0,3);                                 % [i j aij] list for sparse matrix to be created
b = zeros(0,1);

for k = 1:numLevels,
    k
    g = grid{k};
    s = g.stencilOffsets;

    for entry = 1:size(s,1),% Loop over stencil entries

        % Add stencil entries to sparse matrix over each entire patch of each
        % AMR level. Analogous to HYPRE_SStructMatrixSetBoxValues call in HYPRE
        % FAC interface.

        for p = 1:length(g.patch),
            p
            pat = g.patch{p};
            numCells = prod(pat.iupper-pat.ilower+1);
            values = zeros(size(pat.iupper-pat.ilower+1));
            rhsValues = zeros(size(pat.iupper-pat.ilower+1));
            for i1 = pat.ilower(1):pat.iupper(1)
                for i2 = pat.ilower(2):pat.iupper(2)
                    values(i1,i2) = g.stencilValues(entry);
                    rhsValues(i1,i2) = prod(g.h)*rhs((i1-0.5)*g.h(1),(i2-0.5)*g.h(2));
                end
            end
            % Boundary adjustment - Dirichlet B.C. with CC ==> control
            % volume smaller
            for i1 = [pat.ilower(1) pat.iupper(1)]
                for i2 = pat.ilower(2)+1:pat.iupper(2)-1
                    if (entry == 1)
                        values(i1,i2) = g.stencilValues(entry) - 0.5;
                    end
                    rhsValues(i1,i2) = 0.75*prod(g.h)*rhs((i1-0.5)*g.h(1),(i2-0.5)*g.h(2));
                end
            end
            for i2 = [pat.ilower(2) pat.iupper(2)]
                for i1 = pat.ilower(1)+1:pat.iupper(1)-1
                    if (entry == 1)
                        values(i1,i2) = g.stencilValues(entry) - 0.5;
                    end
                    rhsValues(i1,i2) = 0.75*prod(g.h)*rhs((i1-0.5)*g.h(1),(i2-0.5)*g.h(2));
                end
            end            
            % Corners
            for i1 = [pat.ilower(1) pat.iupper(1)]
                for i2 = [pat.ilower(2) pat.iupper(2)]
                    if (entry == 1)
                        values(i1,i2) = g.stencilValues(entry) - 1.0;
                    end
                    rhsValues(i1,i2) = 0.5*prod(g.h)*rhs((i1-0.5)*g.h(1),(i2-0.5)*g.h(2));
                end
            end            
           
            
            %values
            %rhsValues

            %values = repmat(g.stencilValues(entry),[numCells 1]);
            %rhsValues = ...;

            A = setBoxValues(A,g,pat.ilower,pat.iupper,s(entry,:),values(:));
            if (entry == 1)
                b = [b; rhsValues(:)];
            end
        end
    end
end

B = sparse(A(:,1),A(:,2),A(:,3));

%-------------------------------------------------------------------------
% Solve the linear system
%-------------------------------------------------------------------------

x = B\b;

%-------------------------------------------------------------------------
% Translate the solution vector to patch-based
%-------------------------------------------------------------------------

for k = 1:numLevels,
    k
    g = grid{k};

    for p = 1:length(g.patch),
        p
        pat = g.patch{p};
        numCells = prod(pat.iupper-pat.ilower+1);
        u = zeros(pat.iupper-pat.ilower+1);
        u(:) = x(1:numCells);
        uExact = zeros(size(u));
        for i1 = pat.ilower(1):pat.iupper(1)
            for i2 = pat.ilower(2):pat.iupper(2)
                uExact(i1,i2) = exactSolution((i1-0.5)*g.h(1),(i2-0.5)*g.h(2));
            end
        end
    end
end

figure(1);
clf;
surf(u-uExact);
shg;
Lpnorm(u-uExact)





