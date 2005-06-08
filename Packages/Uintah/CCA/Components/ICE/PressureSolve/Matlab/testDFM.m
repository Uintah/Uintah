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
%grid{k}.iBase = 1;                             % Base index of sparse-matrix index space of this level
grid{k}.istart = [1 1];                         % Lower left corner of physical index space of this level
grid{k}.iend = [4 4];                           % Upper right corner of physical index space of this level
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

numLevels = 1;          % Fake # levels to be only 1 for now

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

for k = 1:numLevels,
    k
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
        p
        P = g.patch{p};
        PSize = P.iupper-P.ilower+1;
        PSize
        LHS = zeros([PSize+2 numEntries]);          % Ghost cells on either side
        RHS = zeros(PSize+2);                       % Ghost cells on either side

        %============== Construct stencil coefficients - flux-based =================

        PSize   = prod(P.iupper-P.ilower+1);
        POffset = -P.ilower+2;                   % Add to physical cell index to get patch cell index
        leftSide = 2*ones(size(PSize));
        rightSide = PSize+1;
        
        rhsValues = rhs(([ilower(1):iupper(1)]-0.5)*g.h(1),([ilower(2):iupper(2)]-0.5)*g.h(2));


        % Loop over (interior) patch cells
        for i1 = P.ilower(1):P.iupper(1)
            for i2 = P.ilower(2):P.iupper(2)
                j1 = i1 + POffset(1);
                j2 = i2 + POffset(2);

                % Flux vector: [west east north south] (west=2.0 means
                % 2.0*(uij-u_{i,j-1}), for instance)
                flux = [1 1 1 1];
                rhs = 1.0;

                % Change fluxes near boundaries
                if (j1 == leftSide(1))
                    flux(1) = flux(1)*2;
                    flux(3:4) = flux(3:4)*0.75;
                    rhs = rhs*0.75;
                end
                if (j1 == rightSide(1))
                    flux(2) = flux(2)*2;
                    flux(3:4) = flux(3:4)*0.75;
                    rhs = rhs*0.75;
                end
                if (j2 == leftSide(2))
                    flux(3) = flux(3)*2;
                    flux(1:2) = flux(1:2)*0.75;
                    rhs = rhs*0.75;
                end
                if (j2 == rightSide(2))
                    flux(4) = flux(4)*2;
                    flux(1:2) = flux(1:2)*0.75;
                    rhs = rhs*0.75;
                end

                % Assemble fluxes into a stencil
                for i = 1:4
                    LHS(j1,j2) = LHS(j1,j2) + flux(i);
                    LHS(j1,j2) = LHS(j1,j2) - flux(i);
                end


            end
        end

        A{k}{p} = LHS;
        b{k}{p} = RHS;
            
        
        ilower = P.ilower;                              % Range of indices
        iupper = P.iupper;
        stencilValues = [...                            % -Laplacian stencil entries
            4.0; ...
            -1.0; ...
            -1.0; ...
            -1.0; ...
            -1.0; ...
            ];
        rhsValue = 1.0;
        
        RangeSize = iupper-ilower+1;        
        % Set LHS
        for entry = 1:length(stencilValues)             % Add entry to equations of all cells in the specified range
            A{k}{p} = setBoxValues(A{k}{p},P,ilower,iupper,entry,repmat(stencilValues(entry),RangeSize),'matrix');
        end
        A{k}{p}
        % Set RHS
        rhsValues = rhsValue * rhs(([ilower(1):iupper(1)]-0.5)*g.h(1),([ilower(2):iupper(2)]-0.5)*g.h(2));
        b{k}{p} = setBoxValues(b{k}{p},P,ilower,iupper,entry,rhsValues,'rhs');        
        b{k}{p}

        %============== X-MINUS FACE =================

        ilower = [P.ilower(1) P.ilower(2)];
        iupper = [P.ilower(1) P.iupper(2)];
        stencilValues = [...                            % -Laplacian stencil entries
            4.5; ...
            0; ...
            -2; ...
            -0.75; ...
            -0.75; ...
            ];
        rhsValue = 0.75;
        
        RangeSize = iupper-ilower+1;        
        % Set LHS
        for entry = 1:length(stencilValues)             % Add entry to equations of all cells in the specified range
            A{k}{p} = setBoxValues(A{k}{p},P,ilower,iupper,entry,repmat(stencilValues(entry),RangeSize),'matrix');
        end
        A{k}{p}
        % Set RHS
        rhsValues = rhsValue * rhs(([ilower(1):iupper(1)]-0.5)*g.h(1),([ilower(2):iupper(2)]-0.5)*g.h(2));
        b{k}{p} = setBoxValues(b{k}{p},P,ilower,iupper,entry,rhsValues,'rhs');        
        b{k}{p}
        
        %============== X-PLUS FACE =================

        ilower = [P.iupper(1) P.ilower(2)];
        iupper = [P.iupper(1) P.iupper(2)];
        stencilValues = [...                            % -Laplacian stencil entries
            4.5; ...
            -2; ...
            0; ...
            -0.75; ...
            -0.75; ...
            ];
        rhsValue = 0.75;
        
        RangeSize = iupper-ilower+1;        
        % Set LHS
        for entry = 1:length(stencilValues)             % Add entry to equations of all cells in the specified range
            A{k}{p} = setBoxValues(A{k}{p},P,ilower,iupper,entry,repmat(stencilValues(entry),RangeSize),'matrix');
        end
        A{k}{p}
        % Set RHS
        rhsValues = rhsValue * rhs(([ilower(1):iupper(1)]-0.5)*g.h(1),([ilower(2):iupper(2)]-0.5)*g.h(2));
        b{k}{p} = setBoxValues(b{k}{p},P,ilower,iupper,entry,rhsValues,'rhs');        
        b{k}{p}

        %============== Y-MINUS FACE =================

        ilower = [P.ilower(1) P.ilower(2)];
        iupper = [P.iupper(1) P.ilower(2)];
        stencilValues = [...                            % -Laplacian stencil entries
            4.5; ...
            -0.75; ...
            -0.75; ...
            0; ...
            -2; ...
            ];
        rhsValue = 0.75;
        
        RangeSize = iupper-ilower+1;        
        % Set LHS
        for entry = 1:length(stencilValues)             % Add entry to equations of all cells in the specified range
            A{k}{p} = setBoxValues(A{k}{p},P,ilower,iupper,entry,repmat(stencilValues(entry),RangeSize),'matrix');
        end
        A{k}{p}
        % Set RHS
        rhsValues = rhsValue * rhs(([ilower(1):iupper(1)]-0.5)*g.h(1),([ilower(2):iupper(2)]-0.5)*g.h(2));
        b{k}{p} = setBoxValues(b{k}{p},P,ilower,iupper,entry,rhsValues,'rhs');        
        b{k}{p}

        %============== Y-PLUS FACE =================

        ilower = [P.ilower(1) P.iupper(2)];
        iupper = [P.iupper(1) P.iupper(2)];
        stencilValues = [...                            % -Laplacian stencil entries
            4.5; ...
            -0.75; ...
            -0.75; ...
            -2; ...
            0; ...
            ];
        rhsValue = 0.75;
        
        RangeSize = iupper-ilower+1;        
        % Set LHS
        for entry = 1:length(stencilValues)             % Add entry to equations of all cells in the specified range
            A{k}{p} = setBoxValues(A{k}{p},P,ilower,iupper,entry,repmat(stencilValues(entry),RangeSize),'matrix');
        end
        A{k}{p}
        % Set RHS
        rhsValues = rhsValue * rhs(([ilower(1):iupper(1)]-0.5)*g.h(1),([ilower(2):iupper(2)]-0.5)*g.h(2));
        b{k}{p} = setBoxValues(b{k}{p},P,ilower,iupper,entry,rhsValues,'rhs');        
        b{k}{p}
        
                
        % SW Corner
        stencilValues = [...                      % -Laplacian stencil entries
            4.5; ...
            0; ...
            -0.75; ...
            0; ...
            -0.75; ...
            ];
        low = [pat.ilower(1) pat.ilower(2)];
        up = low;
        A = setBoxValues(A,g,k,p,low,up,stencilValues,'matrix');
        b = setBoxValues(b,g,k,p,low,up,9.0/16.0,'rhs');
        
        % SE Corner
        fprintf('SE corner\n');
        stencilValues = [...                      % -Laplacian stencil entries
            4.5; ...
            -0.75; ...
            0; ...
            0; ...
            -0.75; ...
            ];
        low = [pat.iupper(1) pat.ilower(2)];
        up = low;
        A = setBoxValues(A,g,k,p,low,up,stencilValues,'matrix');
        b = setBoxValues(b,g,k,p,low,up,9.0/16.0,'rhs');
        
        % NW Corner
        stencilValues = [...                      % -Laplacian stencil entries
            4.5; ...
            0; ...
            -0.75; ...
            -0.75; ...
            0; ...
            ];
        low = [pat.ilower(1) pat.iupper(2)];
        up = low;
        A = setBoxValues(A,g,k,p,low,up,stencilValues,'matrix');
        b = setBoxValues(b,g,k,p,low,up,9.0/16.0,'rhs');
        
        % NE Corner
        stencilValues = [...                      % -Laplacian stencil entries
            4.5; ...
            -0.75; ...
            0; ...
            -0.75; ...
            0; ...
            ];
        low = [pat.iupper(1) pat.iupper(2)];
        up = low;
        A = setBoxValues(A,g,k,p,low,up,stencilValues,'matrix');
        b = setBoxValues(b,g,k,p,low,up,9.0/16.0,'rhs');

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





