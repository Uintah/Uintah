function [grid,q,A,b] = addGridPatch(grid,k,ilower,iupper,parentQ,A,b)
global verboseLevel
% Add a patch to level k; patch id returned = q

tStartCPU           = cputime;
tStartElapsed       = clock;

if (max(ilower > iupper))
    error('Cannot create patch -- ilower > iupper');
end

%==============================================================
% 1. Create an empty patch
%==============================================================
if (verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 1. Create an empty patch\n');
    fprintf('#########################################################################\n');
end

grid.level{k}.numPatches    = grid.level{k}.numPatches+1;
q                           = grid.level{k}.numPatches;
P.ilower                    = ilower;
P.iupper                    = iupper;
P.size                      = P.iupper - P.ilower + 3;          % Size including ghost cells
P.parent                    = parentQ;
P.children                  = [];
P.offsetSub                 = -P.ilower+2;                      % Add to level-global cell index to get this-patch cell index. Lower left corner (a ghost cell) is (1,1) in patch indices
P.deletedBoxes              = [];
grid.level{k}.patch{q}      = P;
if (k > 1)    
    grid.level{k-1}.patch{parentQ}.children = [grid.level{k-1}.patch{q}.children parentQ];
end
if (verboseLevel >= 1)
    fprintf('Created level k=%3d patch q=%3d (parentQ = %3d), ilower = [%3d %3d], iupper = [%3d %3d]\n',...
        k,q,parentQ,ilower,iupper);
end

grid                        = updateGrid(grid);
b                           = [b; zeros(grid.totalVars-length(b),1)];

%==============================================================
% 2. Create patch interior equations
%==============================================================
if (verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 2. Create patch interior equations\n');
    fprintf('#########################################################################\n');
end
[A,b,P.indInterior] = setupPatchInterior(grid,k,q,A,b);

%==============================================================
% 3. Create patch edge equations
%==============================================================
if (verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 3. Create patch edge equations\n');
    fprintf('#########################################################################\n');
end

%==============================================================
% 4. Create patch boundary equations
%==============================================================
if (verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 4. Create patch boundary equations\n');
    fprintf('#########################################################################\n');
end

% Add fine fluxes using DFM to equations of coarse nodes at the C/F
% interface.
AlistCF             = distributeFineFluxes(grid,k,q,P.ilower,P.iupper);
Alist               = [Alist; AlistCF];

% Create or update sparse LHS matrix
if (isempty(A))
    Anew = sparse(Alist(:,1),Alist(:,2),Alist(:,3));
else
    B = sparse(Alist(:,1),Alist(:,2),Alist(:,3));
    Anew = sparse(size(B,1),size(B,2));
    Anew(1:size(A,1),1:size(A,2)) = A;
    Anew = Anew + B;
end

% Diagonally scale the Dirichlet boundary equations to make the matrix
% symmetric. This means to pass to flux boundary nodes like the ghost nodes
% flux form below.
i = setupOperatorPatch(grid,k,q,P.ilower,P.iupper,2,0);
if (~isempty(i))
    % We would like to do that, but entries in i(:,2) are not unique; so do
    % it over each face separately, and there i(:,2) entries are unique.
    %Anew(:,i(:,2)) = Anew(:,i(:,2)) + Anew(:,i(:,1));               % Pass to flux ghost points (by means of a Gaussian elimination on the appropriate columns, in effect)
    faceSize = size(i,1)/(2*grid.dim);
    for f = 1:2*grid.dim
        face = [(f-1)*faceSize:f*faceSize-1]+1;
        Anew(:,i(face,2)) = Anew(:,i(face,2)) + Anew(:,i(face,1));               % Pass to flux ghost points (by means of a Gaussian elimination on the appropriate columns, in effect)
    end
    Anew(i(:,1),:) = diag(-i(:,3))*Anew(i(:,1),:);                  % Diagonally scale by (-flux)
    bnew(i(:,1))   = diag(-i(:,3))*bnew(i(:,1));                         % Scale RHS accordingly
end

% Pass to flux ghost points. No need to scale RHS because it's 0 for the
% ghost point equations.
fineFlux = setupOperatorPatch(grid,k,q,P.ilower,P.iupper,0,1);
if (~isempty(fineFlux))
    i = setupOperatorPatch(grid,k,q,P.ilower,P.iupper,0,2);         % Each row: [ghost fineNbhr alpha=interpCoefficient(fineNbhr->ghost)]
    % We would like to do that, but entries in i(:,2) are not unique; so do
    % it over each face separately, and there i(:,2) entries are unique.
    %Anew(:,i(:,2)) = Anew(:,i(:,2)) + Anew(:,i(:,1));               % Pass to flux ghost points (by means of a Gaussian elimination on the appropriate columns, in effect). Every entry in i(:,2) appears exactly once.
    faceSize = size(i,1)/(2*grid.dim);
    for f = 1:2*grid.dim
        face = [(f-1)*faceSize:f*faceSize-1]+1;
        Anew(:,i(face,2)) = Anew(:,i(face,2)) + Anew(:,i(face,1));               % Pass to flux ghost points (by means of a Gaussian elimination on the appropriate columns, in effect)
    end
    Anew(i(:,1),:) = Anew(:,i(:,1))';                                       % Induced interpolation stencils of ghost points is transpose of their appearance in all equations
    Anew(i(:,1),i(:,1)) = diag(1./(i(:,3)-1));                            % Set diagonal coefficient of ghost point interp stencil to 1/(alpha-1) to be consistent with the original interpolation formula
end

%==============================================================
% 5. Initialize patch unused equations
%==============================================================
if (verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 5. Initialize patch unused equations\n');
    fprintf('#########################################################################\n');
end

%==============================================================
% 6. Delete underlying coarse patch equations and replace them by the identity
% operator (including ghost equations).
%==============================================================
if (verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 6. Delete underlying coarse patch equations and replace them by the\n');
    fprintf(' identity operator (including ghost equations).\n');
    fprintf('#########################################################################\n');
end


% Delete data of coarse patch underlying the fine patch (disconnect these
% nodes from rest of coarse patch and put there the identity operator with
% zero RHS).
[Alist,bnew,grid]   = deleteUnderlyingData(grid,k,q,Alist,bnew);


%==============================================================
% 7. Modify coarse patch edge equations
%==============================================================
if (verboseLevel >= 2)
    fprintf('#########################################################################\n');
    fprintf(' 7. Modify coarse patch edge equations\n');
    fprintf('#########################################################################\n');
end


tCPU                = cputime - tStartCPU;
tElapsed            = etime(clock,tStartElapsed);

if (verboseLevel >= 1)
    fprintf('CPU time     = %f\n',tCPU);
    fprintf('Elapsed time = %f\n',tElapsed);
end
