function [Anew,bnew,grid] = updateSystem(grid,k,q,A,b)
%==============================================================
% Update Sparse LHS Matrix A and RHS b as a result of adding
% patch q at level k
%==============================================================
fprintf('-------------------------------------------------------------------------\n');
fprintf(' Update System (level = %d, patch = %d)\n',k,q);
fprintf('-------------------------------------------------------------------------\n');
tStartCPU        = cputime;
tStartElapsed    = clock;

level       = grid.level{k};
numPatches  = length(level.numPatches);
P           = grid.level{k}.patch{q};
map         = P.cellIndex;
Alist       = zeros(0,3);
bnew        = [b; zeros(grid.totalVars-length(b),1)];

% Create equations in the interior of fine patch and set boundary
% conditions on ghost cells at domain boundaries, but not at C/F
% interfaces.
[AlistPatch,bPatch] = setupOperatorPatch(grid,k,q,P.ilower,P.iupper,1,0);
Alist = [Alist; AlistPatch];
bnew(map(:)) = bPatch;

% Delete data of coarse patch underlying the fine patch (disconnect these
% nodes from rest of coarse patch and put there the identity operator with
% zero RHS).
[Alist,bnew,grid] = deleteUnderlyingData(grid,k,q,Alist,bnew);

% Add fine fluxes using DFM to equations of coarse nodes at the C/F
% interface.
AlistCF = distributeFineFluxes(grid,k,q,P.ilower,P.iupper);
Alist = [Alist; AlistCF];

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

tCPU        = cputime - tStartCPU;
tElapsed    = etime(clock,tStartElapsed);
fprintf('CPU time     = %f\n',tCPU);
fprintf('Elapsed time = %f\n',tElapsed);
