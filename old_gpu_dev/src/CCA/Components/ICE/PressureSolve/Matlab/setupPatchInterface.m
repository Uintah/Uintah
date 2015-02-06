function [A,b,T,Alist,Tlist,indDel] = setupPatchInterface(grid,k,q,A,b,T,reallyUpdate)
%SETUPPATCHINTERFACE  Set the discrete operator at coarse-fine interface.
%   [A,B,T,ALIST,TLIST,INDDEL] = SETUPINTERFACE(GRID,K,Q,A,B,T,FLAG)
%   updates the sparse LHS matrix A and the RHS matrix B of the linear
%   system, adding to them all the equations at coarse-fine interface on
%   the coarse side (subtracting the original coarse flux and
%   adding instead hc/hf fine fluxes that use ghost points).
%   Level K is the coarse level, K+1 is the fine level.
%   Equations for ghost points are constructed. The transformation
%   matrix T is also updated; ALIST and TLIST are list-of-nonzeros added to
%   A and to T, respectively. INDDEL are the level K (coarse level) indices
%   deleted from level K underneath the (K+1)-level patch.
%
%   See also: TESTDISC, ADDGRIDPATCH, SETUPPATCHINTERIOR.

% Revision history:
% 12-JUL-2005    Oren Livne    Updated comments

globalParams;

out(2,'--- setupPatchInterface(k = %d, q = %d) BEGIN ---\n',k,q);

if (nargin < 6)
    error('Too few input arguments (need at least grid,k,q,A,b)\n');
end

if (nargin < 7)
    reallyUpdate = 1;
end

%=====================================================================
% Initialize; set fine patch "pointers" (in matlab: we actually copy P).
%=====================================================================
level                   = grid.level{k};
numPatches              = length(level.numPatches);
h                       = level.h;                                  % Fine meshsize h
P                       = grid.level{k}.patch{q};
ind                     = P.cellIndex;                              % Global 1D indices of cells
edgeDomain              = cell(2,1);                                % Domain edges
edgeDomain{1}           = level.minCell + P.offsetSub;              % First domain cell - next to left domain boundary - patch-based sub
edgeDomain{2}           = level.maxCell + P.offsetSub;              % Last domain cell - next to right domain boundary - patch-based sub
e                       = eye(grid.dim);

%=====================================================================
% Set coarse patch "pointers" (in matlab: we actually copy Q).
% Find Q-indices (interior,edge,BC) that lie under the fine patch and
% delete them (indDelete).
%=====================================================================
if (P.parent < 0)                                                 % Base patch at coarsest level, nothing to delete
    out(2,'No parent patch\n');
    Alist = [];
    Tlist = [];
    indDel = [];
    return;
end
r                           = level.refRatio;                     % Refinement ratio H./h (H=coarse meshsize)
Qlevel                      = grid.level{k-1};
hc                          = Qlevel.h;                           % Coarse meshsize h
Q                           = Qlevel.patch{P.parent};             % Parent patch
QedgeDomain                 = cell(2,1);                          % Domain edges
QedgeDomain{1}              = Qlevel.minCell;                     % First domain cell - next to left domain boundary - patch-based sub
QedgeDomain{2}              = Qlevel.maxCell;                     % Last domain cell - next to right domain boundary - patch-based sub

underLower                  = coarsenIndex(grid,k,P.ilower);      % level based sub
underUpper                  = coarsenIndex(grid,k,P.iupper);      % level based sub
if (param.verboseLevel >= 3)
    underLower
    underUpper
end

% underLower,underUpper are inside Q, so add to them BC vars whenever they
% are near the boundary.
lowerNearEdge               = find(underLower == QedgeDomain{1});
underLower(lowerNearEdge)   = underLower(lowerNearEdge) - 1;
upperNearEdge               = find(underUpper == QedgeDomain{2});
underUpper(upperNearEdge)   = underUpper(upperNearEdge) + 1;
if (param.verboseLevel >= 3)
    underLower
    underUpper
end

% Delete the equations at indDel. Note that there still remain connections
% from equations outside the deleted box to indDel variables.
%indDel                      = cell(grid.dim,1);
[indDel,del,matDel]         = indexBox(Q,underLower,underUpper);
if (reallyUpdate)
    A           = deleteRows(A,indDel,indDel,1);
    b(indDel)   = 0.0;
    T           = deleteRows(T,indDel,indDel,0);
end
out(2,'# unused deleted gridpoints at parent patch = %d\n',length(indDel));
if (param.verboseLevel >= 3)
    indDel
    A(indDel,:)
end

% Delete remaining connections from outside the deleted box (indOut) to the
% deleted box (indDel).
[temp1,temp2,temp3,Alist]   = setupPatchInterior(grid,k-1,P.parent,A,b,T,underLower,underUpper,0);
in2out                      = Alist(~ismember(Alist(:,2),indDel),:);
out2in                      = [in2out(:,2) in2out(:,2) -in2out(:,3)];
alreadyDeleted              = ismember(out2in(:,1),grid.level{k-1}.indUnused);
out2in(alreadyDeleted,:)    = [];
indOut                      = unique(out2in(:,1));
Anew                        = spconvert([out2in; [grid.totalVars grid.totalVars 0]]);
if (reallyUpdate)
    A(indOut,:)             = A(indOut,:) - Anew(indOut,:);
end

%=====================================================================
% Loop over all fine patch faces.
%=====================================================================
Alist                       = zeros(0,3);
Tlist                       = zeros(0,3);
indTransformed              = [];

% Restore underLower,underUpper to exclude BC variables
underLower(lowerNearEdge)   = underLower(lowerNearEdge) + 1;
underUpper(upperNearEdge)   = underUpper(upperNearEdge) - 1;
if (param.verboseLevel >= 3)
    out(3,'Readjusting Qunder box to exclude BC vars\n');
    underLower
    underUpper
    out(3,'QedgeDomain{1} = \n');
    QedgeDomain{1}
    out(3,'QedgeDomain{2} = \n');
    QedgeDomain{2}
end

for d = 1:grid.dim,
    for s = [-1 1],
        out(2,'  ==> (Fine Patch Face d = %d, s = %+d) ---\n',d,s);
        dim = d;
        side = -s;                                                          % We look in the direction along the interface from the coarse patch into fine patch
        % Direction vector ("normal") from cell to its nbhr
        nbhrNormal      = zeros(1,grid.dim);
        nbhrNormal(dim) = side;
        sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
        fluxNum         = 2*dim+sideNum-2;
        otherDim        = setdiff(1:grid.dim,d);
        % Area of fine faces at this C/F interface
        volume          = prod(h);                                          % Assumed to be the same for all fine cells
        faceArea        = volume ./ h(dim);
        diffLength      = h(dim);                                           % "Standardized" (fine) scale for all u-differences below
        if (param.verboseLevel >= 3)
            otherDim
        end

        if (    ((s == -1) & (underLower(d) == QedgeDomain{1}(d))) | ...
                ((s ==  1) & (underUpper(d) == QedgeDomain{2}(d))) )
            % This face is at the domain boundary, skip it
            out(2,'Skipping face near domain boundary\n');
            continue;
        end

        qn               = P.nbhrPatch(d,(s+3)/2);
        if (qn > 0)
            % This face is near another patch of this level
            out(2,'Skipping face near nbhring fine patch qn=%d for d=%d, s=%d\n',qn,d,s);
            continue;
        end

        %=====================================================================
        % Prepare a list of all coarse and fine cell indices at this face.
        %=====================================================================
        % Coarse face variables
        Qilower                 = underLower;
        Qiupper                 = underUpper;
        if (s == -1)
            Qilower(d)          = Qilower(d)+s;             % To be outside the fine patch
            Qiupper(d)          = Qilower(d);
        else
            Qiupper(d)          = Qiupper(d)+s;             % To be outside the fine patch
            Qilower(d)          = Qiupper(d);
        end
        QboxSize                = Qiupper-Qilower+1;
        [indCoarse,coarse,matCoarse] = indexBox(Q,Qilower,Qiupper);
        % Coordinates of coarse points
        xCoarse                 = cell(grid.dim,1);
        for allDim = 1:grid.dim,
            xCoarse{allDim}     = (matCoarse{allDim}(:) - Q.offsetSub(allDim) - 0.5) * hc(allDim);
        end

        % Fine face variables
        ilower                  = P.ilower;
        iupper                  = P.iupper;
        if (s == -1)
            iupper(d)           = ilower(d);
        else
            ilower(d)           = iupper(d);
        end
        boxSize                 = iupper-ilower+1;
        [indFine,fine,matFine]  = indexBox(P,ilower,iupper);

        %=====================================================================
        % Compute interpolation stencil of "ghost mirror" points m_i.
        %=====================================================================
        a                       = (r(d)-1)/(r(d)+1);
        D                       = grid.dim-1;
        numPoints               = 2;
        points                  = [0:numPoints-1];              % 2nd order interpolation, lower left corner of stencil marked as (0,...,0) and is u_i (see comment below)

        % Compute interpolation points subscript offsets from u_i, where i
        % is the fine cell whose ghost mirror point will be interpolated
        % later. The same stencil holds for all i.
        subInterp               = graycode(D,repmat(numPoints,[D 1])) + 1;
        numInterp               = size(subInterp,1);

        % Compute barycentric interpolation weights of points to mirror
        % m (separate for each dimension; dimension other weights
        % are w(other,:))

        % Barycentric weights for every dimension, independent of m, stored in w
        w = zeros(D,numPoints);
        for other = 1:D
            w(other,1)  = 1.0;
            for j = 2:numPoints
                w(other,1:j-1)  = (points(1:j-1) - points(j)).*w(other,1:j-1);
                w(other,j)      = prod(points(j) - points(1:j-1));
            end
        end
        w       = 1.0./w;

        % interpolation weights from points to m in dimension other,
        % stored in w
        m = ((a-1)/(2*a))*ones(D,1);                % Mirror relative location
        for other = 1:D
            w(other,:) = w(other,:)./(m(other) - points);
            w(other,:) = w(other,:)./sum(w(other,:));
        end

        % Tensor product of 1D interpolation stencils to obtain the full
        % D-dimensional interpolation stencil from points to m
        wInterp                 = ones(numInterp,1);
        for other = 1:D,
            wInterp             = wInterp .* w(other,subInterp(:,other))';
        end

        % Interpolation weights vector of ghost difference u_g - u_i ~ b1
        % uc + b2 u_i + b2 u(finenbhr,call it u_2) + ... bn u(finenbhr, call it u_n)
        % Summing by parts and using sum(b) = 0, we obtain
        % u_g - u_i ~ c1 (u1-uc) + c2 (u2-u1) + ... c_{n-1} (u_{n} -
        % u_{n-1}) as the interpolation stencil.
        param.fluxInterpOrder = 2;
        if (param.fluxInterpOrder == 1)            
            aa = 1 / (sqrt(sum((r-1).^2) - (r(d)-1).^2 + (r(d)+1).^2)/2);
            if (D == 0)
                cc = aa;
            else
                bb = [1-a ; [wInterp(1)*a-1; wInterp(2:end)*a]];
                cc       = cumsum(bb);
                cc(1) = aa;
                cc(2:end) = 0;
                cc(end)  = [];
            end
        else
            % else: fluxInterpOrder = 2, use the normal a
            if (D == 0)
                cc = 1-a;
            else
                bb = [1-a ; [wInterp(1)*a-1; wInterp(2:end)*a]];
                cc       = cumsum(bb);
                cc(end)  = [];
            end
        end
        % Loop over different types of fine cells with respect to a coarse
        % cell (there are 2^D types) and add connections to Alist.
        if (D == 0)
            numChilds = 1;
            subChilds = [0];
        else
            temp                    = cell(D,1);
            [temp{:}]               = myind2sub(r(otherDim),[1:prod(r(otherDim))]);
            numChilds               = length(temp{1});
            subChilds               = zeros(numChilds,D);
            for allDim = 1:D
                subChilds(:,allDim)    = temp{allDim}' - 1;
            end
        end
        if (param.verboseLevel >= 3)
            subInterp
            wInterp
            subChilds
        end
        matCoarseNbhr           = matCoarse;
        matCoarseNbhr{d}        = matCoarse{d} - s;
        for allDim = 1:grid.dim,
            matCoarseNbhr{allDim}  = matCoarseNbhr{allDim}(:)';
        end
        colCoarseNbhr           = cell2mat(matCoarseNbhr)';
        colCoarseNbhr           = colCoarseNbhr ...
            - repmat(Q.offsetSub,size(colCoarseNbhr)./size(Q.offsetSub));
        childBase               = refineIndex(grid,k-1,colCoarseNbhr);
        if (s == 1)
            % Because childBase returns the lower-left corner of the first
            % coarse cell under the interface, if we are at a right face,
            % add 1 to the coarse cell, get the lower left corner of that
            % coarse cell, and subtract 1 from the result.
            colCoarseNbhr(:,d) = colCoarseNbhr(:,d)+1;
            childBase          = refineIndex(grid,k-1,colCoarseNbhr);
            childBase(:,d) = childBase(:,d)-1;
        end
        if (param.verboseLevel >= 3)
            out(3,'matCoarseNbhr = \n');
            matCoarseNbhr{:}
            colCoarseNbhr
            childBase
        end

        j                   = zeros(1,grid.dim);
        for t = 1:numChilds,
            out(2,'------------------- Fine cell child type t = %d ---------------\n',t);
            %=====================================================================
            % Create a list of non-zeros to be added to A, consisting of the
            % connections of type-t-fine-cells at this face to their coarse
            % counterparts.
            %=====================================================================
            j(otherDim)     = subChilds(t,:);
            j(d)            = 0;
            jump            = r-1-2*j;
            jump(d)         = 0;

            % subInterpFine = offsets for fine cells in the interpolation
            % stencils of all mi of type t
            subInterpFine   = zeros(numInterp,grid.dim);
            subInterpFine(:,otherDim) = (subInterp-1);
            subInterpFine   = subInterpFine .* repmat(jump,size(subInterpFine)./size(jump));
            dupInterpFine   = repmat(subInterpFine,[size(childBase,1),1]);

            % u_i's, also = lower-left corner of interpolation stencil of mi
            thisChild       = childBase + repmat(j,size(childBase)./size(j)) ...
                + repmat(P.offsetSub,size(childBase)./size(P.offsetSub));     % Patch-based indices of the fine interface cells ("childs") of type t
            matThisChild    = mat2cell(thisChild,size(thisChild,1),repmat(1,[1 grid.dim]));
            indThisChild    = ind(mysub2ind(P.size,matThisChild{:}));
            indThisChild    = indThisChild(:);
            dupThisChild    = reshape(repmat((thisChild(:))',[size(subInterpFine,1),1]),[size(dupInterpFine,1) grid.dim]);
            %dupThisChild    = dupThisChild + repmat(P.offsetSub,size(dupThisChild)./size(P.offsetSub));      % Patch-based indices of the fine interface cells ("childs") of type t

            % 1D indices of fine cells in the interpolation stencil of mi
            dupInterpFine   = dupThisChild + dupInterpFine;
            matInterpFine   = mat2cell(dupInterpFine,size(dupInterpFine,1),repmat(1,[1 grid.dim]));
            indInterpFine   = ind(mysub2ind(P.size,matInterpFine{:}));
            indInterpFine   = indInterpFine(:);

            % Coordinates of bases for each stencil
            xThisChild     = cell(grid.dim,1);
            for allDim = 1:grid.dim,
                xThisChild{allDim}   = (thisChild(:,allDim) - P.offsetSub(allDim) - 0.5) * h(allDim);
            end

            % Ghost point coordinates
            xGhost      = xThisChild;
            xGhost{dim} = xThisChild{dim} + s*h(dim);

            % Ghost/child edge point coordinates
            xGhostEdge   = xThisChild;
            xGhostEdge{dim} = xThisChild{dim} + s*0.5*h(dim);

            % Coordinates of dupInterpFine points u_i
            xInterpFine     = cell(grid.dim,1);
            for allDim = 1:grid.dim,
                xInterpFine{allDim}   = (dupInterpFine(:,allDim) - P.offsetSub(allDim) - 0.5) * h(allDim);
            end

            % Coordinates of dupInterpFine points u_{i+1}
            xShifted        = cell(grid.dim,1);
            for allDim = 1:grid.dim,
                xShifted{allDim}   =  circshift(xInterpFine{allDim},1);
            end

            % Coordinates of cell edge between points u_i, u_{i+1}
            xEdge                   = cell(grid.dim,1);
            for allDim = 1:grid.dim,
                xEdge{allDim}       =  0.5*(xInterpFine{allDim} + xShifted{allDim});
            end

            % Compute a_{i,g}
            diffusionCoefGhostEdge  = (faceArea./diffLength) .* harmonicAvg(xThisChild,xGhost,xGhostEdge);

            % Compute a_{i,i+1}
            diffusionCoef = harmonicAvg(xInterpFine,xShifted,xEdge);
            numStencils = length(indCoarse);
            stencilSize = size(diffusionCoef,1) / numStencils;
            diffusionCoef(1:stencilSize:size(diffusionCoef,1)) = -1;    % Dummy values
            % So now diffusionCoef holds the coefficients a_{i,i+1} c_i
            % that multiply the difference
            % (u_{i+1}-u_{i}),i=1,...,stencilSize (u_{1} is uc, u_{2} is
            % the base child, etc.

            % Coordinates of cell edge between points u_i, u_{i+1}
            gam                 = 0.5*repmat(hc(dim),size(xCoarse{dim}))./abs(xThisChild{dim} - xCoarse{dim});
            xCFEdge             = cell(grid.dim,1);
            for allDim = 1:grid.dim,
                xCFEdge{allDim} = (1-gam).*xCoarse{allDim} + gam.*xThisChild{allDim};
            end
            % Compute a_{c,0}, denoted alpha
            alpha               = (faceArea./diffLength) * cc(1) * harmonicAvg(xThisChild,xCoarse,xCFEdge);

            % Prepare list of nbhring fine cell indices
            % u_{i+1},u_{i},u_{i+2},u_{i+1},... for the equation of u_i.
            if (D == 0)
                indNbhr = [];
                dupNbhrCoef = [];
            else
                indNbhr             = reshape([indInterpFine circshift(indInterpFine,1)]',[2*length(indInterpFine) 1]);
                indNbhr(1:2*stencilSize:length(indNbhr)) = -1;  % Dummy values
                indNbhr(2:2*stencilSize:length(indNbhr)) = -1;  % Dummy values
                indNbhr(logical(indNbhr < 0)) = [];
                diffusionCoef(logical(diffusionCoef < 0)) = [];
                dupCC               = repmat(cc(2:end),size(diffusionCoef)./size(cc(2:end)));
                nbhrCoef            = (faceArea./diffLength) .* dupCC .* diffusionCoef;
                dupNbhrCoef         = repmat(nbhrCoef,[1 2]);
                dupNbhrCoef(:,2)    = -dupNbhrCoef(:,2);
                dupNbhrCoef         = reshape(dupNbhrCoef',[2*length(nbhrCoef) 1]);
            end
            % Prepare replication of child, ghost indices to match
            % size(indNbhr)
            indGhost            = indexNbhr(P,indThisChild,-nbhrNormal);
            indDupThisChild     = reshape(repmat(indThisChild',[2*(stencilSize-1) 1]),size(indNbhr));
            indDupGhost         = reshape(repmat(indGhost',[2*(stencilSize-1) 1]),size(indNbhr));
            dupDiffusionCoefGhostEdge         = reshape(repmat(diffusionCoefGhostEdge',[2*(stencilSize-1) 1]),size(indNbhr));

            if (param.verboseLevel >= 3)
                thisChild
                subInterpFine
                indInterpFine
                diffusionCoef
                gam
                alpha
                indCoarse
                indThisChild
                indGhost
            end

            % Create ghost equations: (-1/alpha) u_g - u_c + u_i = 0
            Alist = [Alist; ...                                                 % We are never near boundaries according to the C/F interface existence rules
                [indGhost       indGhost             -1./alpha]; ...
                [indGhost       indCoarse            repmat(-1.0,size(indGhost))]; ...
                [indGhost       indThisChild         repmat(1.0,size(indGhost))] ...
                ];

            % Add (flux-based) ghost points to coarse equations
            Alist = [Alist; ...                                                         % We are never near boundaries according to the C/F interface existence rules
                [indCoarse      indGhost             repmat(-1.0,size(indGhost))]; ...
                ];

            % Add correction terms to fine grid equations due to
            % the linear interpolation (actually extrapolation) scheme
            % of ghost points. For constant interpolation of ghosts, there
            % are no correction terms.

            if (reallyUpdate)
                % Ghost points have coefficients in indThisChild's
                % equations from setupPatchInterior(). Zero out these
                % connections and replace them with the first line of the
                % appended list to Alist below.
                A(indThisChild,indThisChild) = A(indThisChild,indThisChild) + A(indThisChild,indGhost);     % Remove ghost flux from diagonal entry
                A = deleteRows(A,indThisChild,indGhost,0,1);                                         % Remove ghost flux from off-diagonal entry
            end
            Alist = [Alist; ...                                                         % We are never near boundaries according to the C/F interface existence rules
                [indThisChild       indGhost    repmat(1.0,size(indGhost))]; ...   % Ghost flux term
                [indDupThisChild    indNbhr     dupNbhrCoef]; ...
                ];

            Tlist = [Tlist; ...                                                         % We are never near boundaries according to the C/F interface existence rules
                [indGhost       indGhost             -diffusionCoefGhostEdge]; ...   % Ghost flux term
                [indGhost       indThisChild         diffusionCoefGhostEdge]; ...    % Self-term (results from ui in the definition of the ghost flux = a_{i,g}*(ui-gi - mirrorGhostInterpTerms)*F/d
                [indDupGhost    indNbhr              -dupNbhrCoef .* dupDiffusionCoefGhostEdge]; ...                 % Interpolation terms
                ];

            % Add C,F,ghost nodes to global index list for the A-update
            % following this loop.
            indTransformed  = union(indTransformed,indGhost);

        end

    end
end

if (reallyUpdate)
    %=====================================================================
    % Add the links above to the relevant equations (rows) in A.
    %=====================================================================
    [i,j,data]  = find(A);
    nz          = [i j data];
    nz          = [nz; Alist; [grid.totalVars grid.totalVars 0]];
    A           = spconvert(nz);

    %=====================================================================
    % Update transformation matrix T.
    %=====================================================================
    Tnew                    = spconvert([Tlist; [grid.totalVars grid.totalVars 0]]);
    T(indTransformed,:)     = Tnew(indTransformed,:);
end

out(2,'--- setupPatchInterface(k = %d, q = %d) END ---\n',k,q);
