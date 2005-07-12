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

if (param.verboseLevel >= 1)
    fprintf('--- setupPatchInterface(k = %d, q = %d) BEGIN ---\n',k,q);
end

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
    if (param.verboseLevel >= 2)
        fprintf('No parent patch\n');
    end
    Alist = [];
    Tlist = [];
    indDel = [];
    return;
end
r                           = level.refRatio;                     % Refinement ratio H./h (H=coarse meshsize)
Qlevel                      = grid.level{k-1};
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
    A(indDel,:)             = 0.0;
    A(:,indDel)             = 0.0;
    A(indDel,indDel)        = speye(length(indDel));
    b(indDel)               = 0.0;
    T(indDel,:)             = 0.0;
    T(:,indDel)             = 0.0;
end
if (param.verboseLevel >= 2)
    fprintf('# unused deleted gridpoints at parent patch = %d\n',length(indDel));   
end
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
indAll                      = [];
indTransformed              = [];

% Restore underLower,underUpper to exclude BC variables
underLower(lowerNearEdge)   = underLower(lowerNearEdge) + 1;
underUpper(upperNearEdge)   = underUpper(upperNearEdge) - 1;
if (param.verboseLevel >= 3)
    fprintf('Readjusting Qunder box to exclude BC vars\n');
    underLower
    underUpper
end

for d = 1:grid.dim,
    for s = [-1 1],
        if (param.verboseLevel >= 2)
            fprintf('  ==> (Fine Patch Face d = %d, s = %+d) ---\n',d,s);
        end
        dim = d;
        side = -s;                                                          % We look in the direction along the interface from the coarse patch into fine patch
        % Direction vector ("normal") from cell to its nbhr
        nbhrNormal      = zeros(1,grid.dim);
        nbhrNormal(dim) = side;
        sideNum         = (side+3)/2;                                       % side=-1 ==> 1; side=1 ==> 2
        fluxNum         = 2*dim+sideNum-2;
        otherDim        = setdiff(1:grid.dim,d);
        if (param.verboseLevel >= 3)
            otherDim
        end

        if (    (underLower(d) == QedgeDomain{1}(d)) | ...
                (underLower(d) == QedgeDomain{2}(d)) )
            % This face is at the domain boundary, skip it
            if (param.verboseLevel >= 2)
                fprintf('Skipping face near domain boundary\n');
            end
            continue;
        end

        qn               = P.nbhrPatch(d,(s+3)/2);
        if (qn > 0)
            % This face is near another patch of this level
            if (param.verboseLevel >= 2)
                fprintf('Skipping face near nbhring fine patch qn=%d for d=%d, s=%d\n',qn,d,s);
            end
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
        D                       = grid.dim-1;
        numPoints               = 2;
        points                  = [0:numPoints-1];              % 2nd order interpolation, lower left corner of stencil marked as (0,...,0) and is u_i (see comment below)

        % Compute interpolation points subscript offsets from u_i, where i
        % is the fine cell whose ghost mirror point will be interpolated
        % later. The same stencil holds for all i.
        temp                    = cell(D,1);
        [temp{:}]               = ind2sub(2*ones(D,1),[1:2^D]);
        numInterp               = length(temp{1});
        subInterp               = zeros(numInterp,D);
        for other = 1:D
            subInterp(:,other)  = temp{other}';
        end

        % Compute barycentric interpolation weights of points to mirror
        % m (separate for each dimension; dimension other weights
        % are w(other,:))

        % Barycentric weights, independent of m, stored in w
        w = zeros(D,numPoints);
        for other = 1:D
            w(other,1)  = 1.0;
            for j = 2:numPoints
                w(other,1:j-1)  = (points(1:j-1) - points(j)).*w(D,1:j-1);
                w(other,j)      = prod(points(j) - points(1:j-1));
            end
        end
        w       = 1.0./w;

        % interpolation weights from points to m in dimension other,
        % stored in w
        a = (r(d)-1)/(r(d)+1);
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

        % Loop over different types of fine cells with respect to a coarse
        % cell (there are 2^D types) and add connections to Alist.
        temp                    = cell(D,1);
        [temp{:}]               = ind2sub(r(otherDim),[1:prod(r(otherDim))]);
        numChilds               = length(temp{1});
        subChilds               = zeros(numChilds,D);
        for dim = 1:D
            subChilds(:,dim)    = temp{dim}' - 1;
        end
        if (param.verboseLevel >= 3)
            subInterp
            wInterp
            subChilds
        end
        matCoarseNbhr           = matCoarse;
        matCoarseNbhr{d}    = matCoarse{d} - s;
        for dim = 1:grid.dim,
            matCoarseNbhr{dim} = matCoarseNbhr{dim}(:)';
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
            fprintf('matCoarseNbhr = \n');
            matCoarseNbhr{:}
            colCoarseNbhr
            childBase
        end

        j                       = zeros(1,grid.dim);
        jump                    = zeros(1,grid.dim);
        for t = 1:numChilds,
            if (param.verboseLevel >= 2)
                fprintf('------------------- Fine cell child type t = %d ---------------\n',t);
            end
            %=====================================================================
            % Create a list of non-zeros to be added to A, consisting of the
            % connections of type-t-fine-cells at this face to their coarse
            % counterparts.
            %=====================================================================
            j(otherDim)     = subChilds(t,:);
            j(d)            = 0;
            jump            = r-1-2*j;
            jump(d)         = 0;

            % subInterpFine = offsets for interpolation stencils of mi
            subInterpFine   = zeros(numInterp,grid.dim);
            subInterpFine(:,otherDim) = (subInterp-1);
            subInterpFine   = subInterpFine .* repmat(jump,size(subInterpFine)./size(jump));
            dupInterpFine   = repmat(subInterpFine,[size(childBase,1),1]);

            % u_i's, also = lower-left corner of interpolation stencil of mi
            thisChild       = childBase + repmat(j,size(childBase)./size(j)) ...
                + repmat(P.offsetSub,size(childBase)./size(P.offsetSub));     % Patch-based indices of the fine interface cells ("childs") of type t
            matThisChild    = mat2cell(thisChild,size(thisChild,1),[1 1]);
            indThisChild    = ind(sub2ind(P.size,matThisChild{:}));
            indThisChild    = indThisChild(:);
            dupThisChild    = reshape(repmat((thisChild(:))',[size(subInterpFine,1),1]),[size(dupInterpFine,1) grid.dim]);
            %dupThisChild    = dupThisChild + repmat(P.offsetSub,size(dupThisChild)./size(P.offsetSub));      % Patch-based indices of the fine interface cells ("childs") of type t

            dupInterpFine   = dupThisChild + dupInterpFine;
            dupwInterp      = repmat(wInterp,[size(childBase,1),1]);
            matInterpFine   = mat2cell(dupInterpFine,size(dupInterpFine,1),[1 1]);
            indInterpFine   = ind(sub2ind(P.size,matInterpFine{:}));
            indInterpFine   = indInterpFine(:);

            if (param.verboseLevel >= 3)
                thisChild
                subInterpFine
                dupwInterp
                indInterpFine
                a
            end

            dupThisChild    = reshape(repmat((thisChild(:))',[size(subInterpFine,1),1]),[size(dupInterpFine,1) grid.dim]);
            matDupThisChild = mat2cell(dupThisChild,size(dupThisChild,1),[1 1]);
            indDupThisChild = ind(sub2ind(P.size,matDupThisChild{:}));
            indDupThisChild = indDupThisChild(:);
            indGhost        = indexNbhr(P,indThisChild,-nbhrNormal);
            indDupGhost     = repmat((indGhost(:))',[size(subInterpFine,1),1]);
            indDupGhost     = indDupGhost(:);

            if (param.verboseLevel >= 3)
                indCoarse
                indThisChild
                indGhost
            end
            % Create ghost equations
            Alist = [Alist; ...                                                 % We are never near boundaries according to the C/F interface existence rules
                [indGhost       indGhost             repmat(1.0/(a-1.0),size(indGhost))]; ...
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
                A(indThisChild,indGhost) = 0.0;                                         % Remove ghost flux from off-diagonal entry
            end
            Alist = [Alist; ...                                                         % We are never near boundaries according to the C/F interface existence rules
                [indThisChild   indGhost             repmat(1.0,size(indGhost))]; ...   % Ghost flux term
                [indThisChild   indThisChild         repmat(a,size(indGhost))]; ...     % Self-term (results from ui in the definition of the ghost flux = ui-gi+a*(mirrorGhostInterpTerms)
                [indDupThisChild indInterpFine      -a*dupwInterp]; ...                 % Interpolation terms
                ];
            Tlist = [Tlist; ...                                                         % We are never near boundaries according to the C/F interface existence rules
                [indGhost       indGhost             repmat(-1.0,size(indGhost))]; ...   % Ghost flux term
                [indGhost       indThisChild         repmat(1-a,size(indGhost))]; ...    % Self-term (results from ui in the definition of the ghost flux = ui-gi+a*(mirrorGhostInterpTerms)
                [indDupGhost    indInterpFine        a*dupwInterp]; ...                 % Interpolation terms
                ];

            % Add C,F,ghost nodes to global index list for the A-update
            % following this loop.
            indAll          = union(indAll,indCoarse);
            indAll          = union(indAll,indThisChild);
            indAll          = union(indAll,indGhost);
            indTransformed  = union(indTransformed,indGhost);

        end

    end
end

if (reallyUpdate)
    %=====================================================================
    % Add the links above to the relevant equations (rows) in A.
    %=====================================================================
    Anew                = spconvert([Alist; [grid.totalVars grid.totalVars 0]]);
    A(indAll,:)         = A(indAll,:) + Anew(indAll,:);                       % Do not replace the non-zeros in A here, rather add to them.

    %=====================================================================
    % Update transformation matrix T.
    %=====================================================================
    Tnew                    = spconvert([Tlist; [grid.totalVars grid.totalVars 0]]);
    T(indTransformed,:)     = Tnew(indTransformed,:);
end

if (param.verboseLevel >= 1)
    fprintf('--- setupPatchInterface(k = %d, q = %d) END ---\n',k,q);
end
