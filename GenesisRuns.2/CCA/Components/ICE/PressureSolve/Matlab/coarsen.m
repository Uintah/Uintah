function [uc,QLower,QUpper] = coarsen(grid,k,q,u)
%COARSEN  Fine-to-coarse averaging on a patch.
%   UC = COARSEN(GRID,K,Q,U) coarsens the function U defined at patch q at
%   level k, to UC defined at the parent patch part underlying the fine
%   patch.
%
%   See also: MARKREFINEMENT, COARSENINDEX, REFINEINDEX.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

globalParams;

out(2,'--- coarsen(k = %d, q = %d) BEGIN ---\n',k,q);

if (nargin ~= 4)
    error('Too few/many input arguments (need grid,k,q,u)\n');
end

%=====================================================================
% Initialize; set fine patch "pointers" (in matlab: we actually copy P).
%=====================================================================
level                       = grid.level{k};
numPatches                  = length(level.numPatches);
h                           = level.h;                              % Fine meshsize h
P                           = grid.level{k}.patch{q};
ind                         = P.cellIndex;                          % Global 1D indices of cells
[indBox,box,matBox]         = indexBox(P,P.ilower,P.iupper);            % Indices whose equations are created and added to Alist below
listBox                     = zeros(length(matBox{1}(:)),grid.dim);
for dim = 1:grid.dim,
    listBox(:,dim)          = matBox{dim}(:) - P.offsetSub(dim);
end
subBox                      = cell(1,grid.dim);
for dim = 1:grid.dim,
    subBox{dim}             = listBox(:,dim) + P.offsetSub(dim);
end
indFine                     = ind(mysub2ind(P.size,subBox{:}));

%=====================================================================
% Set coarse patch "pointers" (in matlab: we actually copy Q).
% Find Q-indices (interior,edge,BC) that lie under the fine patch.
%=====================================================================
if (P.parent < 0)                                                   % Base patch at coarsest level, nothing to delete
    error('No parent patch');
end
r                           = level.refRatio;                       % Refinement ratio H./h (H=coarse meshsize)
Qlevel                      = grid.level{k-1};
Q                           = Qlevel.patch{P.parent};               % Parent patch
indQ                        = Q.cellIndex;                          % Global 1D indices of cells
QLower                      = coarsenIndex(grid,k,P.ilower);        % level based sub
QUpper                      = coarsenIndex(grid,k,P.iupper);        % level based sub
[indQBox,QBox,matQBox]      = indexBox(Q,QLower,QUpper);
listQBox                    = coarsenIndex(grid,k,listBox);
subQBox                     = cell(1,grid.dim);
for dim = 1:grid.dim,
    subQBox{dim}            = listQBox(:,dim) + Q.offsetSub(dim);
end
indCoarse                   = indQ(mysub2ind(Q.size,subQBox{:}));

%=====================================================================
% Compute interpolation operator I : Q(under) -> P
%=====================================================================
Ilist                       = zeros(0,3);

Ilist = [Ilist; ...
    [indFine   indCoarse    repmat(1.0,size(indFine))]; ...   % Pieceiwise constant interpolation weights
    ];

I                           = spconvert([Ilist; [grid.totalVars grid.totalVars 0]]);

%=====================================================================
% Compute restriction operator = I' scaled so that the row sums are 1.
%=====================================================================
R           = I';
s           = sum(R,2);
t           = s;
t(find(s))  = 1./s(find(s));
R           = diag(t)*R;

s           = sum(I,2);
y           = s;
y(indBox)   = u(box{:});

yc          = t;
yc(find(yc))= 1e-20;
yc          = yc + R*y;
uctemp      = zeros(size(matQBox{1}));
uctemp(:)   = full(yc(indQBox));
uc          = zeros(size(indQ));
uc(QBox{:}) = uctemp;

out(2,'--- coarsen(k = %d, q = %d) END ---\n',k,q);
