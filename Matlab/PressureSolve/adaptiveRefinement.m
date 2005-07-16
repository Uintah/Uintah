function [ilower,iupper,needRefinement] = adaptiveRefinement(AMR,grid,k,threshold)
%ADAPTIVEREFINEMENT  Adaptively find areas of refinements.
%   [ILOWER,IUPPER,FLAG] = ADAPTIVEREFINEMENT(AMR,GRID,K,THRESHOLD) returns
%   the extents [ILOWER,IUPPER] of a box needing refinement (for an L-shaped domain
%   Poisson problem), from the current finest level k. AMR contains the data of the
%   AMR process. AMR{l} is the data of refinement level l (composite grid, LHS matrix,
%   RHS vector, transformations T,TI, solution). Works for 2D L-shaped only
%   for now. FLAG=0 if the region needing refinement is empty, otherwise FLAG=1.
%   GRID is the grid hierarchy; THRESHOLD is a threshold for the tau
%   refinement criterion: if tau > THRESHOLD, we mark the coarse cell for
%   refinement.
%
%   See also: TESTADAPTIVE.

% Revision history:
% 12-JUL-2005    Oren Livne    Added comments

globalParams;

tStartCPU           = cputime;
tStartElapsed       = clock;

out(2,'--- adaptiveRefinement(k = %d) BEGIN ---\n',k);

% Coarse level aliases
kc      = k-1;
gridc   = AMR{kc}.grid;
Ac      = AMR{kc}.A;
bc      = AMR{kc}.b;
Tc      = AMR{kc}.TI;
TIc     = AMR{kc}.TI;
uc      = AMR{kc}.u;

% Fine level aliases
gridf   = AMR{k}.grid;
Af      = AMR{k}.A;
bf      = AMR{k}.b;
Tf      = AMR{k}.TI;
TIf     = AMR{k}.TI;
uf      = AMR{k}.u;

% Compute tau = defect correction = refinement criterion (finite volume
% discretization ==> no scaling is needed for tau)
q   = 1;
Qq   = gridf.level{k}.patch{q}.parent;
uchat           = uc;
uchat{kc}{Qq}    = coarsen(gridf,k,q,uf{k}{q});
Acuchat         = sparseToAMR(Ac*AMRToSparse(uchat,gridc,Tc,1),gridc,TIc,0);
Afuf            = sparseToAMR(Af*AMRToSparse(uf,gridf,Tf,1),gridf,TIf,0);
tau             = Acuchat{kc}{Qq} - coarsen(gridf,k,q,Afuf{k}{q});

% In the current scheme we do not coarsen boundaries, so set tau to zero at
% and next to the fine patch boundaries, assuming we do not need refinement there.
Q                           = gridf.level{kc}.patch{Qq};
[temp,QLower,QUpper]        = coarsen(gridf,k,q,Afuf{k}{q});
[indQBox,QBox,matQBox]      = indexBox(Q,Q.ilower-1,Q.iupper+1);
remove                      = [];

for dim = 1:gridf.dim,
    remove                  = union(remove,find(matQBox{dim} - (QLower(dim)+1 + Q.offsetSub(dim)) <= 0));
    remove                  = union(remove,find(matQBox{dim} - (QUpper(dim)-1 + Q.offsetSub(dim)) >= 0));
end
tau(remove) = 0.0;

% Mark cells for refinement if tau is large
[i,j]   = find(abs(tau) > threshold);
marked  = [i j];
marked = marked - repmat(Q.offsetSub,size([i j])./size(Q.offsetSub));

if (isempty(marked))
    needRefinement = 0;
    ilower = [];
    iupper = [];
else
    needRefinement = 1;
end

% Construct a superscribing block for the marked cells
Qilower     = min(marked,[],1);
Qiupper     = max(marked,[],1);

Tilower      = refineIndex(grid,k-1,Qilower);
% Because refineIndex returns the lower-left corner of the first
% coarse cell under the interface, if we are at a right face,
% add 1 to the coarse cell, get the lower left corner of that
% coarse cell, and subtract 1 from the result.
Tiupper      = refineIndex(grid,k-1,Qiupper+1)-1;

% Now translate to level k+1 subs
ilower      = refineIndex(grid,k,Tilower);
% Because refineIndex returns the lower-left corner of the first
% coarse cell under the interface, if we are at a right face,
% add 1 to the coarse cell, get the lower left corner of that
% coarse cell, and subtract 1 from the result.
iupper      = refineIndex(grid,k,Tiupper+1)-1;
if (param.verboseLevel >= 3)
    D(tau)
    % figure(1);
    % clf;
    % surf(abs(tau))
    marked
    Qilower
    Qiupper
    ilower
    iupper
    Tilower
    Tiupper
end

tCPU        = cputime - tStartCPU;
tElapsed    = etime(clock,tStartElapsed);
out(2,'CPU time     = %f\n',tCPU);
out(2,'Elapsed time = %f\n',tElapsed);
out(2,'--- adaptiveRefinement(k = %d) END ---\n',k);
