% Coarse grid solution
[gridc,Ac,bc,Tc,TIc,uc] = testDisc(0);

% Two-level grid solution
[gridf,Af,bf,Tf,TIf,uf] = testDisc(1);

% Compute tau
k = 2;
q = 1;
Q = gridf.level{k}.patch{q}.parent;

uchat = uc;
uchat{k-1}{Q} = coarsen(gridf,k,q,uf{k}{q});
Acuchat = sparseToAMR(Ac*AMRToSparse(uchat,gridc,Tc,1),gridc,TIc,0);
Afuf = sparseToAMR(Af*AMRToSparse(uf,gridf,Tf,1),gridf,TIf,0);
tau = Acuchat{k-1}{Q} - coarsen(gridf,k,q,Afuf{k}{q});

% In the current scheme we do not coarsen boundaries, so set tau to zero at
% and next to the tau boundaries, assuming we do not need refinement there.
[m,n] = size(tau);
tau([1:2 m-1:m],[1:n]) = 0;
tau([1:m],[1:2 n-1:n]) = 0;

D(tau)
figure(1);
clf;
surf(abs(tau))
[i,j] = find(abs(tau) > 0.001);
