%% 4 dimensional CQMOM moment generation
% for 2x2x2x2 number of nodes list of moments need for 4|3|2|1 permutation
%16 total nodes
clear all;
clc;

%moment indexes
i = [0 1 2 3 0 0 0 1 1 1 0 0 0 1 1 1 0 0 0 1 1 1 ...
     0 0 0 1 1 1 0 0 0 1 1 1 0 0 0 1 1 1 0 0 0 1 1 1];
j = [0 0 0 0 1 2 3 1 2 3 0 0 0 0 0 0 1 1 1 1 1 1 ...
     0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1];
k = [0 0 0 0 0 0 0 0 0 0 1 2 3 1 2 3 1 2 3 1 2 3 ...
     0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1];
l = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
     1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3];
n = length(i);  

% average values and N point locations
u0 = 1; v0 = 1; w0 = 1; ep = 0.01; 
r0 = 10; epR = 1;
weight = ones(1,16)*.125;
u = [u0+ep,u0+ep,u0+ep,u0+ep,u0+ep,u0+ep,u0+ep,u0+ep ...
     u0-ep,u0-ep,u0-ep,u0-ep,u0-ep,u0-ep,u0-ep,u0-ep];
v = [v0+ep,v0+ep,v0+ep,v0+ep,v0-ep,v0-ep,v0-ep,v0-ep ...
     v0+ep,v0+ep,v0+ep,v0+ep,v0-ep,v0-ep,v0-ep,v0-ep];
w = [w0+ep,w0+ep,w0-ep,w0-ep,w0+ep,w0+ep,w0-ep,w0-ep ...
     w0+ep,w0+ep,w0-ep,w0-ep,w0+ep,w0+ep,w0-ep,w0-ep];
r = [r0+epR,r0-epR,r0+epR,r0-epR,r0+epR,r0-epR,r0+epR,r0-epR ...
     r0+epR,r0-epR,r0+epR,r0-epR,r0+epR,r0-epR,r0+epR,r0-epR];
   
moments = zeros(n,1);  

for ii = 1:n
  moments(ii) = sum( weight .* u .^i(ii) .* v .^j(ii) .* w .^k(ii) .* r .^l(ii));
end

fprintf('\nInsert into CQMOM Tags')
fprintf('\n--------------------------------------')
for ii = 1:n
  fprintf('\n        <Moment> <m> [%g,%g,%g,%g] </m> </Moment>',i(ii),j(ii),k(ii),l(ii) )
end
fprintf('\n')

fprintf('\nInsert into Outlet BC')
fprintf('\n--------------------------------------')
for ii = 1:n
  fprintf('\n        <BCType label="m_%g%g%g%g" var="Neumann">',i(ii),j(ii),k(ii),l(ii) )
  fprintf('\n          <value> 0.0 </value>')
  fprintf('\n        </BCType>')
end
fprintf('\n')

fprintf('\nInsert into Wall BC')
fprintf('\n--------------------------------------')
for ii = 1:n
  fprintf('\n        <BCType label="m_%g%g%g%g" var="ForcedDirichlet">',i(ii),j(ii),k(ii),l(ii) )
  fprintf('\n          <value> 0.0 </value>')
  fprintf('\n        </BCType>')
end
fprintf('\n')

fprintf('\nInsert into Inlet BC')
fprintf('\n--------------------------------------')
for ii = 1:n
  fprintf('\n        <BCType label="m_%g%g%g%g" var="ForcedDirichlet">',i(ii),j(ii),k(ii),l(ii) )
  fprintf('\n          <value> %.15f </value>',moments(ii))
  fprintf('\n        </BCType>')
end
fprintf('\n')