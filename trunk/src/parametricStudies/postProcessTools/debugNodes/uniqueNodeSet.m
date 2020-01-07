#!/usr/bin/octave -qf

%______________________________________________________________________
%  This script generates a unique combination of nodes based on the variables
%   n = 10        Number of possible nodes
%   r = 8         Number of nodes in a single combination
%   nodeLo
%
%  This is useful when trying to debug a set of nodes
%
%  It prints out a parametric test for each combination:
%
%<Test>
%     <Title>42</Title>
%     <sus_cmd> mpirun -bind-to core -map-by core -np 48 sus  </sus_cmd>
%     <batchReplace tag="[jobName]" value = "_42" />
%     <batchReplace tag="[nodeList]"  value = 'ash[181,182,183,185,186,188,189,190]' />
%     <batchReplace tag="[nodes]"     value = '8-8' />
%     <batchReplace tag="[resname]"   value = 'perftest1-mc' />
%     <replace_values>
%       <entry path = "/Uintah_specification/CFD/ARCHES/TransportEqns/Sources/src/DORadiationModel/LinearSolver/res_tol" value ='1e-6' />
%     </replace_values>
%</Test>
%
%  These tests are then copied into a tst file
%______________________________________________________________________

clear all;
close all;
format short e;

n = 8;            % Number of possible nodes
r = 4;            % Number of nodes in a single combination

nodeLo  = 181;    % staring node
nodeHi  = nodeLo + n;

nodeSet = nodeLo:nodeHi;
nodeSet = [181,182,183,184,188,190,190,191];     % <<<< Node set that you wish to examine >>>>

% compute the number of possible combinations and the a matrix of combinations
nCombos = nchoosek(n,r);
combo   = nchoosek(nodeSet, r);


%__________________________________
%  print out

for i=1:nCombos
  
  % should generalize this
  if (r == 4)
    nodes = sprintf( '%i,%i,%i,%i',combo(i,1), combo(i,2), combo(i,3),combo(i,4) );
  end
  if (r == 8)
    nodes = sprintf( '%i,%i,%i,%i,%i,%i,%i,%i',combo(i,1), combo(i,2), combo(i,3), combo(i,4), combo(i,5), combo(i,6), combo(i,7),combo(i,8) );
  end
  
  printf( "<Test>\n" )
  printf( "     <Title>%i</Title>\n", i )
  printf( "     <sus_cmd> mpirun -bind-to core -map-by core -np 48 sus  </sus_cmd>\n" )
  printf( "     <batchReplace tag=\"[jobName]\"   value = \"_%i\" />\n", i )
  printf( "     <batchReplace tag=\"[nodeList]\"  value = 'ash[%s]' />\n", nodes )
  printf( "     <batchReplace tag=\"[nodes]\"     value = '%i-%i' />\n", r, r  )
  printf( "     <batchReplace tag=\"[resname]\"   value = 'perftest1-mc' />\n" )
  printf( "     <replace_values>\n" )
  printf( "       <entry path = \"/Uintah_specification/CFD/ARCHES/TransportEqns/Sources/src/DORadiationModel/LinearSolver/res_tol\" value ='1e-6' />\n")
  printf( "     </replace_values>\n" )
  printf( "</Test>\n")
end

exit
