#!/usr/bin/octave -qf
#______________________________________________________________________
# The MIT License
#
# Copyright (c) 1997-2023 The University of Utah
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#______________________________________________________________________


pkg load symbolic
clear all
#______________________________________________________________________
#  This octave script is used to find the optimal patch configuration
#  for scalabilities studies and for production runs.  We define optimal
#  as a layout that is as close to the targetPatchSize as possible for a given
#  number of nodes.
#
#  - For a scalability study the user defines grid parameter and a array
#  of nodes that will be used.
#
#  - For production runs nTotalCells and a range of nodes to test
#
#    The dependencies are Octave, Python, and SymPy. Consult the SymPy website for details on how to install SymPy.
#
#    Start Octave.
#
#    At Octave prompt type pkg install -forge symbolic.
#
#    At Octave prompt, type pkg load symbolic.
#
#    At Octave prompt, type syms x, then f = (sin(x/2))^3, diff(f, x), etc.
#______________________________________________________________________

#______________________________________________________________________
# Function that finds all of the possible patch layouts such that the product
# of the patches in each direction equals the number of cores.
#  https://www.geeksforgeeks.org/count-number-triplets-product-equal-given-number/
#______________________________________________________________________

function [layout] = findAllPatchLayouts(nCores)

  L = double( divisors(sym(nCores)) );
  len = length (L);

  layout = zeros(0,3);

  i_iter = 1:len-2;

  for i = i_iter
    j_iter = i+1:len-1;

    for j = j_iter
      k_iter = j+1:len;

      for k=k_iter

        prod = L(i) * L(j) * L(k);

        if (prod == nCores)

          this = [L(k),L(i),L(j);
                  L(j),L(i),L(k);
                  L(i),L(j),L(k);
                 ];
          layout = vertcat( layout, this );

#          printf( " match [%i,%i,%i] [%i, %i, %i], [%i, %i, %i], [%i, %i, %i] \n",
#                 i, j, k,
#                 L(i),  L(j), L(k),
#                 L(j),  L(i), L(k),
#                 L(k),  L(i), L(j) )
        end
      end
    end
  end

endfunction

#______________________________________________________________________
# Find optimal patch configurations.  The optimal
# the layout the one that most closely matches the targetPatchSize.
#______________________________________________________________________

function [optLayout,minVal] = optimalPatchConfig( nTotalCells, targetPatchSize, allPatchLayouts )
  clear diff;

  # loop over all possible patch layouts
  for i = 1:rows( allPatchLayouts )
    cells       = nTotalCells./allPatchLayouts(i,:);

    diffNorm    = norm( (targetPatchSize - cells) );

    diff(i)     = diffNorm.^2 ;

    #printf( "%i  patchLayout: [%i,%i,%i], cells:[%i, %i, %i], length: %f, diff: %f diffNorm: %f, diffCells: %f\n", i, allPatchLayouts(i,:), cells, length(i), diff(i), diffNorm, diffCells );

  end

  # Find the minimum difference and the index
  [minVal, idx] = min( diff );

  # The optimal layout is the one with the minimum difference
  optLayout = allPatchLayouts(idx,:);

endfunction

#______________________________________________________________________
# Function that adds human suffix to large numbers
#______________________________________________________________________

function [out] = intToStr(size)
  suffixes = ['', 'K', 'M', 'G', 'T'];
  base     = log( size ) / log( 1000 );

  unit = suffixes( floor(base) );
  me   = power(1000, base - floor(base) );
  out  = sprintf( "%3.2f %s", me, unit );
endfunction

#______________________________________________________________________
#   MAIN
#______________________________________________________________________

coresPerNode    = 20;                 # number of cores per node
targetPatchSize = [20,20,20];         # target number of cells per patch
analysisType    = "scaling"           # scaling or rangeOfNodes

#__________________________________
#   Inputs for find
if( strcmp( analysisType, "rangeOfNodes") )
  minNodes    = 1;                          # minimum number of nodes
  maxNodes    = 16;                          # maximum number of nodes
  nTotalCells = [[1024,648,512]];
end


#__________________________________
# inputs for scalability studies
if( strcmp( analysisType, "scaling" ) )

  # define nodes and dx for each study
  #  small
  study(1).nodes = [1; 2; 4; 8; 16; 32; 64; 128; 256; 512];
  study(1).dx    = [3, 3, 3];

  #  medium
  study(2).nodes = [4; 8; 16; 32; 64; 128; 256; 512; 1024 ];
  study(2).dx    = [1.5, 1.5, 1.5];

  # large
  study(3).nodes = [16; 32; 64; 128; 256; 512; 1024; 2048 ];
  study(3).dx    = [0.75, 0.75, 0.75];

  size = 1           # small, medium, large

  # computational domain points
  lower         = [-570.0, 0.0, -1759.5];
  upper         = [2259.0, 360.0, 559.5];

  domainSize    = upper - lower
  nTotalCells   = domainSize./study(size).dx
end

nTotalCells

s_nTotalCells   = intToStr(prod(nTotalCells) )

#______________________________________________________________________
#______________________________________________________________________
# examine user defined nodes for the optimal patch layout over an array of
# nodes
string="declare -A patches=(";


if( strcmp( analysisType, "scaling" ) )
  printf( "__________________________________  \n")
  printf( "#  Total number of cells: %s \n", s_nTotalCells)

  for j = 1:rows( study(size).nodes )
    n          = study(size).nodes(j);
    cores      = n * coresPerNode;

    allLayouts = findAllPatchLayouts(cores);

    [patches,diff] = optimalPatchConfig(nTotalCells, targetPatchSize, allLayouts);

    nPatches   = prod( patches );

    patchSize  = round( nTotalCells./patches );

    printf( "#  nodes %5i cores  %5i  Patches: [%i,%i,%i]  nPatches %5i cellsPerPatch [ %i,%i,%i ] diff: %f\n", n, cores, patches, nPatches, patchSize, diff   )

    patch=sprintf(  " [\"%i\"]=\"[%i,%i,%i]\"",n,patches);
    string=strcat(string, patch  );
  end

end

#______________________________________________________________________
# Loop over nodes from minNodes -> maxNodes

if( strcmp( analysisType, "rangeOfNodes") )
  printf( "__________________________________ Now finding optimal patch configurations between minNodes -> maxNodes \n")
  clear diff
  clear patches
  clear patchSize
  clear nPatches
  i = 0;

  for n = minNodes:maxNodes
    i ++;
    nodes(i)   = n;
    cores(i)   = n * coresPerNode;

    allLayouts = findAllPatchLayouts( cores(i) );

    [patches(i,:),diff(i)] = optimalPatchConfig(nTotalCells, targetPatchSize, allLayouts);

    nPatches(i) = prod( patches(i,:) );

    patchSize(i,:) = round( nTotalCells./patches(i,:) );

    printf( "#  nodes %5i cores  %5i  Patches: [%i,%i,%i]  nPatches %5i cellsPerPatch [ %i,%i,%i ] diff: %f\n", nodes(i), cores(i), patches(i,:), nPatches(i), patchSize(i,:), diff(i)   )

    patch=sprintf(  " [\"%i\"]=\"[%i,%i,%i]\"",n,patches);
    string=strcat(string, patch  );

  end
  [minVal, idx] = min( diff )
  printf( "__________________________________ optimal number of nodes\n")
  printf( "#  nodes %5i cores  %5i  Patches: [%i,%i,%i]  nPatches %5i cellsPerPatch [ %i,%i,%i ] diff: %f\n", nodes(idx), cores(idx), patches(idx,:), nPatches(idx), patchSize(idx,:), diff(idx)   )
end

printf("%s)",string)
printf( " here" )
