#include <Packages/CardioWave/Core/Algorithms/CuthillMcKee.h>
#include <stdio.h>
#include <stdlib.h> 
#include <malloc.h>
#include <math.h>
#include <string.h>

void cuthill_mckee_bandwidth_minimization(const char *spr_in, const char *spr_out, int *reorder_map, int nrows) {
  int i;

  printf("Reading input SPR file: %s\n", spr_in);

  /* read bandwidth minimization code goes here...
     (in the meantime, we'll make the default mapping) */
  for (i=0; i<nrows; i++)
    reorder_map[i]=i;

  printf("Writing output bandwidth-minimized SPR file: %s\n", spr_out);
}
