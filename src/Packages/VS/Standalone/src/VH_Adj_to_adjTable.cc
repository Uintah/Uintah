/******************************************************************************
 * File: VH_Adj_to_adjTable.cc
 *
 * Description: C++ code to convert data in the VH_AdjacencyMapping structure
 *              to a binary adjacency matrix for doing graph algorithms.
 *
 * Author: Stewart Dickson
 ******************************************************************************/

#include <stdio.h>
#include <strings.h>
#include <Packages/VS/Dataflow/Modules/DataFlow/labelmaps.h>
#include "libtrick.h"

void
VH_Adj_to_adjTable(VH_AdjacencyMapping *VH_Mapping, bool **adjTable,
                   int *num_edges)
{
  int i, j, *adjList;

  for(i = 0; i < VH_Mapping->get_num_names(); i++)
  {
    adjList = VH_Mapping->adjacent_to(i);
    for(j = 0; j < VH_Mapping->get_num_rel(i); j++)
    {
      if(!adjTable[i][j]) (*num_edges)++;
      adjTable[i][j] = true;
      adjTable[j][i] = true;
    } /* end for(j = 0; j < VH_Mapping->get_num_rel(i); j++) */
  } /* end for(i = 0; i < VH_Mapping->get_num_names(); i++) */
} /* end VH_Adj_to_adjTable() */

