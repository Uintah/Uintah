/******************************************************************************
 * File: libtrick.h
 *
 * Description: C header source class definitions to provide an API for
 *              Michael Trick's simple implementation of the DSATUR algorithm
 *              for finding the minimal coloring of a graph.
 *
 *              This library has been cleaned up considerably.
 *
 * The author of this software is Michael Trick.  Copyright (c) 1994 by
 * Michael Trick.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose without fee is hereby granted, provided that this entire notice
 * is included in all copies of any software which is or includes a copy
 * or modification of this software and in all copies of the supporting
 * documentation for such software.
 * THIS SOFTWARE IS BEING PROVIDED "AS IS", WITHOUT ANY EXPRESS OR IMPLIED
 * WARRANTY.  IN PARTICULAR, NEITHER THE AUTHOR DOES NOT MAKE ANY
 * REPRESENTATION OR WARRANTY OF ANY KIND CONCERNING THE MERCHANTABILITY
 * OF THIS SOFTWARE OR ITS FITNESS FOR ANY PARTICULAR PURPOSE.
 *
 * COLOR.C: Easy code for graph coloring
 * (source: http://mat.gsia.cmu.edu/COLOR/solvers/trick.c)
 * Author: Michael A. Trick, Carnegie Mellon University, trick+@cmu.edu
 *         http://mat.gsia.cmu.edu/COLOR/color.html
 * Last Modified: November 2, 1994
 *
 * Graph is input in a file.  First line contains the number of nodes and
 * edges.  All following contain the node numbers (from 1 to n) incident to
 * each edge.  Sample:
 *
 * 4 4
 * 1 2
 * 2 3
 * 3 4
 * 1 4
 *
 * represents a four node cycle graph.
 *
 * Code is probably insufficiently debugged, but may be useful to some people.
 *
 * For more information on this code, see Anuj Mehrotra and Michael A. Trick,
 * "A column generation approach to graph coloring", GSIA Technical report
 * series.
 *
 * I think I have improved the robustness of the code in my revisions. - SPD
 * Dustman: Stewart Dickson <mailto:dicksonsp@ornl.gov>
 *         <http://www.csm.ornl.gov/~dickson>
 ******************************************************************************/

#ifndef LIBTRICK_H
#define LIBTRICK_H

#include <sys/times.h>
#include <Packages/VS/Dataflow/Modules/DataFlow/labelmaps.h>

#define MAX_RAND (2.0*(1 << 30))
#define INF 100000.0

typedef struct trickDSATUR {
    struct tms buffer;         /* structure for timing              */
    int current_time,start_time;
    double utime;
                                                                                
    bool **adj;
    int BestColoring;
    int num_node;
    int *ColorClass;
    int prob_count;
    int *Order;
    bool *Handled;
    int **ColorAdj;
    int *ColorCount;
    int lb;
    int num_prob,max_prob;

    int best_clique;
} Dsatur;

void
VH_Adj_to_adjTable(VH_AdjacencyMapping *VH_Mapping, bool **adjTable,
                   int *num_edges);

int greedy_clique(Dsatur *graph, bool *valid, int *clique);

int max_w_clique(Dsatur *graph, bool *valid,
                 int *clique, int lower, int target);

void
AssignColor(Dsatur *graph, int node, int color);

void
RemoveColor(Dsatur *graph, int node, int color);

int color(Dsatur *graph, int i, int current_color);

void
print_colors(Dsatur *graph);

#endif  // end LIBTRICK_H
