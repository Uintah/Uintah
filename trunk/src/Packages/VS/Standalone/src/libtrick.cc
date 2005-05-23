/******************************************************************************
 * File: libtrick.cc
 *
 * Description: C source class implementation to provide an API for
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
 *
 * COLOR.C: Easy code for graph coloring
 * (source: http://mat.gsia.cmu.edu/COLOR/solvers/trick.c)
 * Author: Michael A. Trick, Carnegie Mellon University, trick+@cmu.edu
 *         http://mat.gsia.cmu.edu/COLOR/color.html
 * Last Modified: November 2, 1994
 *
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
 *  series.
 * I think I have improved the robustness of the code in my revisions. - SPD
 * Dustman: Stewart Dickson <mailto:dicksonsp@ornl.gov>
 *         <http://www.csm.ornl.gov/~dickson>
 ******************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/times.h>

#include "libtrick.h"

int greedy_clique(Dsatur *graph, bool *valid, int *clique)
{
  int i,j,k;
  int max;
  int place,done;
  int *order;
  int *weight;
  
  weight = new int[graph->num_node];

  for (i=0;i<graph->num_node;i++) clique[i] = 0;
  order = new int[graph->num_node+1];
  place = 0;
  for (i=0;i<graph->num_node;i++) {
    if (valid[i]) {
      order[place] = i;
      place++;
    }
  }
  for (i=0;i<graph->num_node;i++)
    weight[i] = 0;
  for (i=0;i<graph->num_node;i++) 
    {
      if (!valid[i]) continue;
      for (j=0;j<graph->num_node;j++) 
	{
	  if (!valid[j]) continue;
	  if (graph->adj[i][j]) weight[i]++;
	}
    }
  

  done = false;
  while (!done) {
    done = true;
    for (i=0;i<place-1;i++) {
      j = order[i];
      k = order[i+1];
      if (weight[j] < weight[k]) {
	order[i] = k;
	order[i+1] = j;
	done = false;
      }
    }
  }


  clique[order[0]] = true;
  for (i=1;i<place;i++) {
    j = order[i];
    for (k=0;k<i;k++) {
      if (clique[order[k]] && !graph->adj[j][order[k]]) break;
    }
    if (k==i) {
      clique[j] = true;
    }
    else clique[j] = false;
    
  }
  max = 0;
  for (i=0;i<place;i++) 
    if (clique[order[i]]) max ++;
  
  delete [] order;
  delete [] weight;
/*  printf("Clique found of size %d\n",max);*/
  
  return max;
} /* end greedy_clique() */

/* Target is a goal value:  once a clique is found with value target
   it is possible to return
   
   Lower is a bound representing an already found clique:  once it is
   determined that no clique exists with value better than lower, it
   is permitted to return with a suboptimal clique.

   Note, to find a clique of value 1, it is not permitted to just set
   the lower to 1:  the recursion will not work.  Lower represents a
   value that is the goal for the recursion.
*/

int max_w_clique(Dsatur *graph, bool *valid, int *clique, int lower, int target)
{
  int start,j,k;
  int incumb,new_weight;
  bool *valid1;
  int*clique1;
  int *order;
  int *value;
  int i,place,finish,done,place1,place2;
  int total_left;
  
/*  printf("entered with lower %d target %d\n",lower,target);*/
  graph->num_prob++;
  if (graph->num_prob > graph->max_prob) return -1;
  for (j=0;j<graph->num_node;j++) clique[j] = 0;
  total_left = 0;
  for (i=0;i<graph->num_node;i++)
    if (valid[i]) total_left ++;
  if (total_left < lower) {
    return 0;
  }

  order = new int[graph->num_node+1];
  value =  new int[graph->num_node];

  incumb = greedy_clique(graph, valid, clique);
  if (incumb >=target) return incumb;
  if (incumb > graph->best_clique) {
    graph->best_clique=incumb;
/*    printf("Clique of size %5d found.\n",best_clique);*/
  }
/*  printf("Greedy gave %f\n",incumb);*/
  
  place = 0;
  for (i=0;i<graph->num_node;i++) {
    if (clique[i]) {
      order[place] = i;
      total_left --;
      place++;
    }
  }
  start = place;
  for (i=0;i<graph->num_node;i++) {
    if (!clique[i]&&valid[i]) {
      order[place] = i;
      place++;
    }
  }
  finish = place;
  for (place=start;place<finish;place++) {
    i = order[place];
    value[i] = 0;
    for (j=0;j<graph->num_node;j++) {
      if (valid[j] && graph->adj[i][j]) value[i]++;
    }
  }

  done = false;
  while (!done) {
    done = true;
    for (place=start;place<finish-1;place++) {
      i = order[place];
      j = order[place+1];
      if (value[i] < value[j] ) {
	order[place] = j;
	order[place+1] = i;
	done = false;
      }
    }
  }
  delete [] value;
  for (place=start;place<finish;place++) {
    if (incumb + total_left < lower) {
      return 0;
    }
    
    j = order[place];
    total_left --;
    
    if (clique[j]) continue;
    
    valid1 = new bool[graph->num_node];
    clique1 = new int[graph->num_node];
    for (place1=0;place1 < graph->num_node;place1++) valid1[place1] = false;
    for (place1=0;place1<place;place1++) {
      k = order[place1];
      if (valid[k] && (graph->adj[j][k])){
	valid1[k] = true;
      }
      else
	valid1[k] = false;
    }
    new_weight = max_w_clique(graph,valid1,clique1,incumb-1,target-1);
    if (new_weight+1 > incumb)  {
/*      printf("Taking new\n");*/
      incumb = new_weight+1;
      for (k=0;k<graph->num_node;k++) clique[k] = clique1[k];
      clique[j] = true;
      if (incumb > graph->best_clique) {
	graph->best_clique=incumb;
/*	printf("Clique of size %5d found.\n",graph->best_clique);*/
      }
    }
    
  /*    else printf("Taking incumb\n");*/
    delete [] valid1;
    delete [] clique1;
    if (incumb >=target) break;
  }
  delete [] order;
  return(incumb);
} /* end max_w_clique() */


void
AssignColor(Dsatur *graph, int node, int color)
{
  int node1;

/*  printf("  %d color +%d\n",node,color);*/
  graph->ColorClass[node] = color;
  for (node1=0;node1<graph->num_node;node1++) 
    {
      if (node==node1) continue;
      if (graph->adj[node][node1]) 
	{
	  if (graph->ColorAdj[node1][color]==0) graph->ColorCount[node1]++;
	  graph->ColorAdj[node1][color]++;
	  graph->ColorAdj[node1][0]--;
	  if (graph->ColorAdj[node1][0] < 0) printf("ERROR on assign\n");	
	}
    }
  
} /* end AssignColor() */


void
RemoveColor(Dsatur *graph, int node, int color)
{
  int node1;
/*  printf("  %d color -%d\n",node,color);  */
  graph->ColorClass[node] = 0;
  for (node1=0;node1<graph->num_node;node1++) 
    {
      if (node==node1) continue;
      if (graph->adj[node][node1]) 
	{
	  graph->ColorAdj[node1][color]--;
	  if (graph->ColorAdj[node1][color]==0) graph->ColorCount[node1]--;
	  if (graph->ColorAdj[node1][color] < 0) printf("ERROR on assign\n");
	  graph->ColorAdj[node1][0]++;
	}
    }
  
} /* end RemoveColor() */


int color(Dsatur *graph, int i, int current_color)
{
  int j,new_val;
  int k,max,count,place;
  

  graph->prob_count++;
  if (current_color >= graph->BestColoring) return(current_color);
  if (graph->BestColoring <=graph->lb) return(graph->BestColoring);
  
  if (i >= graph->num_node) return(current_color);
/*  printf("Node %d, num_color %d\n",i,current_color);*/
  
/* Find node with maximum color_adj */
  max = -1;
  place = -1;
  for(k=0;k< graph->num_node;k++) 
    {
      if (graph->Handled[k]) continue;
/*      printf("ColorCount[%3d] = %d\n",k,ColorCount[k]);*/
      if ((graph->ColorCount[k] > max) ||
         ((graph->ColorCount[k]==max) &&
          (graph->ColorAdj[k][0] > graph->ColorAdj[place][0])))
	{
/*	  printf("Best now at %d\n",k);*/
	  max = graph->ColorCount[k];
	  place = k;
	}
    }
  if (place==-1) 
    {
      fprintf(stderr, "Graph is disconnected.  ");
      fprintf(stderr, "This code needs to be updated for that case.\n");
      exit(1);
    }

  
  graph->Order[i] = place;
  graph->Handled[place] = true;
/*  printf("Using node %d at level %d\n",place,i);*/
  for (j=1;j<=current_color;j++) 
    {
      if (!graph->ColorAdj[place][j]) 
	{
	  graph->ColorClass[place] = j;
	  AssignColor(graph, place, j);
	  new_val = color(graph, i+1, current_color);
	  if (new_val < graph->BestColoring){
	    graph->BestColoring = new_val;
	    print_colors(graph);
	  }
	  RemoveColor(graph,place,j);
	  if (graph->BestColoring<=current_color) {
	    graph->Handled[place] = false;
	    return(graph->BestColoring);
	  }
	}
    }
  if (current_color+1 < graph->BestColoring) 
    {
      graph->ColorClass[place] = current_color+1;
      AssignColor(graph,place,current_color+1);
      new_val = color(graph,i+1,current_color+1);
      if (new_val < graph->BestColoring) {
	graph->BestColoring = new_val;
	print_colors(graph);
      }
      
      RemoveColor(graph,place,current_color+1);
    }
  graph->Handled[place] = false;
  return(graph->BestColoring);
} /* end color() */

void
print_colors(Dsatur *graph) 
{
  int i,j;

  times(&(graph->buffer));
  graph->current_time = graph->buffer.tms_utime;
  
  printf("Best coloring is %d at time %7.1f\n",
         graph->BestColoring,(graph->current_time-graph->start_time)/60.0);
  
/*  for (i=0;i<num_node;i++)
    printf("Color[%3d] = %d\n",i,ColorClass[i]);*/
  for (i=0;i<graph->num_node;i++)
    for (j=0;j<graph->num_node;j++) 
      {
	if (i==j) continue;
	if (graph->adj[i][j] && (graph->ColorClass[i]==graph->ColorClass[j]))
	  printf("Error with nodes %d and %d and color %d\n",
                 i,j,graph->ColorClass[i]);
      }
} /* end print_colors() */

	


