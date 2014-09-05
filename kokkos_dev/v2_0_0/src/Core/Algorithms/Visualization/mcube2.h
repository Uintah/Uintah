//
// ** SCIRun version of vtkMarchingCubesCases.hh -- added nbr info to cases **
//
// marching cubes case table for generating isosurfaces
//

#ifndef MCUBE2_H
#define MCUBE2_H

namespace SCIRun {

typedef struct {
  int edges[16];
  int nbrs;
} TRIANGLE_CASES;

/* REFERENCED */
extern int edge_tab[12][2];

//
// Edges to intersect. Three at a time form a triangle. Comments at end of line
// indicate case number (0->255) and base case number (0->15).
//

extern TRIANGLE_CASES triCases[];

} // End namespace SCIRun

#endif // MCUBE2_H
