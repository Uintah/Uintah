/*
 *  MeshToTet.h : function to load TetVolMesh from the old Mesh object.
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef SCI_FieldConverters_Mesh_To_Tet_h
#define SCI_FieldConverters_Mesh_To_Tet_h

#include <FieldConverters/Core/Datatypes/Mesh.h>

namespace FieldConverters {

// Functor to extract a Point from a NodeHandle.
struct NodePointFtor{
  Point operator()(NodeHandle nh) {
    return nh->p;
  }
};

// Functor to extract a cell from an Element*.
struct ElementCellFtor{
  int* operator()(Element *e) {
    return e->n;
  }
};

// Functor to extract a cell from an Element*.
struct FaceNeighborsFtor{
  int* operator()(Element *e) {
    return e->faces;
  }
};

void
load_mesh(Mesh *mesh, TetVolMeshHandle tvm) 
{
  // fill the nodes in the same order as the old mesh.
  NodeHandle &s = mesh->nodes[0];
  NodeHandle *begin = &s;
  NodeHandle *end = begin + mesh->nodes.size();
  tvm->fill_points(begin, end, NodePointFtor());

  // fill the cells (tets)
  Element *&e = mesh->elems[0];
  Element **begin_el = &e;
  Element **end_el = begin_el + mesh->elems.size();
  tvm->fill_cells(begin_el, end_el, ElementCellFtor());
  tvm->fill_neighbors(begin_el, end_el, FaceNeighborsFtor());
}

} // End of namespace FieldConverters 

#endif 
