/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  FieldMeasures.cc:  Unfinished modules
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/QuadraticTetVolField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

class FieldMeasures : public Module
{
private:
  FieldIPort *ifp;
  MatrixOPort *omp;
  GuiString nodeBased_;
  GuiInt xFlag_;
  GuiInt yFlag_;
  GuiInt zFlag_;
  GuiInt sizeFlag_;
  GuiInt valenceFlag_;
  GuiInt lengthFlag_;
  GuiInt aspectRatioFlag_;
  GuiInt elemSizeFlag_;
  MeshHandle m_;
  void measure_tetvol();
  void measure_quadtetvol();
  void measure_pointcloud();
  void measure_trisurf();
  void measure_latvol();
public:
  FieldMeasures(GuiContext* ctx);
  virtual ~FieldMeasures();

  virtual void execute();

};


DECLARE_MAKER(FieldMeasures)

FieldMeasures::FieldMeasures(GuiContext* ctx)
  : Module("FieldMeasures", ctx, Filter, "Fields", "SCIRun"),
    nodeBased_(ctx->subVar("nodeBased")),
    xFlag_(ctx->subVar("xFlag")), yFlag_(ctx->subVar("yFlag")),
    zFlag_(ctx->subVar("zFlag")), sizeFlag_(ctx->subVar("sizeFlag")),
    valenceFlag_(ctx->subVar("valenceFlag")), 
    lengthFlag_(ctx->subVar("lengthFlag")), 
    aspectRatioFlag_(ctx->subVar("aspectRatioFlag")),
    elemSizeFlag_(ctx->subVar("elemSizeFlag"))
{
}



FieldMeasures::~FieldMeasures()
{
}

void
FieldMeasures::measure_pointcloud()
{
  PointCloudMesh *mesh = dynamic_cast<PointCloudMesh *>(m_.get_rep());
  int x = xFlag_.get();
  int y = yFlag_.get();
  int z = zFlag_.get();
  int ncols = 0;
  if (x) ncols++;
  if (y) ncols++;
  if (z) ncols++;
  PointCloudMesh::Node::size_type nnodes;
  mesh->size(nnodes);
  if (ncols<=0) {
    warning("No measures selected.  Sending empty matrix");
    MatrixHandle matH=0;
    omp->send(matH);
    return;
  }
  DenseMatrix *dm = scinew DenseMatrix(nnodes, ncols);
  
  PointCloudMesh::Node::iterator ni, nie;
  mesh->begin(ni); mesh->end(nie);

  int row=0;
  int col=0;
  while (ni != nie) {
    col=0;
    Point p;
    mesh->get_center(p, *ni);
    if (x) { (*dm)[row][col]=p.x(); col++; }
    if (y) { (*dm)[row][col]=p.y(); col++; }
    if (z) { (*dm)[row][col]=p.z(); col++; }
    ++ni;
    row++;
  }

  MatrixHandle matH(dm);
  omp->send(matH);
}

void
FieldMeasures::measure_trisurf()
{
  TriSurfMesh *mesh = dynamic_cast<TriSurfMesh *>(m_.get_rep());
  
  mesh->synchronize(Mesh::ALL_ELEMENTS_E);
 
  int x = xFlag_.get();
  int y = yFlag_.get();
  int z = zFlag_.get();
  int length = lengthFlag_.get();
  int valence = valenceFlag_.get();
  if (valence) {
    warning("TriSurfMesh node valence not yet implemented.");
    valence=0;
  }
  int aspectRatio = aspectRatioFlag_.get();
  if (aspectRatio) {
    warning("TriSurfMesh element aspect ratio not yet implemented.");
    aspectRatio=0;
  }
  int elemSize = elemSizeFlag_.get();
  int ncols = 0;
  if (x) ncols++;
  if (y) ncols++;
  if (z) ncols++;
  const string &type = nodeBased_.get();
  if (type == "node") {
    if (valence) ncols++;
    TriSurfMesh::Node::size_type nnodes;
    mesh->size(nnodes);
    if (ncols<=0) {
      warning("No measures selected.  Sending empty matrix");
      MatrixHandle matH=0;
      omp->send(matH);
      return;
    }
    DenseMatrix *dm = scinew DenseMatrix(nnodes, ncols);
    TriSurfMesh::Node::iterator ni, nie;
    mesh->begin(ni); mesh->end(nie);
    int row=0;
    int col=0;
    while (ni != nie) {
      col=0;
      Point p;
      mesh->get_center(p, *ni);
      if (x) { (*dm)[row][col]=p.x(); col++; }
      if (y) { (*dm)[row][col]=p.y(); col++; }
      if (z) { (*dm)[row][col]=p.z(); col++; }
      if (valence) { (*dm)[row][col]=0; col++; }  // FIXME: should be num_nbrs
      ++ni;
      row++;
    }
    MatrixHandle matH(dm);
    omp->send(matH);
  } else if (type == "edge") {
    if (length) ncols++;
    TriSurfMesh::Edge::size_type nedges;
    mesh->size(nedges);
    if (ncols<=0) {
      warning("No measures selected.  Sending empty matrix");
      MatrixHandle matH=0;
      omp->send(matH);
      return;
    }
    DenseMatrix *dm = scinew DenseMatrix(nedges, ncols);
    TriSurfMesh::Edge::iterator ni, nie;
    mesh->begin(ni); mesh->end(nie);
    int row=0;
    int col=0;
    while (ni != nie) {
      col=0;
      Point p,p0,p1;
      mesh->get_center(p, *ni);
      TriSurfMesh::Node::array_type nodes;
      mesh->get_nodes(nodes, *ni);
      mesh->get_center(p0,nodes[0]);
      mesh->get_center(p1,nodes[1]);
      if (x) { (*dm)[row][col]=p.x(); col++; }
      if (y) { (*dm)[row][col]=p.y(); col++; }
      if (z) { (*dm)[row][col]=p.z(); col++; }
      if (length) { (*dm)[row][col]=(p1-p0).length(); col++; }
      ++ni;
      row++;
    }
    MatrixHandle matH(dm);
    omp->send(matH);
  } else if (type == "element") {
    if (aspectRatio) ncols++;
    if (elemSize) ncols++;
    TriSurfMesh::Elem::size_type nelems;
    mesh->size(nelems);
    if (ncols<=0) {
      warning("No measures selected.  Sending empty matrix");
      MatrixHandle matH=0;
      omp->send(matH);
      return;
    }
    DenseMatrix *dm = scinew DenseMatrix(nelems, ncols);
    TriSurfMesh::Elem::iterator ei, eie;
    mesh->begin(ei); mesh->end(eie);
    int row=0;
    int col=0;
    while (ei != eie) {
      col=0;
      Point p;
      mesh->get_center(p, *ei);
      if (x) { (*dm)[row][col]=p.x(); col++; }
      if (y) { (*dm)[row][col]=p.y(); col++; }
      if (z) { (*dm)[row][col]=p.z(); col++; }
      if (elemSize) { (*dm)[row][col]=mesh->get_area(*ei); col++; }
      if (aspectRatio) { (*dm)[row][col]=0; col++; }  // FIXME: should be aspect_ratio
      ++ei;
      row++;
    }
    MatrixHandle matH(dm);
    omp->send(matH);
  } else {
    warning("Unkown element type.");
  }

}

void
FieldMeasures::measure_tetvol()
{
  TetVolMesh *mesh = dynamic_cast<TetVolMesh *>(m_.get_rep());

  mesh->synchronize(Mesh::ALL_ELEMENTS_E);

  int x = xFlag_.get();
  int y = yFlag_.get();
  int z = zFlag_.get();
  int length = lengthFlag_.get();
  int valence = valenceFlag_.get();
  int aspectRatio = aspectRatioFlag_.get();
  if (aspectRatio) {
    warning("TetVolMesh element aspect ratio not yet implemented.");
    aspectRatio=0;
  }
  int elemSize = elemSizeFlag_.get();
  int ncols = 0;
  if (x) ncols++;
  if (y) ncols++;
  if (z) ncols++;
  const string &type = nodeBased_.get();
  if (type == "node") {
    if (valence) ncols++;
    TetVolMesh::Node::size_type nnodes;
    mesh->size(nnodes);
    if (ncols<=0) {
      warning("No measures selected.  Sending empty matrix");
      MatrixHandle matH=0;
      omp->send(matH);
      return;
    }
    DenseMatrix *dm = scinew DenseMatrix(nnodes, ncols);
    TetVolMesh::Node::iterator ni, nie;
    mesh->begin(ni); mesh->end(nie);
    int row=0;
    int col=0;
    TetVolMesh::Node::array_type nbrs;
    while (ni != nie) {
      col=0;
      Point p;
      mesh->get_center(p, *ni);
      if (x) { (*dm)[row][col]=p.x(); col++; }
      if (y) { (*dm)[row][col]=p.y(); col++; }
      if (z) { (*dm)[row][col]=p.z(); col++; }
      if (valence) { 
	mesh->get_neighbors(nbrs, *ni); 
	(*dm)[row][col]=nbrs.size(); 
	col++; 
      }
      ++ni;
      row++;
    }
    MatrixHandle matH(dm);
    omp->send(matH);
  } else if (type == "edge") {
    if (length) ncols++;
    TetVolMesh::Edge::size_type nedges;
    mesh->size(nedges);
    if (ncols<=0) {
      warning("No measures selected.  Sending empty matrix");
      MatrixHandle matH=0;
      omp->send(matH);
      return;
    }
    DenseMatrix *dm = scinew DenseMatrix(nedges, ncols);
    TetVolMesh::Edge::iterator ni, nie;
    mesh->begin(ni); mesh->end(nie);
    int row=0;
    int col=0;
    while (ni != nie) {
      col=0;
      Point p,p0,p1;
      mesh->get_center(p, *ni);
      TetVolMesh::Node::array_type nodes;
      mesh->get_nodes(nodes, *ni);
      mesh->get_center(p0,nodes[0]);
      mesh->get_center(p1,nodes[1]);
      if (x) { (*dm)[row][col]=p.x(); col++; }
      if (y) { (*dm)[row][col]=p.y(); col++; }
      if (z) { (*dm)[row][col]=p.z(); col++; }
      if (length) { (*dm)[row][col]=(p1-p0).length(); col++; }
      ++ni;
      row++;
    }
    MatrixHandle matH(dm);
    omp->send(matH);
  } else if (type == "element") {
    if (aspectRatio) ncols++;
    if (elemSize) ncols++;
    TetVolMesh::Elem::size_type nelems;
    mesh->size(nelems);
    if (ncols<=0) {
      warning("No measures selected. Sending empty matrix");
      MatrixHandle matH=0;
      omp->send(matH);
      return;
    }
    DenseMatrix *dm = scinew DenseMatrix(nelems, ncols);
    TetVolMesh::Elem::iterator ei, eie;
    mesh->begin(ei); mesh->end(eie);
    int row=0;
    int col=0;
    while (ei != eie) {
      col=0;
      Point p;
      mesh->get_center(p, *ei);
      if (x) { (*dm)[row][col]=p.x(); col++; }
      if (y) { (*dm)[row][col]=p.y(); col++; }
      if (z) { (*dm)[row][col]=p.z(); col++; }
      if (elemSize) { (*dm)[row][col]=mesh->get_volume(*ei); col++; }
      if (aspectRatio) { (*dm)[row][col]=0; col++; }  // FIXME: should be aspect_ratio
      ++ei;
      row++;
    }
    MatrixHandle matH(dm);
    omp->send(matH);
  } else {
    warning("Unkown element type.");
  }

}

void
FieldMeasures::measure_quadtetvol()
{
  QuadraticTetVolMesh *mesh = dynamic_cast<QuadraticTetVolMesh *>(m_.get_rep());
  mesh->synchronize(Mesh::ALL_ELEMENTS_E);

  int x = xFlag_.get();
  int y = yFlag_.get();
  int z = zFlag_.get();
  int length = lengthFlag_.get();
  int valence = valenceFlag_.get();
  int aspectRatio = aspectRatioFlag_.get();
  if (aspectRatio) {
    warning("QuadraticTetVolMesh element aspect ratio not yet implemented.");
    aspectRatio=0;
  }
  int elemSize = elemSizeFlag_.get();
  int ncols = 0;
  if (x) ncols++;
  if (y) ncols++;
  if (z) ncols++;
  const string &type = nodeBased_.get();
  if (type == "node") {
    if (valence) ncols++;
    QuadraticTetVolMesh::Node::size_type nnodes;
    mesh->size(nnodes);
    if (ncols<=0) {
      warning("No measures selected.  Sending empty matrix");
      MatrixHandle matH=0;
      omp->send(matH);
      return;
    }
    DenseMatrix *dm = scinew DenseMatrix(nnodes, ncols);
    QuadraticTetVolMesh::Node::iterator ni, nie;
    mesh->begin(ni); mesh->end(nie);
    int row=0;
    int col=0;
    QuadraticTetVolMesh::Node::array_type nbrs;
    while (ni != nie) {
      col=0;
      Point p;
      mesh->get_center(p, *ni);
      if (x) { (*dm)[row][col]=p.x(); col++; }
      if (y) { (*dm)[row][col]=p.y(); col++; }
      if (z) { (*dm)[row][col]=p.z(); col++; }
      if (valence) { 
	mesh->get_neighbors(nbrs, *ni); 
	(*dm)[row][col]=nbrs.size(); 
	col++; 
      }
      ++ni;
      row++;
    }
    MatrixHandle matH(dm);
    omp->send(matH);
  } else if (type == "edge") {
    if (length) ncols++;
    QuadraticTetVolMesh::Edge::size_type nedges;
    mesh->size(nedges);
    if (ncols<=0) {
      warning("No measures selected.  Sending empty matrix");
      MatrixHandle matH=0;
      omp->send(matH);
      return;
    }
    DenseMatrix *dm = scinew DenseMatrix(nedges, ncols);
    QuadraticTetVolMesh::Edge::iterator ni, nie;
    mesh->begin(ni); mesh->end(nie);
    int row=0;
    int col=0;
    while (ni != nie) {
      col=0;
      Point p,p0,p1;
      mesh->get_center(p, *ni);
      QuadraticTetVolMesh::Node::array_type nodes;
      mesh->get_nodes(nodes, *ni);
      mesh->get_center(p0,nodes[0]);
      mesh->get_center(p1,nodes[1]);
      if (x) { (*dm)[row][col]=p.x(); col++; }
      if (y) { (*dm)[row][col]=p.y(); col++; }
      if (z) { (*dm)[row][col]=p.z(); col++; }
      if (length) { (*dm)[row][col]=(p1-p0).length(); col++; }
      ++ni;
      row++;
    }
    MatrixHandle matH(dm);
    omp->send(matH);
  } else if (type == "element") {
    if (aspectRatio) ncols++;
    if (elemSize) ncols++;
    QuadraticTetVolMesh::Elem::size_type nelems;
    mesh->size(nelems);
    if (ncols<=0) {
      warning("No measures selected.  Sending empty matrix");
      MatrixHandle matH=0;
      omp->send(matH);
      return;
    }
    DenseMatrix *dm = scinew DenseMatrix(nelems, ncols);
    QuadraticTetVolMesh::Elem::iterator ei, eie;
    mesh->begin(ei); mesh->end(eie);
    int row=0;
    int col=0;
    while (ei != eie) {
      col=0;
      Point p;
      mesh->get_center(p, *ei);
      if (x) { (*dm)[row][col]=p.x(); col++; }
      if (y) { (*dm)[row][col]=p.y(); col++; }
      if (z) { (*dm)[row][col]=p.z(); col++; }
      if (elemSize) { (*dm)[row][col]=mesh->get_volume(*ei); col++; }
      if (aspectRatio) { (*dm)[row][col]=0; col++; }  // FIXME: should be aspect_ratio
      ++ei;
      row++;
    }
    MatrixHandle matH(dm);
    omp->send(matH);
  } else {
    warning("Unkown element type.");
  }

}

void
FieldMeasures::measure_latvol()
{
  LatVolMesh *mesh = dynamic_cast<LatVolMesh *>(m_.get_rep());
  int x = xFlag_.get();
  int y = yFlag_.get();
  int z = zFlag_.get();
  int length = lengthFlag_.get();
  int valence = valenceFlag_.get();
  if (valence) {
    warning("TriSurfMesh node valence not yet implemented.");
    valence=0;
  }
  int aspectRatio = aspectRatioFlag_.get();
  if (aspectRatio) {
    warning("LatVolMesh element aspect ratio not yet implemented.");
    aspectRatio=0;
  }
  int elemSize = elemSizeFlag_.get();
  if (elemSize) {
    warning("LatVolMesh element size not yet implemented.");
    elemSize=0;
  }
  int ncols = 0;
  if (x) ncols++;
  if (y) ncols++;
  if (z) ncols++;
  const string &type = nodeBased_.get();
  if (type == "node") {
    if (valence) ncols++;
    LatVolMesh::Node::size_type nnodes;
    mesh->size(nnodes);
    if (ncols<=0) {
      warning("No measures selected.  Sending empty matrix");
      MatrixHandle matH=0;
      omp->send(matH);
      return;
    }
    DenseMatrix *dm = scinew DenseMatrix(nnodes, ncols);
    LatVolMesh::Node::iterator ni, nie;
    mesh->begin(ni); mesh->end(nie);
    int row=0;
    int col=0;
    while (ni != nie) {
      col=0;
      Point p;
      mesh->get_center(p, *ni);
      if (x) { (*dm)[row][col]=p.x(); col++; }
      if (y) { (*dm)[row][col]=p.y(); col++; }
      if (z) { (*dm)[row][col]=p.z(); col++; }
      if (valence) { (*dm)[row][col]=0; col++; }  // FIXME: should be num_nbrs
      ++ni;
      row++;
    }
    MatrixHandle matH(dm);
    omp->send(matH);
  } else if (type == "edge") {
    if (length) ncols++;
    LatVolMesh::Edge::size_type nedges;
    mesh->size(nedges);
    if (ncols<=0) {
      warning("No measures selected.  Sending empty matrix");
      MatrixHandle matH=0;
      omp->send(matH);
      return;
    }
    DenseMatrix *dm = scinew DenseMatrix(nedges, ncols);
    LatVolMesh::Edge::iterator ni, nie;
    mesh->begin(ni); mesh->end(nie);
    int row=0;
    int col=0;
    while (ni != nie) {
      col=0;
      Point p,p0,p1;
      mesh->get_center(p, *ni);
      LatVolMesh::Node::array_type nodes;
      mesh->get_nodes(nodes, *ni);
      mesh->get_center(p0,nodes[0]);
      mesh->get_center(p1,nodes[1]);
      if (x) { (*dm)[row][col]=p.x(); col++; }
      if (y) { (*dm)[row][col]=p.y(); col++; }
      if (z) { (*dm)[row][col]=p.z(); col++; }
      if (length) { (*dm)[row][col]=(p1-p0).length(); col++; }
      ++ni;
      row++;
    }
    MatrixHandle matH(dm);
    omp->send(matH);
  } else if (type == "element") {
    if (aspectRatio) ncols++;
    if (elemSize) ncols++;
    LatVolMesh::Elem::size_type nelems;
    mesh->size(nelems);
    if (ncols<=0) {
      warning("No measures selected.  Sending empty matrix");
      MatrixHandle matH=0;
      omp->send(matH);
      return;
    }
    DenseMatrix *dm = scinew DenseMatrix(nelems, ncols);
    LatVolMesh::Elem::iterator ei, eie;
    mesh->begin(ei); mesh->end(eie);
    int row=0;
    int col=0;
    while (ei != eie) {
      col=0;
      Point p;
      mesh->get_center(p, *ei);
      if (x) { (*dm)[row][col]=p.x(); col++; }
      if (y) { (*dm)[row][col]=p.y(); col++; }
      if (z) { (*dm)[row][col]=p.z(); col++; }
      if (elemSize) { (*dm)[row][col]=0; col++; } // FIXME: should be volume
      if (aspectRatio) { (*dm)[row][col]=0; col++; }  // FIXME: should be aspect_ratio
      ++ei;
      row++;
    }
    MatrixHandle matH(dm);
    omp->send(matH);
  } else {
    warning("Unkown element type.");
  }

}

void
FieldMeasures::execute()
{
  ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle fieldhandle;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  if (!(ifp->get(fieldhandle) && fieldhandle.get_rep()))
  {
    return;
  }

  omp = (MatrixOPort *)get_oport("Output Measures Matrix");
  if (!omp) {
    error("Unable to initialize oport 'Output Measures Matrix'.");
    return;
  }

  m_ = fieldhandle->mesh();
  string mesh_name(m_->get_type_description()->get_name());
  
  if (mesh_name == "PointCloudMesh") 
    measure_pointcloud();
  else if (mesh_name == "TriSurfMesh") 
    measure_trisurf();
  else if (mesh_name == "TetVolMesh") 
    measure_tetvol();
  else if (mesh_name == "QuadraticTetVolMesh") 
    measure_quadtetvol();
  else if (mesh_name == "LatVolMesh") 
    measure_latvol();
  else 
    error("Unable to handle a mesh of type '" + mesh_name + "'.");

}


} // End namespace SCIRun


