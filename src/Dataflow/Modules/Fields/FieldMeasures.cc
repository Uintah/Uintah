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
#include <Core/Datatypes/TriSurfField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

class FieldMeasures : public Module
{
private:
  FieldIPort *ifp;
  MatrixOPort *omp;
  GuiInt nodeBased_;
  GuiInt xFlag_;
  GuiInt yFlag_;
  GuiInt zFlag_;
  GuiInt sizeFlag_;
  GuiInt valenceFlag_;
  GuiInt aspectRatioFlag_;
  GuiInt elemSizeFlag_;
  MeshHandle m_;
  void measure_tetvol();
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
  int x = xFlag_.get();
  int y = yFlag_.get();
  int z = zFlag_.get();
  int valence = valenceFlag_.get();
  if (valence) {
    cerr << "FieldMeasures: Error - TriSurfMesh node valence not yet implemented.\n";
    valence=0;
  }
  int aspectRatio = aspectRatioFlag_.get();
  if (aspectRatio) {
    cerr << "FieldMeasures: Error - TriSurfMesh element aspect ratio not yet implemented.\n";
    aspectRatio=0;
  }
  int elemSize = elemSizeFlag_.get();
  int ncols = 0;
  if (x) ncols++;
  if (y) ncols++;
  if (z) ncols++;
  if (nodeBased_.get()) {
    if (valence) ncols++;
    TriSurfMesh::Node::size_type nnodes;
    mesh->size(nnodes);
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
  } else {
    if (aspectRatio) ncols++;
    if (elemSize) ncols++;
    TriSurfMesh::Elem::size_type nelems;
    mesh->size(nelems);
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
  }    
}

void
FieldMeasures::measure_tetvol()
{
  TetVolMesh *mesh = dynamic_cast<TetVolMesh *>(m_.get_rep());
  int x = xFlag_.get();
  int y = yFlag_.get();
  int z = zFlag_.get();
  int valence = valenceFlag_.get();
  int aspectRatio = aspectRatioFlag_.get();
  if (aspectRatio) {
    cerr << "FieldMeasures: Error - TetVolMesh element aspect ratio not yet implemented.\n";
    aspectRatio=0;
  }
  int elemSize = elemSizeFlag_.get();
  int ncols = 0;
  if (x) ncols++;
  if (y) ncols++;
  if (z) ncols++;
  if (nodeBased_.get()) {
    if (valence) ncols++;
    TetVolMesh::Node::size_type nnodes;
    mesh->size(nnodes);
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
  } else {
    if (aspectRatio) ncols++;
    if (elemSize) ncols++;
    TetVolMesh::Elem::size_type nelems;
    mesh->size(nelems);
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
  }    
}

void
FieldMeasures::measure_latvol()
{
  LatVolMesh *mesh = dynamic_cast<LatVolMesh *>(m_.get_rep());
  int x = xFlag_.get();
  int y = yFlag_.get();
  int z = zFlag_.get();
  int valence = valenceFlag_.get();
  if (valence) {
    cerr << "FieldMeasures: Error - TriSurfMesh node valence not yet implemented.\n";
    valence=0;
  }
  int aspectRatio = aspectRatioFlag_.get();
  if (aspectRatio) {
    cerr << "FieldMeasures: Error - LatVolMesh element aspect ratio not yet implemented.\n";
    aspectRatio=0;
  }
  int elemSize = elemSizeFlag_.get();
  if (elemSize) {
    cerr << "FieldMeasures: Error - LatVolMesh element size not yet implemented.\n";
    elemSize=0;
  }
  int ncols = 0;
  if (x) ncols++;
  if (y) ncols++;
  if (z) ncols++;
  if (nodeBased_.get()) {
    if (valence) ncols++;
    LatVolMesh::Node::size_type nnodes;
    mesh->size(nnodes);
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
  } else {
    if (aspectRatio) ncols++;
    if (elemSize) ncols++;
    LatVolMesh::Elem::size_type nelems;
    mesh->size(nelems);
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
  }    
}


void
FieldMeasures::execute()
{
  ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle fieldhandle;
  Field *field;
  if (!ifp) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!(ifp->get(fieldhandle) && (field = fieldhandle.get_rep())))
  {
    return;
  }

  omp = (MatrixOPort *)get_oport("Output Measures Matrix");
  if (!omp) {
    postMessage("Unable to initialize "+name+"'s oport\n");
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
  else if (mesh_name == "LatVolMesh") 
    measure_latvol();
  else 
    error("Unable to handle a mesh of type '" + mesh_name + "'.");

}


} // End namespace SCIRun

