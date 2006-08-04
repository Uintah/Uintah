/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  FieldMeasures.h:  Build a matrix of measured quantities (rows) 
 *                      associated with mesh simplices (columns)
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#if !defined(FieldMeasures_h)
#define FieldMeasures_h

#include <Core/Util/ProgressReporter.h>  
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Geometry/Point.h>
#include <Core/Datatypes/Mesh.h>

namespace SCIRun {

class FieldMeasuresAlgo : public DynamicAlgoBase
{
public:
  virtual Matrix *execute(ProgressReporter *reporter,
                          MeshHandle meshH, bool x, bool y, bool z,
                          bool idx, bool size, bool nnbrs)=0;

  //! Support the dynamically compiled algorithm concept.
  static CompileInfoHandle get_compile_info(const TypeDescription *mesh_td,
					    const TypeDescription *simplex_td,
					    bool nnormals, bool fnormals);
};

template <class MESH, class SIMPLEX>
class FieldMeasuresAlgoT : public FieldMeasuresAlgo
{
public:
  //! virtual interface. 
  virtual Matrix *execute(ProgressReporter *reporter,
                          MeshHandle meshH, bool x, bool y, bool z, bool idx,
			  bool nnbrs, bool size);
};

//! MESH -- e.g. TetVolMeshHandle
//! SIMPLEX -- e.g. TetVolMesh::Node

template <class MESH, class SIMPLEX>
Matrix *
FieldMeasuresAlgoT<MESH,SIMPLEX>::execute(ProgressReporter *reporter,
                                          MeshHandle meshH, bool x, bool y, 
					  bool z, bool idx, bool size,
					  bool nnbrs)
{
  MESH *mesh = dynamic_cast<MESH *>(meshH.get_rep());
  int ncols=0;

  if (x)     ncols++;
  if (y)     ncols++;
  if (z)     ncols++;
  if (idx)   ncols++;
  if (nnbrs) ncols++;
  if (size)  ncols++;

  if (ncols==0) {
    reporter->error("No measures selected.");
    return 0;
  }

  typename SIMPLEX::size_type nsimplices;
  mesh->size(nsimplices);
  Matrix *m;
  if (ncols==1) m = scinew ColumnMatrix(nsimplices);
  else m = scinew DenseMatrix(nsimplices, ncols);

  typename SIMPLEX::array_type nbrs;

  typename SIMPLEX::iterator si, sie;
  mesh->begin(si);
  mesh->end(sie);
  Point p;
  int row=0;
  while (si != sie) {
    int col=0;
    if (x || y || z) mesh->get_center(p, *si);
    if (x)     { m->put(row, col++, p.x()); }
    if (y)     { m->put(row, col++, p.y()); }
    if (z)     { m->put(row, col++, p.z()); }
    if (idx)   { m->put(row, col++, row); }
    if (nnbrs) { m->put(row, col++, mesh->get_valence(*si)); }
    if (size)  { m->put(row, col++, mesh->get_size(*si)); }
    ++si;
    row++;
  }
  return m;
}



template <class MESH, class SIMPLEX>
class FieldMeasuresAlgoTNN : public FieldMeasuresAlgo
{
public:
  //! virtual interface. 
  virtual Matrix *execute(ProgressReporter *reporter,
                          MeshHandle meshH, bool x, bool y, bool z, bool idx,
			  bool nnbrs, bool size);
};

//! MESH -- e.g. TetVolMeshHandle
//! SIMPLEX -- e.g. TetVolMesh::Node

template <class MESH, class SIMPLEX>
Matrix *
FieldMeasuresAlgoTNN<MESH,SIMPLEX>::execute(ProgressReporter *reporter,
                                            MeshHandle meshH, bool x, bool y, 
					    bool z, bool idx, bool size,
					    bool nnbrs)
{
  MESH *mesh = dynamic_cast<MESH *>(meshH.get_rep());
  int ncols=0;

  if (x)     ncols++;
  if (y)     ncols++;
  if (z)     ncols++;
  if (idx)   ncols++;
  if (nnbrs) ncols++;
  if (size)  ncols++;
  ncols+=3;

  if (ncols==0) {
    reporter->error("No measures selected.");
    return 0;
  }

  typename SIMPLEX::size_type nsimplices;
  mesh->size(nsimplices);
  Matrix *m;
  if (ncols==1) m = scinew ColumnMatrix(nsimplices);
  else m = scinew DenseMatrix(nsimplices, ncols);

  typename SIMPLEX::array_type nbrs;

  typename SIMPLEX::iterator si, sie;
  mesh->begin(si);
  mesh->end(sie);
  Point p;
  Vector n;
  int row=0;
  while (si != sie) {
    int col=0;
    if (x || y || z) mesh->get_center(p, *si);
    if (x)     { m->put(row, col++, p.x()); }
    if (y)     { m->put(row, col++, p.y()); }
    if (z)     { m->put(row, col++, p.z()); }
    if (idx)   { m->put(row, col++, row); }
    if (nnbrs) { m->put(row, col++, mesh->get_valence(*si)); }
    if (size)  { m->put(row, col++, mesh->get_size(*si)); }

    // Add in the node normals.
    mesh->get_normal(n, *si);
    m->put(row, col++, n.x());
    m->put(row, col++, n.y());
    m->put(row, col++, n.z());

    ++si;
    row++;
  }
  return m;
}

template <class MESH, class SIMPLEX>
class FieldMeasuresAlgoTFN : public FieldMeasuresAlgo
{
public:
  //! virtual interface. 
  virtual Matrix *execute(ProgressReporter *reporter,
                          MeshHandle meshH, bool x, bool y, bool z, bool idx,
			  bool nnbrs, bool size);
};

//! MESH -- e.g. TetVolMeshHandle
//! SIMPLEX -- e.g. TetVolMesh::Node

template <class MESH, class SIMPLEX>
Matrix *
FieldMeasuresAlgoTFN<MESH,SIMPLEX>::execute(ProgressReporter *reporter,
                                            MeshHandle meshH, bool x, bool y, 
					    bool z, bool idx, bool size,
					    bool nnbrs)
{
  MESH *mesh = dynamic_cast<MESH *>(meshH.get_rep());
  int ncols=0;

  if (x)     ncols++;
  if (y)     ncols++;
  if (z)     ncols++;
  if (idx)   ncols++;
  if (nnbrs) ncols++;
  if (size)  ncols++;
  ncols+=3;

  if (ncols==0) {
    reporter->error("No measures selected.");
    return 0;
  }

  typename SIMPLEX::size_type nsimplices;
  mesh->size(nsimplices);
  Matrix *m;
  if (ncols==1) m = scinew ColumnMatrix(nsimplices);
  else m = scinew DenseMatrix(nsimplices, ncols);

  typename SIMPLEX::array_type nbrs;
  typename SIMPLEX::iterator si, sie;
  mesh->begin(si);
  mesh->end(sie);

  typename MESH::Node::array_type nodes;
  Point p0, p1, p2;
  Point p;
  Vector n;

  int row=0;
  while (si != sie) {
    int col=0;
    if (x || y || z) mesh->get_center(p, *si);
    if (x)     { m->put(row, col++, p.x()); }
    if (y)     { m->put(row, col++, p.y()); }
    if (z)     { m->put(row, col++, p.z()); }
    if (idx)   { m->put(row, col++, row); }
    if (nnbrs) { m->put(row, col++, mesh->get_valence(*si)); }
    if (size)  { m->put(row, col++, mesh->get_size(*si)); }

    // Add in the face normals.
    mesh->get_nodes(nodes, *si);
    if (nodes.size() >= 3)
    {
      mesh->get_point(p0, nodes[0]);
      mesh->get_point(p1, nodes[1]);
      mesh->get_point(p2, nodes[2]);
      n = Cross(p1-p0, p2-p0);
      n.safe_normalize();
    }
    else
    {
      n = Vector(0.0, 0.0, 0.0);
    }
    m->put(row, col++, n.x());
    m->put(row, col++, n.y());
    m->put(row, col++, n.z());

    ++si;
    row++;
  }
  return m;
}



}

#endif
