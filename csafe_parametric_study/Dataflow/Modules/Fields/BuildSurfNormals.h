//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : BuildSurfNormals.h
//    Author : Martin Cole
//    Date   : Mon Feb 27 13:29:17 2006

#if !defined(BuildSurfNormals_h)
#define BuildSurfNormals_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Containers/Handle.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <algorithm>
#include <math.h>

#include <Dataflow/Modules/Fields/share.h>

namespace SCIRun {

//! This supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! BuildSurfNormalsAlgoAux from the DynamicAlgoBase 
//! they will have a pointer to.
class SCISHARE BuildSurfNormalsAlgo : public DynamicAlgoBase
{
public:
  virtual MatrixHandle execute(ProgressReporter *reporter, 
			       const MeshHandle mesh) = 0;
  
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *mesh);

};



template <class Msh>
class BuildSurfNormalsAlgoT : public BuildSurfNormalsAlgo
{
public:

  //! virtual interface. 
  virtual MatrixHandle execute(ProgressReporter *reporter, 
			       const MeshHandle mesh);

};

inline
double get_normalized_angle(const Vector &v0, const Vector &v1)
{
  double mv0 = v0.length();
  double mv1 = v1.length();
  double dot = Dot(v0, v1);
  return acos(dot / (mv0 * mv1)) / (2. * M_PI); 
}


template <class Msh>
MatrixHandle
BuildSurfNormalsAlgoT<Msh>::execute(ProgressReporter *reporter,
				    const MeshHandle mesh_untyped)
{
  // Must be a surface mesh.
  ASSERT(Msh::basis_type::domain_dimension() == 2);
  Msh *mesh = dynamic_cast<Msh*>(mesh_untyped.get_rep());
    
  typename Msh::Node::size_type nsz;    mesh->size(nsz);
  typename Msh::Node::iterator  iter;   mesh->begin(iter);
  typename Msh::Node::iterator  endi;   mesh->end(endi);

  reporter->report_progress(ProgressReporter::Starting);
  mesh->synchronize(Mesh::NODE_NEIGHBORS_E | Mesh::EDGES_E);
  int cur_idx = 0; int sz = nsz;
  DenseMatrix *omatrix = scinew DenseMatrix(sz, 3);
  //! Loop over the nodes in the mesh and calculate per node normals
  //! from the connected faces.
  while (iter != endi) {
    // update progress meter on module.
    reporter->update_progress(double(cur_idx) / (double)sz);
    Point pnts[3];
    mesh->get_center(pnts[0], *iter); 

    typename Msh::Elem::array_type farr;
    mesh->get_elems(farr, *iter);
    double total_area = 0.0;
    vector<double> areas(farr.size());
    vector<Vector> vectors(farr.size());

    // Loop over the connected faces.
    // create area weighted average vector;
    Vector norm(0.0, 0.0, 0.0);
    typename Msh::Elem::array_type::iterator fiter = farr.begin();
    while (fiter != farr.end()) {
      // cache the area.
      double area = mesh->get_area(*fiter);
      areas.push_back(area);
      total_area += area;
      // use this node as the pivot to calculate the normal
      typename Msh::Node::array_type fnodes;
      mesh->get_nodes(fnodes, *fiter);
      ++fiter;
      
      Point p0, p1, p2;
      mesh->get_center(p0, fnodes[0]);
      mesh->get_center(p2, fnodes[1]);
      mesh->get_center(p1, fnodes[2]);
      
      Vector v0 = p1 - p0;
      Vector v1 = p2 - p1;

      // cache the face vector.
      norm += Cross(v0, v1);
    }

    omatrix->put(cur_idx, 0, norm.x());
    omatrix->put(cur_idx, 1, norm.y());
    omatrix->put(cur_idx, 2, norm.z());
    ++iter; ++cur_idx;
  }
  reporter->update_progress(1.0);
  reporter->report_progress(ProgressReporter::Done);
  return MatrixHandle(omatrix);
}



} // end namespace SCIRun

#endif // BuildSurfNormals_h
