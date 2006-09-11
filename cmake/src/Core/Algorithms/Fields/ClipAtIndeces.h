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
//    File   : ClipAtIndeces.h
//    Author : Martin Cole
//    Date   : Tue Aug  8 14:19:27 2006

#if !defined(ClipAtIndeces_h)
#define ClipAtIndeces_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Basis/NoData.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <vector>

#include <Core/Algorithms/Fields/share.h>

namespace SCIRun {

using std::vector;

class SCISHARE ClipAtIndecesBase : public DynamicAlgoBase
{
public:

  enum idx_target_e {
    NODES_E,
    EDGES_E,
    FACES_E,
    CELL_E
  };

  virtual FieldHandle clip_nodes(FieldHandle field, 
				 const set<unsigned int>& indeces) = 0;
  virtual FieldHandle clip_faces(FieldHandle field, 
				 const set<unsigned int>& indeces) = 0;
  
  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc);

};

template <class Field>
class ClipAtIndecesAlgo : public ClipAtIndecesBase
{
public:
  virtual FieldHandle clip_nodes(FieldHandle field, 
				 const set<unsigned int>& indeces);
  virtual FieldHandle clip_faces(FieldHandle field, 
				 const set<unsigned int>& indeces);
  
};



template<class Field>
FieldHandle
ClipAtIndecesAlgo<Field>::clip_nodes(FieldHandle field, 
				     const set<unsigned int>& indeces)
{
  typedef typename Field::value_type value_t; 
  typedef typename Field::mesh_type  imesh_t; 
  typedef typename imesh_t::Node::index_type nidx_t;

  typedef PointCloudMesh<ConstantBasis<value_t> >  mesh_t;
  typedef ConstantBasis<value_t>                 data_basis_t;
  typedef vector<value_t>                        data_collection_t;
  typedef GenericField<mesh_t, data_basis_t, data_collection_t> fld_t;
  
  Field* ifld = dynamic_cast<Field*>(field.get_rep());
  if (! ifld) return 0;

  typename Field::mesh_handle_type imesh = ifld->get_typed_mesh();

  vector<typename mesh_t::Node::index_type> new_idxs;
  mesh_t *msh = new mesh_t();
  set<unsigned int>::iterator iter = indeces.begin();
  while (iter != indeces.end()) {
    nidx_t idx = (nidx_t)*iter++;

    Point p;
    imesh->get_center(p, idx);
    new_idxs.push_back(msh->add_point(p));
  }
  
  typename fld_t::mesh_handle_type mh(msh);
  fld_t* ofld = new fld_t(mh);

  typename vector<typename mesh_t::Node::index_type>::iterator niter = 
    new_idxs.begin();

  while (iter != indeces.end()) {
    nidx_t idx = (nidx_t)*iter++;
    value_t val;
    if (! ifld->value(val, idx)) return 0;
    ofld->set_value(val, *niter++);
  }  
  
  return FieldHandle(ofld);
}

template<class Field>
FieldHandle
ClipAtIndecesAlgo<Field>::clip_faces(FieldHandle field, 
				     const set<unsigned int>& indeces)
{
  typedef typename Field::value_type value_t; 
  typedef typename Field::mesh_type  imesh_t; 
  typedef typename imesh_t::Face::index_type fidx_t;
  typedef typename imesh_t::Node::array_type narr_t;

  typedef TriSurfMesh<TriLinearLgn<Point> >    mesh_t;
  typedef NoDataBasis<double>                  data_basis_t;
  typedef vector<double>                       data_collection_t;
  typedef GenericField<mesh_t, data_basis_t, data_collection_t> fld_t;
  
  Field* ifld = dynamic_cast<Field*>(field.get_rep());
  if (! ifld) return 0;

  typename Field::mesh_handle_type imesh = ifld->get_typed_mesh();

  mesh_t *msh = new mesh_t();
  set<unsigned int>::iterator iter = indeces.begin();
  while (iter != indeces.end()) {
    fidx_t idx = (fidx_t)*iter++;

    Point p0, p1, p2;
    narr_t nodes;
    imesh->get_nodes(nodes, idx);

    if (nodes.size() == 3) {
      imesh->get_center(p0, nodes[0]);
      imesh->get_center(p1, nodes[1]);
      imesh->get_center(p2, nodes[2]);
      msh->add_triangle(p0, p1, p2);
    }
  }
  
  typename fld_t::mesh_handle_type mh(msh);
  fld_t* ofld = new fld_t(mh);

  return FieldHandle(ofld);
}


SCISHARE FieldHandle clip_nodes(FieldHandle fld, const set<unsigned int>& indeces);
SCISHARE FieldHandle clip_faces(FieldHandle fld, const set<unsigned int>& indeces);

} // namespace SCIRun

#endif //ClipAtIndeces_h
