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


//    File   : IntersectTri.h
//    Author : Michael Callahan
//    Date   : Jan 2006

#if !defined(IntersectTri_h)
#define IntersectTri_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Geometry/CompGeom.h>
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <set>

namespace SCIRun {

class IntersectTriAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(ProgressReporter *reporter,
                       FieldHandle tris,
                       vector<unsigned int> &new_nodes,
                       vector<unsigned int> &new_elems) = 0;


  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftri);
};


template <class FIELD>
class IntersectTriAlgoT : public IntersectTriAlgo
{
public:

  //! virtual interface. 
  virtual void execute(ProgressReporter *reporter,
                       FieldHandle tris,
                       vector<unsigned int> &new_nodes,
                       vector<unsigned int> &new_elems);
};


template <class FIELD>
void
IntersectTriAlgoT<FIELD>::execute(ProgressReporter *reporter,
                                  FieldHandle tris_h,
                                  vector<unsigned int> &new_nodes,
                                  vector<unsigned int> &new_elems)
{
  FIELD *tfield = dynamic_cast<FIELD *>(tris_h.get_rep());
  typename FIELD::mesh_handle_type tmesh = tfield->get_typed_mesh();

  tmesh->synchronize(Mesh::EDGES_E | Mesh::EDGE_NEIGHBORS_E
                     | Mesh::FACES_E | Mesh::LOCATE_E);

  typename FIELD::mesh_type::Elem::iterator abi, aei;
  tmesh->begin(abi);
  tmesh->end(aei);

  vector<Point> newpoints;
  
  while (abi != aei)
  {
    typename FIELD::mesh_type::Node::array_type anodes;
    tmesh->get_nodes(anodes, *abi);
    Point apoints[3];
    for (int i = 0; i < 3; i++)
    {
      tmesh->get_point(apoints[i], anodes[i]);
    }

    typename FIELD::mesh_type::Elem::iterator bbi, bei;
    tmesh->begin(bbi);
    tmesh->end(bei);

    while (bbi != bei)
    {
      if (abi != bbi)
      {
        typename FIELD::mesh_type::Node::array_type bnodes;
        tmesh->get_nodes(bnodes, *bbi);
        Point bpoints[3];
        for (int i = 0; i < 3; i++)
        {
          tmesh->get_point(bpoints[i], bnodes[i]);
        }

        TriTriIntersection(apoints[0], apoints[1], apoints[2],
                           bpoints[0], bpoints[1], bpoints[2],
                           newpoints);
      }
      ++bbi;
    }
    ++abi;
  }      

  typename FIELD::mesh_type::Node::index_type newnode;
  typename FIELD::mesh_type::Elem::array_type newelems;
  for (unsigned int i = 0; i < newpoints.size(); i++)
  {
    Point closest;
    vector<typename FIELD::mesh_type::Elem::index_type> elem;
    tmesh->find_closest_elems(closest, elem, newpoints[i]);
    for (unsigned int k = 0; k < elem.size(); k++)
    {
      newelems.clear();
      tmesh->insert_node_in_face(newelems, newnode, elem[k], closest);
      new_nodes.push_back(newnode);
      for (int j = newelems.size()-1; j >= 0; j--)
      {
        new_elems.push_back(newelems[j]);
      }
    }
  }

  tmesh->synchronize(Mesh::EDGES_E);

  tfield->resize_fdata();
}


} // end namespace SCIRun

#endif // IntersectTri_h
