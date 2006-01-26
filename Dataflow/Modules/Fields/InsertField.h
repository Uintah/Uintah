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


//    File   : InsertField.h
//    Author : Michael Callahan
//    Date   : Jan 2006

#if !defined(InsertField_h)
#define InsertField_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

namespace SCIRun {

class InsertFieldAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle tet, FieldHandle insert) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftet,
					    const TypeDescription *finsert);
};


template <class TFIELD, class IFIELD>
class InsertFieldAlgoT : public InsertFieldAlgo
{
public:

  //! virtual interface. 
  virtual void execute(FieldHandle tet, FieldHandle insert);
};


template <class TFIELD, class IFIELD>
void
InsertFieldAlgoT<TFIELD, IFIELD>::execute(FieldHandle tet_h,
                                          FieldHandle insert_h)
{
  TFIELD *tfield = dynamic_cast<TFIELD *>(tet_h.get_rep());
  typename TFIELD::mesh_handle_type tmesh = tfield->get_typed_mesh();
  IFIELD *ifield = dynamic_cast<IFIELD *>(insert_h.get_rep());
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  typename IFIELD::mesh_type::Node::iterator ibi, iei;
  imesh->begin(ibi);
  imesh->end(iei);

  tmesh->synchronize(Mesh::FACE_NEIGHBORS_E | Mesh::FACES_E);

  while (ibi != iei)
  {
    Point p;
    imesh->get_center(p, *ibi);

    typename TFIELD::mesh_type::Elem::index_type elem;
    tmesh->locate(elem, p);
    typename TFIELD::mesh_type::Node::index_type newnode;
    typename TFIELD::mesh_type::Elem::array_type newelems;
    tmesh->insert_node_in_cell_2(newelems, newnode, elem, p);

    ++ibi;
  }

  tfield->resize_fdata();
}


} // end namespace SCIRun

#endif // InsertField_h
