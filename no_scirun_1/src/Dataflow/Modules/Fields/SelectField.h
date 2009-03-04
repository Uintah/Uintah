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


//    File   : SelectField.h
//    Author : Michael Callahan
//    Date   : August 2001

#if !defined(SelectField_h)
#define SelectField_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Clipper.h>
#include <Core/Datatypes/Matrix.h>

namespace SCIRun {

class SelectFieldCreateAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(MeshHandle src, int basis_order) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *msrc,
					    const TypeDescription *fsrc);
};


template <class MESH, class FIELD>
class SelectFieldCreateAlgoT : public SelectFieldCreateAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(MeshHandle src, int basis_order);
};


template <class MESH, class FIELD>
FieldHandle
SelectFieldCreateAlgoT<MESH, FIELD>::execute(MeshHandle mesh_h,
					     int basis_order)
{
  MESH *msrc = dynamic_cast<MESH *>(mesh_h.get_rep());
  FieldHandle ofield = scinew FIELD(msrc, basis_order);
  return ofield;
}




class SelectFieldFillAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle src, ClipperHandle clipper, int value,
		       bool replace_p, int replace_value) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fld,
					    const TypeDescription *loc);
};


template <class FIELD, class LOC>
class SelectFieldFillAlgoT : public SelectFieldFillAlgo
{
public:
  //! virtual interface. 
  virtual void execute(FieldHandle src, ClipperHandle clipper, int value,
		       bool replace_p, int replace_value);
};


template <class FIELD, class LOC>
void
SelectFieldFillAlgoT<FIELD, LOC>::execute(FieldHandle field_h,
					  ClipperHandle clipper,
					  int value, bool replace_p,
					  int replace_value)
{
  FIELD *field = dynamic_cast<FIELD *>(field_h.get_rep());

  typename LOC::iterator iter, eiter;
  field->get_typed_mesh()->begin(iter);
  field->get_typed_mesh()->end(eiter);

  while (iter != eiter)
  {
    Point p;
    field->get_typed_mesh()->get_center(p, *iter);

    if (clipper->inside_p(p))
    {
      field->set_value(value, *iter);
    }
    else if (replace_p)
    {
      field->set_value(replace_value, *iter);
    }
    ++iter;
  }
}


} // end namespace SCIRun

#endif // SelectField_h
