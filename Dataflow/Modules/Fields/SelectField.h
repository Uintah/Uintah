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

//    File   : SelectField.h
//    Author : Michael Callahan
//    Date   : August 2001

#if !defined(SelectField_h)
#define SelectField_h

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>
#include <Dataflow/Widgets/ScaledBoxWidget.h>
#include <Core/Datatypes/Matrix.h>

namespace SCIRun {

class SelectFieldCreateAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(MeshHandle src, Field::data_location at) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *msrc,
				       const TypeDescription *fsrc);
};


template <class MESH, class FIELD>
class SelectFieldCreateAlgoT : public SelectFieldCreateAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(MeshHandle src, Field::data_location at);
};


template <class MESH, class FIELD>
FieldHandle
SelectFieldCreateAlgoT<MESH, FIELD>::execute(MeshHandle mesh_h,
					     Field::data_location loc)
{
  MESH *msrc = dynamic_cast<MESH *>(mesh_h.get_rep());
  FieldHandle ofield = scinew FIELD(msrc, loc);
  return ofield;
}




class SelectFieldFillAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle src, ScaledBoxWidget *box, int value,
		       bool replace_p, int replace_value) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *fld,
				       const TypeDescription *loc);
};


template <class FIELD, class LOC>
class SelectFieldFillAlgoT : public SelectFieldFillAlgo
{
public:
  //! virtual interface. 
  virtual void execute(FieldHandle src, ScaledBoxWidget *box, int value,
		       bool replace_p, int replace_value);
};


template <class FIELD, class LOC>
void
SelectFieldFillAlgoT<FIELD, LOC>::execute(FieldHandle field_h,
					  ScaledBoxWidget *box,
					  int value, bool replace_p,
					  int replace_value)
{
  FIELD *field = dynamic_cast<FIELD *>(field_h.get_rep());

  typename LOC::iterator iter, eiter;
  field->get_typed_mesh()->begin(iter);
  field->get_typed_mesh()->end(eiter);
  
  Point center, right, down, in;
  box->GetPosition(center, right, down, in);

  // Rotate * Scale * Translate.
  Transform t, r;
  Point unused;
  t.load_identity();
  r.load_frame(unused, (right-center).normal(),
	       (down-center).normal(),
	       (in-center).normal());
  t.pre_trans(r);
  t.pre_scale(Vector((right-center).length(),
		     (down-center).length(),
		     (in-center).length()));
  t.pre_translate(Vector(center.x(), center.y(), center.z()));
  t.invert();

  while (iter != eiter)
  {
    Point p, ptrans;
    field->get_typed_mesh()->get_center(p, *iter);

    ptrans = t.project(p);
    if (ptrans.x() >= -1.0 && ptrans.x() < 1.0 &&
	ptrans.y() >= -1.0 && ptrans.y() < 1.0 &&
	ptrans.z() >= -1.0 && ptrans.z() < 1.0)
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
