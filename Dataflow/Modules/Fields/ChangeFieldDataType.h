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

//    File   : ChangeFieldDataType.h
//    Author : McKay Davis
//    Date   : July 2002

#if !defined(ChangeFieldDataType_h)
#define ChangeFieldDataType_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

namespace SCIRun {


class ChangeFieldDataTypeAlgoCreate : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(FieldHandle fsrc_h) = 0;
  virtual void set_val_scalar(FieldHandle fout_h, 
			      unsigned ind, double val) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const string &fdstname);
};


template <class FSRC, class FOUT>
class ChangeFieldDataTypeAlgoCreateT : public ChangeFieldDataTypeAlgoCreate
{
public:

  virtual FieldHandle execute(FieldHandle fsrc_h);
  virtual void set_val_scalar(FieldHandle fout_h, 
			      unsigned ind, double val);
};


template <class FSRC, class FOUT>
FieldHandle
ChangeFieldDataTypeAlgoCreateT<FSRC, FOUT>::execute(FieldHandle fsrc_h)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(fsrc_h.get_rep());

  // Create the field with the new mesh and data location.
  FOUT *fout = scinew FOUT(fsrc->get_typed_mesh(), fsrc_h->data_at());

  // Copy the properties from old field to new field.
  *((PropertyManager *)fout) = *(PropertyManager *)fsrc;

  return fout;
}

template <class T>
bool
double_to_data_type(T &dat, double d);

template <>
bool
double_to_data_type<Vector>(Vector &dat, double d)
{
  return false;

}

template <>
bool
double_to_data_type<Tensor>(Tensor &dat, double d)
{
  return false;
}

template <class T>
bool
double_to_data_type(T &dat, double d)
{
  dat = d;
  return true;
}


template <class FSRC, class FOUT>
void
ChangeFieldDataTypeAlgoCreateT<FSRC, FOUT>::set_val_scalar(FieldHandle fout_h, 
							   unsigned ind, 
							   double val)
{
  typename FOUT::value_type dat;
  if (double_to_data_type(dat, val)) {
    FOUT *fout = dynamic_cast<FOUT *>(fout_h.get_rep());
    typedef typename FOUT::mesh_type::Node::index_type ni;
    fout->set_value(dat, (ni)ind);
  }
}


class ChangeFieldDataTypeAlgoCopy : public DynamicAlgoBase
{
public:

  virtual void execute(FieldHandle fsrc_h, FieldHandle fout_h) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *fdst);
};


template <class FSRC, class FOUT>
class ChangeFieldDataTypeAlgoCopyT : public ChangeFieldDataTypeAlgoCopy
{
public:

  //! virtual interface. 
  virtual void execute(FieldHandle fsrc_h, FieldHandle fout_h);
};


template <class FSRC, class FOUT>
void
ChangeFieldDataTypeAlgoCopyT<FSRC, FOUT>::execute(FieldHandle fsrc_h,
					FieldHandle fout_h)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(fsrc_h.get_rep());
  FOUT *fout = dynamic_cast<FOUT *>(fout_h.get_rep());

  typename FSRC::fdata_type::iterator in = fsrc->fdata().begin();
  typename FOUT::fdata_type::iterator out = fout->fdata().begin();
  typename FSRC::fdata_type::iterator end = fsrc->fdata().end();
  if (fout->data_at() == fsrc->data_at())
  {
    while (in != end)
    {
      *out = (typename FOUT::value_type)(*in);
      ++in; ++out;
    }
  }
}


} // end namespace SCIRun

#endif // ChangeFieldDataType_h
