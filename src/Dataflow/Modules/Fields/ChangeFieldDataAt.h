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


//    File   : ChangeFieldDataAt.h
//    Author : McKay Davis
//    Date   : July 2002


#if !defined(ChangeFieldDataAt_h)
#define ChangeFieldDataAt_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>

namespace SCIRun {


class ChangeFieldDataAtAlgoCreate : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(FieldHandle fsrc_h,
			      Field::data_location fout_at,
			      bool &same_value_type_p) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const string &fdstname);
};


template <class FSRC, class FOUT>
class ChangeFieldDataAtAlgoCreateT : public ChangeFieldDataAtAlgoCreate
{
public:

  virtual FieldHandle execute(FieldHandle fsrc_h,
			      Field::data_location fout_at,
			      bool &same_value_type_p);
};


template <class FSRC, class FOUT>
FieldHandle
ChangeFieldDataAtAlgoCreateT<FSRC, FOUT>::execute(FieldHandle fsrc_h,
					  Field::data_location at,
					  bool &same_value_type_p)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(fsrc_h.get_rep());

  // Create the field with the new mesh and data location.
  FOUT *fout = scinew FOUT(fsrc->get_typed_mesh(), at);
  fout->resize_fdata();

  *((PropertyManager *)fout) = *(PropertyManager *)fsrc;

  same_value_type_p =
    (get_type_description((typename FSRC::value_type *)0)->get_name() ==
     get_type_description((typename FOUT::value_type *)0)->get_name());

  return fout;
}


} // end namespace SCIRun

#endif // ChangeFieldDataAt_h
