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
#include <Core/Datatypes/Matrix.h>

namespace SCIRun {


class ChangeFieldDataAtAlgoCreate : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *mod,
			      FieldHandle fsrc_h,
			      Field::data_location fout_at,
			      MatrixHandle &interp) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc);
};


template <class FSRC>
class ChangeFieldDataAtAlgoCreateT : public ChangeFieldDataAtAlgoCreate
{
public:

  virtual FieldHandle execute(ProgressReporter *mod,
			      FieldHandle fsrc_h,
			      Field::data_location fout_at,
			      MatrixHandle &interp);
};


template <class FSRC>
FieldHandle
ChangeFieldDataAtAlgoCreateT<FSRC>::execute(ProgressReporter *mod,
					    FieldHandle fsrc_h,
					    Field::data_location at,
					    MatrixHandle &interp)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(fsrc_h.get_rep());

  // Create the field with the new mesh and data location.
  FSRC *fout = scinew FSRC(fsrc->get_typed_mesh(), at);
  fout->resize_fdata();

  fout->copy_properties(fsrc);

  return fout;
}


} // end namespace SCIRun

#endif // ChangeFieldDataAt_h
