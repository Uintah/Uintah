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

//    File   : ClipField.h
//    Author : Michael Callahan
//    Date   : August 2001

#if !defined(ClipField_h)
#define ClipField_h

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>

namespace SCIRun {

class ClipFieldAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(MeshHandle src, Field::data_location at) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *fsrc);
};


template <class FIELD>
class ClipFieldAlgoT : public ClipFieldAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(MeshHandle src, Field::data_location at);
};


template <class FIELD>
FieldHandle
ClipFieldAlgoT<FIELD>::execute(MeshHandle mesh_h, Field::data_location loc)
{
  typename FIELD::mesh_type *msrc =
    dynamic_cast<typename FIELD::mesh_type *>(mesh_h.get_rep());
  FieldHandle ofield = scinew FIELD(msrc, loc);
  return ofield;
}


} // end namespace SCIRun

#endif // ClipField_h
