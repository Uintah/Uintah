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

//    File   : MapScalarData.h
//    Author : Michael Callahan
//    Date   : June 2002

#if !defined(MapScalarData_h)
#define MapScalarData_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

namespace SCIRun {

class MapScalarDataAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src, double oldvalue, 
			      double newvalue) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *fsrc,
				       const TypeDescription *lsrc);
};


template <class FIELD, class LOC>
class MapScalarDataAlgoT : public MapScalarDataAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src, double oldvalue, 
			      double newvalue);
};


template <class FIELD, class LOC>
FieldHandle
MapScalarDataAlgoT<FIELD, LOC>::execute(FieldHandle field_h,
					double oldvalue,
					double newvalue)
{
  FIELD *ifield = dynamic_cast<FIELD *>(field_h.get_rep());
  FIELD *ofield = ifield->clone();

  typename FIELD::fdata_type::iterator ibi, iei, obi;
  ibi = ifield->fdata().begin();
  iei = ifield->fdata().end();
  obi = ofield->fdata().begin();

  while (ibi != iei)
  {
    typename FIELD::value_type val = *ibi;
    double tmp = (double)val;
    if (tmp == oldvalue) tmp=newvalue;
    *obi = (typename FIELD::value_type)(tmp);
    ++ibi;
    ++obi;

  }

  return ofield;
}


} // end namespace SCIRun

#endif // MapScalarData_h
