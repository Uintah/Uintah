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

/*
 *  VectorMagnitude.h:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computering
 *   University of Utah
 *   May 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#if !defined(VectorMagnitude_h)
#define VectorMagnitude_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

namespace SCIRun {

class VectorMagnitudeAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src) = 0;

  //! support the dynamically compiled algorithm concept
#ifdef __sgi
  static CompileInfoHandle get_compile_info(const TypeDescription *iftd,
					    const TypeDescription *oftd);
#else
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd);
#endif
};


#ifdef __sgi
template< class IFIELD, class OFIELD >
class VectorMagnitudeAlgoT : public VectorMagnitudeAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src);
};

template< class IFIELD, class OFIELD >
FieldHandle
VectorMagnitudeAlgoT<IFIELD, OFIELD>::execute(FieldHandle field_h)
{
  IFIELD *ifield = (IFIELD *) field_h.get_rep();

  OFIELD *ofield = 
    scinew OFIELD(ifield->get_typed_mesh(), ifield->data_at());

  typename IFIELD::fdata_type::iterator in  = ifield->fdata().begin();
  typename IFIELD::fdata_type::iterator end = ifield->fdata().end();
  typename OFIELD::fdata_type::iterator out = ofield->fdata().begin();

  while (in != end) {
    *out = in->length();;
    ++in; ++out;
  }

  return FieldHandle( ofield );
}

#else
template< template<class> class FIELD >

class VectorMagnitudeAlgoT : public VectorMagnitudeAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src);
};

template< template<class> class FIELD >

FieldHandle
VectorMagnitudeAlgoT<FIELD>::execute(FieldHandle field_h)
{
  FIELD<Vector> *ifield = (FIELD<Vector> *) field_h.get_rep();

  FIELD<double> *ofield = 
    scinew FIELD<double>(ifield->get_typed_mesh(), ifield->data_at());

  typename FIELD<Vector>::fdata_type::iterator in  = ifield->fdata().begin();
  typename FIELD<Vector>::fdata_type::iterator end = ifield->fdata().end();
  typename FIELD<double>::fdata_type::iterator out = ofield->fdata().begin();

  while (in != end) {
    *out = in->length();;
    ++in; ++out;
  }

  return FieldHandle( ofield );
}
#endif

} // end namespace SCIRun

#endif // VectorMagnitude_h















