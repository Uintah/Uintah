
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

//    File   : FusionSlicer.h
//    Author : Michael Callahan
//    Date   : April 2002

#if !defined(FusionSlicer_h)
#define FusionSlicer_h

#include <Core/Containers/Handle.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/StructHexVolField.h>
#include <Core/Datatypes/StructQuadSurfField.h>
#include <Core/Datatypes/StructCurveField.h>

namespace Fusion {

using namespace SCIRun;

class FusionSlicerAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src,
			      int axis) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    const TypeDescription *ttd);
};


#ifdef __sgi
template< class FIELD, class TYPE >
#else
template< template<class> class FIELD, class TYPE >
#endif
class FusionSlicerAlgoT : public FusionSlicerAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src,
			      int axis);
};


class FusionSlicerWorkAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle src,
		       FieldHandle dst,
		       unsigned int index,
		       int axis) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *iftd,
				       const TypeDescription *oftd);
};


template< class IFIELD, class OFIELD >
class FusionSlicerWorkAlgoT : public FusionSlicerWorkAlgo
{
public:
  //! virtual interface. 
  virtual void execute(FieldHandle src,
		       FieldHandle dst,
		       unsigned int index,
		       int axis);
};



#ifdef __sgi
template< class FIELD, class TYPE >
#else
template< template<class> class FIELD, class TYPE >
#endif
FieldHandle
FusionSlicerAlgoT<FIELD, TYPE>::execute(FieldHandle ifield_h,
					int axis)
{
  FieldHandle ofield_h;

  FIELD<TYPE> *ifield = (FIELD<TYPE> *) ifield_h.get_rep();
  typename FIELD<TYPE>::mesh_handle_type imesh = ifield->get_typed_mesh();

  unsigned int omx, omy, omz;
  unsigned int onx, ony, onz;

  unsigned int mx, my, nx, ny;

  Array1<unsigned int> dim = imesh->get_dim();
  Array1<unsigned int> min = imesh->get_min();

  if( dim.size() == 3 ) {
    omx = min[0];
    omy = min[1];
    omz = min[2];

    onx = dim[0];
    ony = dim[1];
    onz = dim[2];
  } else if( dim.size() == 2 ) {
    omx = min[0];
    omy = min[1];
    omy = 0;

    onx = dim[0];
    ony = dim[1];
    onz = 1;
  }

  if (axis == 0) {
    nx = ony;
    ny = onz;

    mx = omy;
    my = omz;
  } else if (axis == 1) {
    nx = onx;
    ny = onz;

    mx = omx;
    my = omz;
  } else if (axis == 2) {
    nx = onx;
    ny = ony;

    mx = omx;
    my = omy;
  }

  if( ifield->get_type_description(0)->get_name() == "LatVolField" ) {
    
    typename ImageField<TYPE>::mesh_type *omesh =
      scinew typename ImageField<TYPE>::mesh_type();

    omesh->set_min_i( mx );
    omesh->set_min_j( my );
    omesh->set_ni( nx );
    omesh->set_nj( ny );

    ImageField<TYPE> *ofield = scinew ImageField<TYPE>(omesh, Field::NODE);

    ofield_h = ofield;

  } else if( ifield->get_type_description(0)->get_name() == "StructHexVolField" ) {
    typename StructQuadSurfField<TYPE>::mesh_type *omesh =
      scinew typename StructQuadSurfField<TYPE>::mesh_type(nx,ny);

    StructQuadSurfField<TYPE> *ofield = scinew StructQuadSurfField<TYPE>(omesh, Field::NODE);

    ofield_h = ofield;

  } else if( ifield->get_type_description(0)->get_name() == "ImageField" ) {
    typename ScanlineField<TYPE>::mesh_type *omesh =
      scinew typename ScanlineField<TYPE>::mesh_type();

    omesh->set_min_i( mx );
    omesh->set_ni( nx );

    ScanlineField<TYPE> *ofield = scinew ScanlineField<TYPE>(omesh, Field::NODE);

    ofield_h = ofield;

  } else if( ifield->get_type_description(0)->get_name() == "StructQuadSurfField" ) {
    typename StructCurveField<TYPE>::mesh_type *omesh =
      scinew typename StructCurveField<TYPE>::mesh_type(nx);

    StructCurveField<TYPE> *ofield = scinew StructCurveField<TYPE>(omesh, Field::NODE);

    ofield_h = ofield;
  }

  return ofield_h;
}


template< class IFIELD, class OFIELD >
void
FusionSlicerWorkAlgoT<IFIELD, OFIELD>::execute(FieldHandle ifield_h,
					       FieldHandle ofield_h,
					       unsigned int index,
					       int axis)
{
  IFIELD *ifield = (IFIELD *) ifield_h.get_rep();
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  OFIELD *ofield = (OFIELD *) ofield_h.get_rep();
  typename OFIELD::mesh_handle_type omesh = ofield->get_typed_mesh();

  unsigned int omx, omy, omz;
  unsigned int onx, ony, onz;

  unsigned int mx, my, nx, ny;

  Array1<unsigned int> dim = imesh->get_dim();
  Array1<unsigned int> min = imesh->get_min();

  if( dim.size() == 3 ) {
    omx = min[0];
    omy = min[1];
    omz = min[2];

    onx = dim[0];
    ony = dim[1];
    onz = dim[2];
  } else if( dim.size() == 2 ) {
    omx = min[0];
    omy = min[1];
    omz = 0;

    onx = dim[0];
    ony = dim[1];
    onz = 1;
  }

  if (axis == 0) {
    nx = ony;
    ny = onz;

    mx = omy;
    my = omz;
  } else if (axis == 1) {
    nx = onx;
    ny = onz;

    mx = omx;
    my = omz;
  } else if (axis == 2) {
    nx = onx;
    ny = ony;

    mx = omx;
    my = omy;
  }

  typename IFIELD::mesh_type::Node::iterator inodeItr;
  typename OFIELD::mesh_type::Node::iterator onodeItr;

  imesh->begin( inodeItr );
  omesh->begin( onodeItr );

  Point p;
  typename IFIELD::value_type v;
 
  unsigned int i, j;
  unsigned int it, jt, kt;

  if (axis == 0) {                    // node.i_ = index
    for (j=0; j < ny; j++) {          // node.j_ = i
      for (i=0; i < nx; i++) {        // node.k_ = j
	
	// Since an iterator can not be added to, increment to the correct location.
	for (it=0; it<index; it++)
	  ++inodeItr;

	if( ifield->get_type_description(0)->get_name() == "StructHexVolField" ||
	    ifield->get_type_description(0)->get_name() == "StructQuadSurfField" ) {
	  imesh->get_center(p, *inodeItr);
	  omesh->set_point(*onodeItr, p);
	}
       
	ifield->value(v, *inodeItr);
	ofield->set_value(v, *onodeItr);
       
	// Since an iterator can not be added to, increment to the correct location.
	for (;it<onx; it++)
	  ++inodeItr;

	++onodeItr;       
      }
    }
  } else if(axis == 1) {              // node.i_ = i
    for (j=0; j < ny; j++) {          // node.j_ = index
                                      // node.k_ = j
      for (jt=0; jt<index; jt++)
	for (it=0; it<onx; it++)
	  ++inodeItr;

      for (i=0; i < nx; i++) {

	if( ifield->get_type_description(0)->get_name() == "StructHexVolField" ||
	    ifield->get_type_description(0)->get_name() == "StructQuadSurfField" ) {
	  imesh->get_center(p, *inodeItr);
	  omesh->set_point(*onodeItr, p);
	}

	ifield->value(v, *inodeItr);
	ofield->set_value(v, *onodeItr);

	++inodeItr;
	++onodeItr;

      }

      jt++;

      for (; jt<ony; jt++)
	for (it=0; it<onx; it++)
	  ++inodeItr;
    }

  } else if(axis == 2) {              // node.i_ = i
    for (kt=0; kt<index; kt++)        // node.j_ = j
      for (j=0; j<ony; j++)           // node.k_ = index
	for (i=0; i<onx; i++)
	  ++inodeItr;

    for (j=0; j < ny; j++) {
      for (i=0; i < nx; i++) {

	if( ifield->get_type_description(0)->get_name() == "StructHexVolField" ||
	    ifield->get_type_description(0)->get_name() == "StructQuadSurfField" ) {
	  imesh->get_center(p, *inodeItr);
	  omesh->set_point(*onodeItr, p);
	}

	ifield->value(v, *inodeItr);
	ofield->set_value(v, *onodeItr);

	++inodeItr;
	++onodeItr;
      }
    }
  }
}


} // end namespace SCIRun

#endif // FusionSlicer_h



