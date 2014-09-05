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

//    File   : FieldSlicer.h
//    Author : Michael Callahan &&
//             Allen Sanderson
//             School of Computing
//             University of Utah
//    Date   : December 2002

#if !defined(FieldSlicer_h)
#define FieldSlicer_h

#include <Core/Containers/Handle.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/StructHexVolField.h>
#include <Core/Datatypes/StructQuadSurfField.h>
#include <Core/Datatypes/StructCurveField.h>
#include <Core/Math/Trig.h>


namespace SCIRun {

class FieldSlicerAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src,
			      int axis) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    const TypeDescription *ttd);
};

template< class FIELD, class TYPE >
class FieldSlicerAlgoT : public FieldSlicerAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src,
			      int axis);
};


class FieldSlicerWorkAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle src,
		       FieldHandle dst,
		       unsigned int index,
		       int axis) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *iftd,
					    const TypeDescription *oftd);
};

template< class IFIELD, class OFIELD >
class FieldSlicerWorkAlgoT : public FieldSlicerWorkAlgo
{
public:
  //! virtual interface. 
  virtual void execute(FieldHandle src,
		       FieldHandle dst,
		       unsigned int index,
		       int axis);
};



template< class FIELD, class TYPE >
FieldHandle
FieldSlicerAlgoT<FIELD, TYPE>::execute(FieldHandle ifield_h,
					int axis)
{
  FieldHandle ofield_h;

  FIELD *ifield = (FIELD *) ifield_h.get_rep();
  typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  unsigned int old_min_i, old_min_j, old_min_k;
  unsigned int old_i, old_j, old_k;

  unsigned int new_min_i, new_min_j, new_i, new_j;

  vector<unsigned int> dim;
  imesh->get_dim( dim );

  vector<unsigned int> min;
  imesh->get_min( min );

  if( dim.size() == 3 ) {
    old_min_i = min[0];           old_i = dim[0];
    old_min_j = min[1];           old_j = dim[1];
    old_min_k = min[2];           old_k = dim[2];

  } else if( dim.size() == 2 ) {
    old_min_i = min[0];           old_i = dim[0];
    old_min_j = min[1];           old_j = dim[1];
    old_min_j = 0;                old_k = 1;
  } else if( dim.size() == 1 ) {
    old_min_i = min[0];           old_i = dim[0];
    old_min_j = 0;                old_j = 1;
    old_min_j = 0;                old_k = 1;
  }

  if (axis == 0) {
    new_i = old_j;                new_min_i = old_min_j;
    new_j = old_k;                new_min_j = old_min_k;
  } else if (axis == 1) {
    new_i = old_i;                new_min_i = old_min_i;
    new_j = old_k;                new_min_j = old_min_k;
  } else if (axis == 2) {
    new_i = old_i;                new_min_i = old_min_i;
    new_j = old_j;                new_min_j = old_min_j;
  }

  // Build the correct output field given the input field and and type.

  // 3D LatVol to 2D Image
  if( ifield->get_type_description(0)->get_name() == "LatVolField" ) {
    
    typename ImageField<TYPE>::mesh_type *omesh =
      scinew typename ImageField<TYPE>::mesh_type();

    omesh->set_min_i( new_min_i );
    omesh->set_min_j( new_min_j );
    omesh->set_ni( new_i );
    omesh->set_nj( new_j );

    ImageField<TYPE> *ofield =
      scinew ImageField<TYPE>(omesh, ifield->data_at());

    ofield_h = ofield;

    // 3D StructHexVol to 2D StructQuadSurf
  } else if( ifield->get_type_description(0)->get_name() == "StructHexVolField" ) {
    typename StructQuadSurfField<TYPE>::mesh_type *omesh =
      scinew typename StructQuadSurfField<TYPE>::mesh_type(new_i,new_j);

    StructQuadSurfField<TYPE> *ofield =
      scinew StructQuadSurfField<TYPE>(omesh, ifield->data_at());

    ofield_h = ofield;

    // 2D Image to 1D Scanline
  } else if( ifield->get_type_description(0)->get_name() == "ImageField" ) {
    typename ScanlineField<TYPE>::mesh_type *omesh =
      scinew typename ScanlineField<TYPE>::mesh_type();

    omesh->set_min_i( new_min_i );
    omesh->set_ni( new_i );

    ScanlineField<TYPE> *ofield = 
      scinew ScanlineField<TYPE>(omesh, ifield->data_at());

    ofield_h = ofield;

    // 2D StructQuadSurf to 1D StructCurve
  } else if( ifield->get_type_description(0)->get_name() == "StructQuadSurfField" ) {
    typename StructCurveField<TYPE>::mesh_type *omesh =
      scinew typename StructCurveField<TYPE>::mesh_type(new_i);

    StructCurveField<TYPE> *ofield =
      scinew StructCurveField<TYPE>(omesh, ifield->data_at());

    ofield_h = ofield;

    // 1D Scanline to 0D Scanline
  } else if( ifield->get_type_description(0)->get_name() == "ScanlineField" ) {
    typename ScanlineField<TYPE>::mesh_type *omesh =
      scinew typename ScanlineField<TYPE>::mesh_type();

    omesh->set_min_i( new_min_i );
    omesh->set_ni( new_i );

    ScanlineField<TYPE> *ofield =
      scinew ScanlineField<TYPE>(omesh, ifield->data_at());

    ofield_h = ofield;

    // 1D StructCurve to 0D StructCurve
  } else if( ifield->get_type_description(0)->get_name() == "StructCurveField" ) {
    typename StructCurveField<TYPE>::mesh_type *omesh =
      scinew typename StructCurveField<TYPE>::mesh_type(new_i);

    StructCurveField<TYPE> *ofield =
      scinew StructCurveField<TYPE>(omesh, ifield->data_at());

    ofield_h = ofield;
  }

  return ofield_h;
}


template< class IFIELD, class OFIELD >
void
FieldSlicerWorkAlgoT<IFIELD, OFIELD>::execute(FieldHandle ifield_h,
					      FieldHandle ofield_h,
					      unsigned int index,
					      int axis)
{
  IFIELD *ifield = (IFIELD *) ifield_h.get_rep();
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  OFIELD *ofield = (OFIELD *) ofield_h.get_rep();
  typename OFIELD::mesh_handle_type omesh = ofield->get_typed_mesh();

  unsigned int old_i, old_j, old_k;
  unsigned int new_i, new_j;

  vector<unsigned int> dim;
  imesh->get_dim( dim );

  if( dim.size() == 3 ) {
    old_i = dim[0];
    old_j = dim[1];
    old_k = dim[2];
  } else if( dim.size() == 2 ) {
    old_i = dim[0];
    old_j = dim[1];
    old_k = 1;        // By setting this it is possible to slice from 2D to 1D easily.
  } else if( dim.size() == 1 ) {
    old_i = dim[0];
    old_j = 1;
    old_k = 1;        // By setting this it is possible to slice from 1D to 0D easily.
  }

  if (axis == 0) {
    new_i = old_j;
    new_j = old_k;

  } else if (axis == 1) {
    new_i = old_i;
    new_j = old_k;

  } else if (axis == 2) {
    new_i = old_i;
    new_j = old_j;
  }

  typename IFIELD::mesh_type::Node::iterator inodeItr;
  typename OFIELD::mesh_type::Node::iterator onodeItr;

  imesh->begin( inodeItr );
  omesh->begin( onodeItr );

  Point p;
  typename IFIELD::value_type v;
 
  unsigned int i, j;
  unsigned int it, jt, kt;

#ifndef SET_POINT_DEFINED
  // For structured geometery we need to set the correct plane.
  if( ifield->get_type_description(0)->get_name() == "LatVolField" ||
      ifield->get_type_description(0)->get_name() == "ImageField" ||
      ifield->get_type_description(0)->get_name() == "ScanlineField" )
  {
    Transform trans = imesh->get_transform();
    double offset = 0.0;
    if (axis == 0)
    {
      trans.post_permute(2, 3, 1);
      offset = index / (double)old_i;
    }
    else if (axis == 1)
    {
      trans.post_permute(1, 3, 2);
      offset = index / (double)old_j;
    }
    else
    {
      offset = index / (double)old_k;
    }
    trans.post_translate(Vector(0.0, 0.0, index));

    omesh->transform( trans );
  }
#endif

  imesh->begin( inodeItr );

  // Slicing along the i axis. In order to get the slice in this direction only one
  // location can be obtained at a time. So the iterator must increment to the correct
  // column (i) index, the location samples, and then incremented to the end of the row.
  if (axis == 0) {                  // node.i_ = index
    for (j=0; j<new_j; j++) {       // node.j_ = i
      for (i=0; i<new_i; i++) {     // node.k_ = j
	
	// Set the iterator to the correct column (i) .
	for (it=0; it<index; it++)
	  ++inodeItr;

	// Get the point and value at this location
#ifdef SET_POINT_DEFINED
	imesh->get_center(p, *inodeItr);
	omesh->set_point(p, *onodeItr);
#endif       
	ifield->value(v, *inodeItr);
	ofield->set_value(v, *onodeItr);

	++onodeItr;

	// Move to the end of the row.
	for (;it<old_i; it++)
	  ++inodeItr;
      }
    }
    // Slicing along the j axis. In order to get the slice in this direction a complete
    // row can be obtained at a time. So the iterator must increment to the correct
    // row (j) index, the location samples, and then incremented to the end of the slice.
  } else if(axis == 1) {              // node.i_ = i
    for (j=0; j < new_j; j++) {       // node.j_ = index
                                      // node.k_ = j
      // Set the iterator to the correct row (j).
      for (jt=0; jt<index; jt++)
	for (it=0; it<old_i; it++) 
	  ++inodeItr;

      // Get all of the points and values along this row.
      for (i=0; i<new_i; i++) {

	// Get the point and value at this location
#ifdef SET_POINT_DEFINED
	imesh->get_center(p, *inodeItr);
	omesh->set_point(p, *onodeItr);
#endif
	ifield->value(v, *inodeItr);
	ofield->set_value(v, *onodeItr);

	++inodeItr;
	++onodeItr;
      }

      // Just got one full row line so increment.
      jt++;

      // Move to the end of all the rows..
      for (; jt<old_j; jt++)
	for (it=0; it<old_i; it++)
	  ++inodeItr;
    }

    // Slicing along the k axis.
  } else if(axis == 2) {
    // Set the iterator to the correct k slice.
    for (kt=0; kt<index; kt++)        // node.i_ = i
      for (j=0; j<old_j; j++)         // node.j_ = j
	for (i=0; i<old_i; i++)       // node.k_ = index
	  ++inodeItr;

    // Get all of the points and values along this slice.
    for (j=0; j < new_j; j++) {
      for (i=0; i < new_i; i++) {

	// Get the point and value at this location
#ifdef SET_POINT_DEFINED
	imesh->get_center(p, *inodeItr);
	omesh->set_point(p, *onodeItr);
#endif
	ifield->value(v, *inodeItr);
	ofield->set_value(v, *onodeItr);

	++inodeItr;
	++onodeItr;
      }
    }
  }
}

} // end namespace SCIRun

#endif // FieldSlicer_h
