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

//    File   : FieldSubSample.h
//    Author : Michael Callahan &&
//             Allen Sanderson
//             School of Computing
//             University of Utah
//    Date   : January 2003

#if !defined(FieldSubSample_h)
#define FieldSubSample_h

#include <Core/Containers/Handle.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/StructHexVolField.h>
#include <Core/Datatypes/StructQuadSurfField.h>
#include <Core/Datatypes/StructCurveField.h>


namespace SCIRun {

class FieldSubSampleAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src,
			      unsigned int istart,
			      unsigned int jstart,
			      unsigned int kstart,
			      unsigned int istop,
			      unsigned int jstop,
			      unsigned int kstop,
			      unsigned int iskip,
			      unsigned int jskip,
			      unsigned int kskip,
			      unsigned int iwrap,
			      unsigned int jwrap,
			      unsigned int kwrap) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd);
};


template <class FIELD>
class FieldSubSampleAlgoT : public FieldSubSampleAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src,
			      unsigned int istart,
			      unsigned int jstart,
			      unsigned int kstart,
			      unsigned int istop,
			      unsigned int jstop,
			      unsigned int kstop,
			      unsigned int iskip,
			      unsigned int jskip,
			      unsigned int kskip,
			      unsigned int iwrap,
			      unsigned int jwrap,
			      unsigned int kwrap);
};


template <class FIELD>
FieldHandle
FieldSubSampleAlgoT<FIELD>::execute(FieldHandle field_h,
				    unsigned int istart,
				    unsigned int jstart,
				    unsigned int kstart,
				    unsigned int istop,
				    unsigned int jstop,
				    unsigned int kstop,
				    unsigned int iskip,
				    unsigned int jskip,
				    unsigned int kskip,
				    unsigned int iwrap,
				    unsigned int jwrap,
				    unsigned int kwrap)
{
  FIELD *ifield = dynamic_cast<FIELD *>(field_h.get_rep());
  typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  unsigned int idim_in, jdim_in, kdim_in;

  register unsigned int ic;

  register unsigned int i, j, k, inode, jnode, knode;

  Array1<unsigned int> dim = imesh->get_dim();

  if( dim.size() == 3 ) {
    idim_in = dim[0];
    jdim_in = dim[1];
    kdim_in = dim[2];
  } else if( dim.size() == 2 ) {
    idim_in = dim[0];
    jdim_in = dim[1];
    kdim_in = 1;
  } else if( dim.size() == 1 ) {
    idim_in = dim[0];
    jdim_in = 1;
    kdim_in = 1;
  }

  // This happens when wrapping.
  if( istop <= istart ) istop += idim_in;
  if( jstop <= jstart ) jstop += jdim_in;
  if( kstop <= kstart ) kstop += kdim_in;

  // Add one because we want the last node.
  unsigned int idim_out = (istop - istart) / iskip + 1;
  unsigned int jdim_out = (jstop - jstart) / jskip + 1;
  unsigned int kdim_out = (kstop - kstart) / kskip + 1;

  unsigned int istop_skip;
  unsigned int jstop_skip;
  unsigned int kstop_skip; 

  if( field_h->get_type_description(0)->get_name() == "StructHexVolField" ||
      field_h->get_type_description(0)->get_name() == "StructQuadSurfField" ||
      field_h->get_type_description(0)->get_name() == "StructCurveField" ) {

    // Account for the modulo of skipping nodes so that the last node will be
    // included even if it "partial" cell when compared to the others.
    if( (istop - istart) % iskip ) idim_out += 1;
    if( (jstop - jstart) % jskip ) jdim_out += 1;
    if( (kstop - kstart) % kskip ) kdim_out += 1;

    istop_skip = istop + iskip;
    jstop_skip = jstop + jskip;
    kstop_skip = kstop + kskip; 
  } else {
    istop_skip = istop + 1;
    jstop_skip = jstop + 1;
    kstop_skip = kstop + 1; 
  }

  typename FIELD::mesh_handle_type omesh =
    scinew typename FIELD::mesh_type();

  if( dim.size() == 3 ) {
    dim[0] = idim_out;
    dim[1] = jdim_out;
    dim[2] = kdim_out;
  } else if( dim.size() == 2 ) {
    dim[0] = idim_out;
    dim[1] = jdim_out;
  } else if( dim.size() == 1 ) {
    dim[0] = idim_out;
  }

  omesh->set_dim( dim );
   
  // Now after the mesh has been created, create the field.
  FIELD *ofield = scinew FIELD(omesh, Field::NODE);


  Point p, o, pt;
  typename FIELD::value_type value;

  typename FIELD::mesh_type::Node::iterator inodeItr, jnodeItr, knodeItr;
  typename FIELD::mesh_type::Node::iterator onodeItr;

  imesh->begin( inodeItr );
  omesh->begin( onodeItr );

  // For structured geometery we need to set the correct location.
  if( ifield->get_type_description(0)->get_name() == "LatVolField" ||
      ifield->get_type_description(0)->get_name() == "ImageField" ||
      ifield->get_type_description(0)->get_name() == "ScanlineField" ) {

    omesh->set_min( imesh->get_min() );

    // Set the orginal transform.
    omesh->set_transform( imesh->get_transform() );

    imesh->begin( inodeItr );

    // Get the orgin of mesh. */
    imesh->get_center(o, *inodeItr);    

    // Set the iterator to the first point.
    for (k=0; k<kstart; k++) {
      for( ic=0; ic<jdim_in*idim_in; ic++ )
	++inodeItr;
    }
     
    for (j=0; j<jstart; j++) {
      for( ic=0; ic<idim_in; ic++ )
	++inodeItr;
    }
	
    for (i=0; i<istart; i++)
      ++inodeItr;

    // Get the point.
    imesh->get_center(p, *inodeItr);

    // Put the new field into the correct location.
    Transform trans;

    trans.pre_translate( (Vector) (-o) );

    trans.pre_scale( Vector( iskip, jskip, kskip ) );

    trans.pre_translate( (Vector) (o) );

    trans.pre_translate( (Vector) (p-o) );
      
    omesh->transform( trans );
  }

  // Index based on the old mesh so that we are assured of getting the last
  // node even if it forms a "partial" cell.
  for( k=kstart; k<kstop_skip; k+=kskip ) {

    // Check for going past the stop.
    if( k > kstop )
      k = kstop;

    // Check for overlap.
    if( k-kskip <= kstart+kdim_in && kstart+kdim_in <= k )
      knode = kstart;
    else
      knode = k % kdim_in;

    // A hack here so that an iterator can be used.
    // Set this iterator to be at the correct kth index.
    imesh->begin( knodeItr );

    for( ic=0; ic<knode*jdim_in*idim_in; ic++ )
      ++knodeItr;

    for( j=jstart; j<jstop_skip; j+=jskip ) {

      // Check for going past the stop.
      if( j > jstop )
	j = jstop;

      // Check for overlap.
      if( j-jskip <= jstart+jdim_in && jstart+jdim_in <= j )
	jnode = jstart;
      else
	jnode = j % jdim_in;

      // A hack here so that an iterator can be used.
      // Set this iterator to be at the correct jth index.
      jnodeItr = knodeItr;

      for( ic=0; ic<jnode*idim_in; ic++ )
	++jnodeItr;

      for( i=istart; i<istop_skip; i+=iskip ) {

	// Check for going past the stop.
	if( i > istop )
	  i = istop;

	// Check for overlap.
	if( i-iskip <= istart+idim_in && istart+idim_in <= i )
	  inode = istart;
	else
	  inode = i % idim_in;

	// A hack here so that an iterator can be used.
	// Set this iterator to be at the correct ith index.
	inodeItr = jnodeItr;

	for( ic=0; ic<inode; ic++ )
	  ++inodeItr;

#ifdef SET_POINT_DEFINED
	imesh->get_center(pt, *inodeItr);
	omesh->set_point(*onodeItr, pt);
#endif
	ifield->value(value, *inodeItr);
	ofield->set_value(value, *onodeItr);

	++onodeItr;
      }
    }
  }

  return ofield;
}

} // end namespace SCIRun

#endif // FieldSubSample_h
