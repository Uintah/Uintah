/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


//!    File   : FieldSlicer.h
//!    Author : Michael Callahan &&
//!             Allen Sanderson
//!             SCI Institute
//!             University of Utah
//!    Date   : March 2006
//
//!    Copyright (C) 2006 SCI Group


#if !defined(ClipRasterFieldByIndices_h)
#define ClipRasterFieldByIndices_h

#include <Core/Containers/Handle.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/ScanlineMesh.h>
#include <Core/Datatypes/StructHexVolMesh.h>
#include <Core/Datatypes/StructQuadSurfMesh.h>
#include <Core/Datatypes/StructCurveMesh.h>


namespace SCIRun {

class ClipRasterFieldByIndicesAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle& src,
			      unsigned int i_start,
			      unsigned int j_start,
			      unsigned int k_start,
			      unsigned int i_stop,
			      unsigned int j_stop,
			      unsigned int k_stop,
			      unsigned int i_stride,
			      unsigned int j_stride,
			      unsigned int k_stride,
			      unsigned int i_wrap,
			      unsigned int j_wrap,
			      unsigned int k_wrap) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd,
					    bool geometry_irregular);
};


template <class FIELD>
class ClipRasterFieldByIndicesAlgoT : public ClipRasterFieldByIndicesAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle& src,
			      unsigned int i_start,
			      unsigned int j_start,
			      unsigned int k_start,
			      unsigned int i_stop,
			      unsigned int j_stop,
			      unsigned int k_stop,
			      unsigned int i_stride,
			      unsigned int j_stride,
			      unsigned int k_stride,
			      unsigned int i_wrap,
			      unsigned int j_wrap,
			      unsigned int k_wrap);
};


template <class FIELD>
FieldHandle
ClipRasterFieldByIndicesAlgoT<FIELD>::execute(FieldHandle& field_h,
				    unsigned int i_start,
				    unsigned int j_start,
				    unsigned int k_start,
				    unsigned int i_stop,
				    unsigned int j_stop,
				    unsigned int k_stop,
				    unsigned int i_stride,
				    unsigned int j_stride,
				    unsigned int k_stride,
				    unsigned int i_wrap,
				    unsigned int j_wrap,
				    unsigned int k_wrap)
{
  FIELD *ifield = dynamic_cast<FIELD *>(field_h.get_rep());
  typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  unsigned int idim_in, jdim_in, kdim_in;

  register unsigned int ic;

  register unsigned int i, j, k, inode, jnode, knode;

  vector<unsigned int> dim;

  imesh->get_dim( dim );

  unsigned int rank = dim.size();

  if( rank == 3 ) {
    idim_in = dim[0];
    jdim_in = dim[1];
    kdim_in = dim[2];
  } else if( rank == 2 ) {
    idim_in = dim[0];
    jdim_in = dim[1];
    kdim_in = 1;
  } else if( rank == 1 ) {
    idim_in = dim[0];
    jdim_in = 1;
    kdim_in = 1;
  }

  //! This happens when _wrapping.
  if( i_stop <= i_start ) i_stop += idim_in;
  if( j_stop <= j_start ) j_stop += jdim_in;
  if( k_stop <= k_start ) k_stop += kdim_in;

  //! Add one because we want the last node.
  unsigned int idim_out = (i_stop - i_start) / i_stride + (rank >= 1 ? 1 : 0);
  unsigned int jdim_out = (j_stop - j_start) / j_stride + (rank >= 2 ? 1 : 0);
  unsigned int kdim_out = (k_stop - k_start) / k_stride + (rank >= 3 ? 1 : 0);

  unsigned int i_stop_stride;
  unsigned int j_stop_stride;
  unsigned int k_stop_stride; 

  string field_name =
    ifield->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name();

  if( field_name.find( "StructHexVolField"   ) != string::npos ||
      field_name.find( "StructQuadSurfField" ) != string::npos ||
      field_name.find( "StructCurveField"    ) != string::npos ) {

    //! Account for the modulo of stride so that the last node will be
    //! included even if it "partial" cell when compared to the others.
    if( (i_stop - i_start) % i_stride ) idim_out += (rank >= 1 ? 1 : 0);
    if( (j_stop - j_start) % j_stride ) jdim_out += (rank >= 2 ? 1 : 0);
    if( (k_stop - k_start) % k_stride ) kdim_out += (rank >= 3 ? 1 : 0);

    i_stop_stride = i_stop + (rank >= 1 ? i_stride : 0); 
    j_stop_stride = j_stop + (rank >= 2 ? j_stride : 0); 
    k_stop_stride = k_stop + (rank >= 3 ? k_stride : 0); 

  } else {
    i_stop_stride = i_stop + (rank >= 1 ? 1 : 0);
    j_stop_stride = j_stop + (rank >= 2 ? 1 : 0);
    k_stop_stride = k_stop + (rank >= 3 ? 1 : 0); 
  }

  typename FIELD::mesh_handle_type omesh = scinew typename FIELD::mesh_type();

  omesh->copy_properties(imesh.get_rep());

  if( rank == 3 ) {
    dim[0] = idim_out;
    dim[1] = jdim_out;
    dim[2] = kdim_out;
  } else if( rank == 2 ) {
    dim[0] = idim_out;
    dim[1] = jdim_out;
  } else if( rank == 1 ) {
    dim[0] = idim_out;
  }

  omesh->set_dim( dim );

  //! Now after the mesh has been created, create the field.
  FIELD *ofield = scinew FIELD(omesh);

  ofield->copy_properties(ifield);

#ifdef SET_POINT_DEFINED
  Point pt;
#endif

  Point p, o;
  typename FIELD::value_type value;

  typename FIELD::mesh_type::Node::iterator inodeItr, jnodeItr, knodeItr;
  typename FIELD::mesh_type::Node::iterator onodeItr;

  
  typename FIELD::mesh_type::Cell::iterator icellItr, jcellItr, kcellItr;
  typename FIELD::mesh_type::Cell::iterator ocellItr;

  imesh->begin( inodeItr );
  omesh->begin( onodeItr );

  imesh->begin( icellItr );
  omesh->begin( ocellItr );

  string mesh_name = ifield->get_type_description(Field::MESH_TD_E)->get_name();

  //! For structured geometry we need to set the correct location.
  if(mesh_name.find("LatVolMesh"  ) != string::npos ||
     mesh_name.find("ImageMesh"   ) != string::npos  ||
     mesh_name.find("ScanlineMesh") != string::npos ) 
  {
    vector< unsigned int > min;
    imesh->get_min( min );
    omesh->set_min( min );

    //! Set the orginal transform.
    omesh->set_transform( imesh->get_transform() );

    imesh->begin( inodeItr );

    //! Get the orgin of mesh. */
    imesh->get_center(o, *inodeItr);    

    //! Set the iterator to the first point.
    for (k=0; k<k_start; k++) {
      for( ic=0; ic<jdim_in*idim_in; ic++ )
	++inodeItr;
    }
     
    for (j=0; j<j_start; j++) {
      for( ic=0; ic<idim_in; ic++ )
	++inodeItr;
    }
	
    for (i=0; i<i_start; i++)
      ++inodeItr;

    //! Get the point.
    imesh->get_center(p, *inodeItr);

    //! Put the new field into the correct location.
    Transform trans;

    trans.pre_translate( (Vector) (-o) );

    trans.pre_scale( Vector( i_stride, j_stride, k_stride ) );

    trans.pre_translate( (Vector) (o) );

    trans.pre_translate( (Vector) (p-o) );
      
    omesh->transform( trans );
  }

  //! Index based on the old mesh so that we are assured of getting the last
  //! node even if it forms a "partial" cell.
  for( k=k_start; k<k_stop_stride; k+=k_stride ) {

    //! Check for going past the stop.
    if( k > k_stop )
      k = k_stop;

    //! Check for overlap.
    if( k-k_stride <= k_start+kdim_in && k_start+kdim_in <= k )
      knode = k_start;
    else
      knode = k % kdim_in;

    //! A hack here so that an iterator can be used.
    //! Set this iterator to be at the correct kth index.
    imesh->begin( knodeItr );
    imesh->begin( kcellItr );

    for( ic=0; ic<knode*jdim_in*idim_in; ic++ )
      ++knodeItr;

    for( ic=0; ic<knode*(jdim_in-1)*(idim_in-1); ic++ )
      ++kcellItr;

    for( j=j_start; j<j_stop_stride; j+=j_stride ) {

      //! Check for going past the stop.
      if( j > j_stop )
	j = j_stop;

      //! Check for overlap.
      if( j-j_stride <= j_start+jdim_in && j_start+jdim_in <= j )
	jnode = j_start;
      else
	jnode = j % jdim_in;

      //! A hack here so that an iterator can be used.
      //! Set this iterator to be at the correct jth index.
      jnodeItr = knodeItr;
      jcellItr = kcellItr;

      for( ic=0; ic<jnode*idim_in; ic++ )
	++jnodeItr;

      for( ic=0; ic<jnode*(idim_in-1); ic++ )
	++jcellItr;

      for( i=i_start; i<i_stop_stride; i+=i_stride ) {

	//! Check for going past the stop.
	if( i > i_stop )
	  i = i_stop;

	//! Check for overlap.
	if( i-i_stride <= i_start+idim_in && i_start+idim_in <= i )
	  inode = i_start;
	else
	  inode = i % idim_in;

	//! A hack here so that an iterator can be used.
	//! Set this iterator to be at the correct ith index.
	inodeItr = jnodeItr;
	icellItr = jcellItr;

	for( ic=0; ic<inode; ic++ )
	  ++inodeItr;

	for( ic=0; ic<inode; ic++ )
	  ++icellItr;

#ifdef SET_POINT_DEFINED
	imesh->get_center(pt, *inodeItr);
	omesh->set_point(pt, *onodeItr);
#endif

	//	cout << knode << "  " << jnode << "  " << inode << endl;

	switch( ifield->basis_order() ) {
	case 1:
	  ifield->value(value, *inodeItr);
	  ofield->set_value(value, *onodeItr);
	  break;
	case 0:

	  if( i+i_stride<i_stop_stride &&
	      j+j_stride<j_stop_stride &&
	      k+k_stride<k_stop_stride ) {
	    ifield->value(value, *icellItr);
	    ofield->set_value(value, *ocellItr);

	    /*	    
	    cout << knode << "  " << jnode << "  " << inode << "        ";
	    cout << (*icellItr).k_ << "  " << (*icellItr).j_ << "  " << (*icellItr).i_ << "        ";
	    cout << (*ocellItr).k_ << "  " << (*ocellItr).j_ << "  " << (*ocellItr).i_ << endl;
	    */

	    ++ocellItr;
	  }
	  break;

	default:
	  break;
	}

	++onodeItr;
      }
    }
  }

  return ofield;
}

} //! end namespace SCIRun

#endif //! ClipRasterFieldByIndices_h
