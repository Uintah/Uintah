/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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
  virtual FieldHandle execute(FieldHandle& src,
			      unsigned int istart,
			      unsigned int jstart,
			      unsigned int kstart,
			      unsigned int istop,
			      unsigned int jstop,
			      unsigned int kstop,
			      unsigned int istride,
			      unsigned int jstride,
			      unsigned int kstride,
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
  virtual FieldHandle execute(FieldHandle& src,
			      unsigned int istart,
			      unsigned int jstart,
			      unsigned int kstart,
			      unsigned int istop,
			      unsigned int jstop,
			      unsigned int kstop,
			      unsigned int istride,
			      unsigned int jstride,
			      unsigned int kstride,
			      unsigned int iwrap,
			      unsigned int jwrap,
			      unsigned int kwrap);
};


template <class FIELD>
FieldHandle
FieldSubSampleAlgoT<FIELD>::execute(FieldHandle& field_h,
				    unsigned int istart,
				    unsigned int jstart,
				    unsigned int kstart,
				    unsigned int istop,
				    unsigned int jstop,
				    unsigned int kstop,
				    unsigned int istride,
				    unsigned int jstride,
				    unsigned int kstride,
				    unsigned int iwrap,
				    unsigned int jwrap,
				    unsigned int kwrap)
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

  // This happens when wrapping.
  if( istop <= istart ) istop += idim_in;
  if( jstop <= jstart ) jstop += jdim_in;
  if( kstop <= kstart ) kstop += kdim_in;

  // Add one because we want the last node.
  unsigned int idim_out = (istop - istart) / istride + (rank >= 1 ? 1 : 0);
  unsigned int jdim_out = (jstop - jstart) / jstride + (rank >= 2 ? 1 : 0);
  unsigned int kdim_out = (kstop - kstart) / kstride + (rank >= 3 ? 1 : 0);

  unsigned int istop_stride;
  unsigned int jstop_stride;
  unsigned int kstop_stride; 

  if( field_h->get_type_description(0)->get_name() == "StructHexVolField" ||
      field_h->get_type_description(0)->get_name() == "StructQuadSurfField" ||
      field_h->get_type_description(0)->get_name() == "StructCurveField" ) {

    // Account for the modulo of stride so that the last node will be
    // included even if it "partial" cell when compared to the others.
    if( (istop - istart) % istride ) idim_out += (rank >= 1 ? 1 : 0);
    if( (jstop - jstart) % jstride ) jdim_out += (rank >= 2 ? 1 : 0);
    if( (kstop - kstart) % kstride ) kdim_out += (rank >= 3 ? 1 : 0);

    istop_stride = istop + (rank >= 1 ? istride : 0); 
    jstop_stride = jstop + (rank >= 2 ? jstride : 0); 
    kstop_stride = kstop + (rank >= 3 ? kstride : 0); 

  } else {
    istop_stride = istop + (rank >= 1 ? 1 : 0);
    jstop_stride = jstop + (rank >= 2 ? 1 : 0);
    kstop_stride = kstop + (rank >= 3 ? 1 : 0); 
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

  // Now after the mesh has been created, create the field.
  FIELD *ofield = scinew FIELD(omesh, ifield->basis_order());

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

  // For structured geometery we need to set the correct location.
  if( ifield->get_type_description(0)->get_name() == "LatVolField" ||
      ifield->get_type_description(0)->get_name() == "ImageField" ||
      ifield->get_type_description(0)->get_name() == "ScanlineField" ) {

    vector< unsigned int > min;
    imesh->get_min( min );
    omesh->set_min( min );

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

    trans.pre_scale( Vector( istride, jstride, kstride ) );

    trans.pre_translate( (Vector) (o) );

    trans.pre_translate( (Vector) (p-o) );
      
    omesh->transform( trans );
  }

  // Index based on the old mesh so that we are assured of getting the last
  // node even if it forms a "partial" cell.
  for( k=kstart; k<kstop_stride; k+=kstride ) {

    // Check for going past the stop.
    if( k > kstop )
      k = kstop;

    // Check for overlap.
    if( k-kstride <= kstart+kdim_in && kstart+kdim_in <= k )
      knode = kstart;
    else
      knode = k % kdim_in;

    // A hack here so that an iterator can be used.
    // Set this iterator to be at the correct kth index.
    imesh->begin( knodeItr );
    imesh->begin( kcellItr );

    for( ic=0; ic<knode*jdim_in*idim_in; ic++ )
      ++knodeItr;

    for( ic=0; ic<knode*(jdim_in-1)*(idim_in-1); ic++ )
      ++kcellItr;

    for( j=jstart; j<jstop_stride; j+=jstride ) {

      // Check for going past the stop.
      if( j > jstop )
	j = jstop;

      // Check for overlap.
      if( j-jstride <= jstart+jdim_in && jstart+jdim_in <= j )
	jnode = jstart;
      else
	jnode = j % jdim_in;

      // A hack here so that an iterator can be used.
      // Set this iterator to be at the correct jth index.
      jnodeItr = knodeItr;
      jcellItr = kcellItr;

      for( ic=0; ic<jnode*idim_in; ic++ )
	++jnodeItr;

      for( ic=0; ic<jnode*(idim_in-1); ic++ )
	++jcellItr;

      for( i=istart; i<istop_stride; i+=istride ) {

	// Check for going past the stop.
	if( i > istop )
	  i = istop;

	// Check for overlap.
	if( i-istride <= istart+idim_in && istart+idim_in <= i )
	  inode = istart;
	else
	  inode = i % idim_in;

	// A hack here so that an iterator can be used.
	// Set this iterator to be at the correct ith index.
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

	  if( i+istride<istop_stride &&
	      j+jstride<jstop_stride &&
	      k+kstride<kstop_stride ) {
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

} // end namespace SCIRun

#endif // FieldSubSample_h
