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

//    File   : EditFusionField.h
//    Author : Michael Callahan
//    Date   : April 2002

#if !defined(EditFusionField_h)
#define EditFusionField_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

namespace Fusion {

using namespace SCIRun;

class EditFusionFieldAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle src,
			      unsigned int istart,
			      unsigned int jstart,
			      unsigned int kstart,
			      unsigned int iend,
			      unsigned int jend,
			      unsigned int kend,
			      unsigned int iskip,
			      unsigned int jskip,
			      unsigned int kskip) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftd);
};


template <class FIELD>
class EditFusionFieldAlgoT : public EditFusionFieldAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle src,
			      unsigned int istart,
			      unsigned int jstart,
			      unsigned int kstart,
			      unsigned int iend,
			      unsigned int jend,
			      unsigned int kend,
			      unsigned int iskip,
			      unsigned int jskip,
			      unsigned int kskip);
};


template <class FIELD>
FieldHandle
EditFusionFieldAlgoT<FIELD>::execute(FieldHandle field_h,
				      unsigned int istart,
				      unsigned int jstart,
				      unsigned int kstart,
				      unsigned int iend,
				      unsigned int jend,
				      unsigned int kend,
				      unsigned int iskip,
				      unsigned int jskip,
				      unsigned int kskip)
{
  FIELD *ifield = dynamic_cast<FIELD *>(field_h.get_rep());
  typename FIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  // Account for the repeated values in the theta and phi directions
  // by subtracting 1 in the j and k directions.
  const unsigned int jdim_in = imesh->get_nj() - 1;
  const unsigned int kdim_in = imesh->get_nk() - 1;

  // Add one because we want the last node.
  unsigned int idim_out = (iend - istart) / iskip + 1;
  unsigned int jdim_out = (jend - jstart) / jskip + 1;
  unsigned int kdim_out = (kend - kstart) / kskip + 1;

  // Account for the modulo of skipping nodes so that the last node will be
  // included even if it "partial" cell when compared to the others.
  if( (iend - istart) % iskip ) idim_out += 1;
  if( (jend - jstart) % jskip ) jdim_out += 1;
  if( (kend - kstart) % kskip ) kdim_out += 1;

  typename FIELD::mesh_handle_type omesh =
    scinew typename FIELD::mesh_type(idim_out, jdim_out, kdim_out);

  // Now after the mesh has been created, create the field.
  FIELD *ofield = scinew FIELD(omesh, Field::NODE);

  unsigned int i, j, k;
  unsigned int iend_skip = iend+iskip;
  unsigned int jend_skip = jend+jskip;
  unsigned int kend_skip = kend+kskip;

  Point pt;
  typename FIELD::value_type value;
  typename FIELD::mesh_type::Node::index_type old_node;
  typename FIELD::mesh_type::Node::index_type new_node;

  // Index based on the old mesh so that we are assured of getting the last
  // node even if it forms a "partial" cell.
  for( k=kstart, new_node.j_=0; k<kend_skip; k+=kskip, new_node.k_++ ) {

    // Check for going past the end.
    if( k > kend )
      k = kend;

    // Check for overlap.
    if( k-kskip <= kstart+kdim_in && kstart+kdim_in <= k )
      old_node.k_ = kstart;
    else
      old_node.k_ = k % kdim_in;

    for( j=jstart, new_node.j_=0; j<jend_skip; j+=jskip, new_node.j_++ ) {

      // Check for going past the end.
      if( j > jend )
	j = jend;

      // Check for overlap.
      if( j-jskip <= jstart+jdim_in && jstart+jdim_in <= j )
	old_node.j_ = jstart;
      else
	old_node.j_ = j % jdim_in;

      for( i=istart, new_node.i_=0; i<iend_skip; i+=iskip, new_node.i_++ ) {

	// Check for going past the end.
	if( i > iend )
	  i = iend;

	old_node.i_ = i;
 
	imesh->get_center(pt, old_node);
	omesh->set_point(new_node, pt);

	ifield->value(value, old_node);
	ofield->set_value(value, new_node);
      }
    }
  }

  return ofield;
}

} // end namespace SCIRun

#endif // EditFusionField_h
