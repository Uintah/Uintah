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

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>

namespace SCIRun {

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
  static CompileInfo *get_compile_info(const TypeDescription *fsrc);
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

  const unsigned int ini = imesh->get_nx();
  const unsigned int inj = imesh->get_ny();
  const unsigned int ink = imesh->get_nz();

  const unsigned int ni = (iend - istart) / iskip; 
  const unsigned int nj = (jend - jstart) / jskip; 
  const unsigned int nk = (kend - kstart) / kskip; 

  typename FIELD::mesh_handle_type omesh =
    scinew typename FIELD::mesh_type(ni, nj, nk);

  // Now after the mesh has been created, create the field.
  FIELD *ofield = scinew FIELD(omesh, Field::NODE);

  for (unsigned int i=0; i < ni; i++)
  {
    for (unsigned int j=0; j < nj; j++)
    {
      for (unsigned int k=0; k < nk; k++)
      {
	typename FIELD::mesh_type::Node::index_type
	  old_node((i*iskip+istart)%ini,
		   (j*jskip+jstart)%inj,
		   (k*kskip+kstart)%ink);
	typename FIELD::mesh_type::Node::index_type new_node(i, j, k);
	  
	Point p;
	imesh->get_center(p, old_node);
	omesh->set_point(new_node, p);
	typename FIELD::value_type value;
	ifield->value(value, old_node);
	ofield->set_value(value, new_node);
      }
    }
  }

  return ofield;
}


} // end namespace SCIRun

#endif // EditFusionField_h
