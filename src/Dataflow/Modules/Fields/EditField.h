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

//    File   : EditField.h
//    Author : Michael Callahan
//    Date   : June 2001

#if !defined(EditField_h)
#define EditField_h

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Disclosure/DynamicLoader.h>
#include <Core/Datatypes/Field.h>

namespace SCIRun {

class EditFieldAlgoCount : public DynamicAlgoBase
{
public:
  virtual void execute(MeshHandle src, int &num_nodes, int &num_elems,
		       int &dimension) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *msrc);
};


template <class MESH>
class EditFieldAlgoCountT : public EditFieldAlgoCount
{
public:
  //! virtual interface. 
  virtual void execute(MeshHandle src, int &num_nodes, int &num_elems,
		       int &dimension);
};


template <class MESH>
void 
EditFieldAlgoCountT<MESH>::execute(MeshHandle mesh_h,
				   int &num_nodes, int &num_elems,
				   int &dimension)
{
  typedef typename MESH::Node::iterator node_iter_type;
  typedef typename MESH::Elem::iterator elem_iter_type;

  MESH *mesh = dynamic_cast<MESH *>(mesh_h.get_rep());

  int count = 0;
  node_iter_type ni; mesh->begin(ni);
  node_iter_type nie; mesh->end(nie);
  while (ni != nie)
  {
    count++;
    ++ni;
  }
  num_nodes = count;

  count = 0;
  elem_iter_type ei; mesh->begin(ei);
  elem_iter_type eie; mesh->end(eie);
  while (ei != eie)
  {
    count++;
    ++ei;
  }
  num_elems = count;

  dimension = 0;
  
  typename MESH::Edge::iterator eb, ee; mesh->begin(eb); mesh->end(ee);
  if (eb != ee);
  {
    dimension = 1;
  }
  typename MESH::Face::iterator fb, fe; mesh->begin(fb); mesh->end(fe);
  if (fb != fe)
  {
    dimension = 2;
  }
  typename MESH::Cell::iterator cb, ce; mesh->begin(cb); mesh->end(ce);
  if (cb != ce)
  {
    dimension = 3;
  }
}



class EditFieldAlgoCopy : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(FieldHandle fsrc_h,
			      Field::data_location fout_at,
			      bool transform_p, double scale,
			      double translate) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *fsrc,
				       const string &fdstname);
};


template <class FSRC, class FOUT>
class EditFieldAlgoCopyT : public EditFieldAlgoCopy
{
public:

  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle fsrc_h,
			      Field::data_location fout_at,
			      bool transform_p, double scale,
			      double translate);
};


template <class FSRC, class FOUT>
FieldHandle
EditFieldAlgoCopyT<FSRC, FOUT>::execute(FieldHandle fsrc_h,
					Field::data_location fout_at,
					bool transform_p, double scale,
					double translate)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(fsrc_h.get_rep());

  // Create the field with the new mesh and data location.
  FOUT *fout = scinew FOUT(fsrc->get_typed_mesh(), fout_at);

  // Copy the (possibly transformed) data to the new field.
  fout->resize_fdata();
  typename FSRC::fdata_type::iterator in = fsrc->fdata().begin();
  typename FOUT::fdata_type::iterator out = fout->fdata().begin();
  typename FSRC::fdata_type::iterator end = fsrc->fdata().end();
  if (fout_at == fsrc->data_at())
  {
    while (in != end)
    {
      if (transform_p)
      {
	// Linearly transform the data.
	*out = (typename FOUT::value_type)(*in * scale + translate);
      }
      else
      {
	*out = (typename FOUT::value_type)(*in);
      }
      ++in; ++out;
    }
  }

  return fout;
}


} // end namespace SCIRun

#endif // EditField_h
