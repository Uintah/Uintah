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

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
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

  mesh->synchronize(Mesh::ALL_ELEMENTS_E);

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



class EditFieldAlgoCreate : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(FieldHandle fsrc_h,
			      Field::data_location fout_at,
			      bool &same_value_type_p) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *fsrc,
				       const string &fdstname);
};


template <class FSRC, class FOUT>
class EditFieldAlgoCreateT : public EditFieldAlgoCreate
{
public:

  virtual FieldHandle execute(FieldHandle fsrc_h,
			      Field::data_location fout_at,
			      bool &same_value_type_p);
};


template <class FSRC, class FOUT>
FieldHandle
EditFieldAlgoCreateT<FSRC, FOUT>::execute(FieldHandle fsrc_h,
					  Field::data_location at,
					  bool &same_value_type_p)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(fsrc_h.get_rep());

  // Create the field with the new mesh and data location.
  FOUT *fout = scinew FOUT(fsrc->get_typed_mesh(), at);

  // Copy the (possibly transformed) data to the new field.
  fout->resize_fdata();

  same_value_type_p =
    (get_type_description((typename FSRC::value_type *)0)->get_name() ==
     get_type_description((typename FOUT::value_type *)0)->get_name());

  return fout;
}




class EditFieldAlgoCopy : public DynamicAlgoBase
{
public:

  virtual void execute(FieldHandle fsrc_h, FieldHandle fout_h,
		       double scale, double translate) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfo *get_compile_info(const TypeDescription *fsrc,
				       const TypeDescription *fdst,
				       bool transform_p);
};


template <class FSRC, class FOUT>
class EditFieldAlgoCopyT : public EditFieldAlgoCopy
{
public:

  //! virtual interface. 
  virtual void execute(FieldHandle fsrc_h, FieldHandle fout_h,
		       double scale, double translate);
};


template <class FSRC, class FOUT>
void
EditFieldAlgoCopyT<FSRC, FOUT>::execute(FieldHandle fsrc_h,
					FieldHandle fout_h,
					double scale,
					double translate)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(fsrc_h.get_rep());
  FOUT *fout = dynamic_cast<FOUT *>(fout_h.get_rep());

  typename FSRC::fdata_type::iterator in = fsrc->fdata().begin();
  typename FOUT::fdata_type::iterator out = fout->fdata().begin();
  typename FSRC::fdata_type::iterator end = fsrc->fdata().end();
  if (fout->data_at() == fsrc->data_at())
  {
    while (in != end)
    {
      *out = (typename FOUT::value_type)(*in);
      ++in; ++out;
    }
  }
}



template <class FSRC, class FOUT>
class EditFieldAlgoCopyTT : public EditFieldAlgoCopy
{
public:

  //! virtual interface. 
  virtual void execute(FieldHandle fsrc_h, FieldHandle fout_h,
		       double scale, double translate);
};


template <class FSRC, class FOUT>
void
EditFieldAlgoCopyTT<FSRC, FOUT>::execute(FieldHandle fsrc_h,
					 FieldHandle fout_h,
					 double scale,
					 double translate)
{
  FSRC *fsrc = dynamic_cast<FSRC *>(fsrc_h.get_rep());
  FOUT *fout = dynamic_cast<FOUT *>(fout_h.get_rep());

  typename FSRC::fdata_type::iterator in = fsrc->fdata().begin();
  typename FOUT::fdata_type::iterator out = fout->fdata().begin();
  typename FSRC::fdata_type::iterator end = fsrc->fdata().end();
  if (fout->data_at() == fsrc->data_at())
  {
    while (in != end)
    {
      *out = (typename FOUT::value_type)(*in * scale + translate);
      ++in; ++out;
    }
  }
}



} // end namespace SCIRun

#endif // EditField_h
