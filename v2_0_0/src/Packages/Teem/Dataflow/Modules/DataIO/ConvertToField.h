//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : ConvertToField.h
//    Author : Martin Cole
//    Date   : Tue Jan 21 09:36:39 2003

#if !defined(ConvertToField_h)
#define ConvertToField_h

#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Teem/Core/Datatypes/NrrdData.h>
#include <Core/Geometry/BBox.h>
#include <vector>
#include <iostream>

namespace SCIRun {
using namespace SCITeem;
using namespace std;

//! ConvertToFieldBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! ConvertToFieldBase from the DynamicAlgoBase they will have a pointer to.
class ConvertToFieldBase : public DynamicAlgoBase
{
public:
  virtual bool convert_to_field(SCIRun::FieldHandle, NrrdDataHandle, 
				SCIRun::FieldHandle &) = 0;
  virtual ~ConvertToFieldBase();

  static const string& get_h_file_path();
  static string dyn_file_name(const TypeDescription *td) {
    // add no extension.
    return template_class_name() + "." + td->get_filename() + ".";
  }

  static const string base_class_name() {
    static string name("ConvertToFieldBase");
    return name;
  }

  static const string template_class_name() {
    static string name("ConvertToField");
    return name;
  }

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *td);
};

template <class Fld>
class ConvertToField : public ConvertToFieldBase
{
public:
  //! virtual interface.
  virtual bool convert_to_field(SCIRun::FieldHandle, NrrdDataHandle in, 
				SCIRun::FieldHandle &);
};


void get_val_and_eigens_and_inc_nrrdptr(Tensor &v, void *&ptr);

template <class Val>
void get_val_and_inc_nrrdptr(Val &v, void *&ptr, unsigned);

template <>
void get_val_and_inc_nrrdptr<Vector>(Vector &v, void *&ptr, unsigned);

template <>
void get_val_and_inc_nrrdptr<Tensor>(Tensor &v, void *&ptr, unsigned);

template <class Val>
void get_val_and_inc_nrrdptr(Val &v, void *&ptr, unsigned) 
{
  Val *&p = (Val*&)ptr;
  v = *p;
  ++p;
}

template <class Fld, class Iter>
void 
fill_eigen_data(Fld *fld, Nrrd *inrrd, Iter &iter, Iter &end) 
{
  void *p = inrrd->data;

  while (iter != end) {
    Tensor tmp;
    get_val_and_eigens_and_inc_nrrdptr(tmp, p);
    fld->set_value(tmp, *iter);
    ++iter;
  }
}
template <class Fld, class Iter>
void 
fill_data(Fld *fld, Nrrd *inrrd, Iter &iter, Iter &end) 
{
  typedef typename Fld::value_type val_t;
  void *p = inrrd->data;

  while (iter != end) {
    val_t tmp;
    get_val_and_inc_nrrdptr(tmp, p, inrrd->type);
    fld->set_value(tmp, *iter);
    ++iter;
  }
}

template <class Fld>
bool
ConvertToField<Fld>::convert_to_field(SCIRun::FieldHandle  fld, 
				      NrrdDataHandle       in,
				      SCIRun::FieldHandle &out)
{
  Nrrd *inrrd = in->nrrd;
  vector<unsigned int> dims;
  // The input fld in not neccessarily the exact type Fld, 
  // it will have the exact same mesh type however.
  typedef typename Fld::mesh_type Msh;
  Msh *mesh = dynamic_cast<Msh*>(fld->mesh().get_rep());
  ASSERT(mesh != 0);
  int off = 0;
  bool uns = false;
  if (! mesh->get_dim(dims)) {
    uns = true;
    // Unstructured fields fall into this category, for them we create nrrds
    // of dimension 1 (2 with the tuple axis).
    switch (fld->data_at()) {
    case Field::NODE :
      {
	typename Fld::mesh_type::Node::size_type sz;
	mesh->size(sz);
	dims.push_back(sz);
      }
    break;
    case Field::EDGE :
      {
	typename Fld::mesh_type::Edge::size_type sz;
	mesh->size(sz);
	dims.push_back(sz);
      }
    break;
    case Field::FACE :
      {
	typename Fld::mesh_type::Face::size_type sz;
	mesh->size(sz);
	dims.push_back(sz);
      }
    break;
    case Field::CELL :
      {
	typename Fld::mesh_type::Cell::size_type sz;
	mesh->size(sz);
	dims.push_back(sz);
      }
    break;
    default:
      return false;
    }
  }
  if ((!uns) && fld->data_at() == Field::CELL) {
    off = 1;
  }
  // All sci nrrds should have a tuple axis, we assume it.
  // It is axis 0.  Make sure sizes along each dim still match.

  if (inrrd->dim != (int)dims.size() + 1) {
    return false;
  }
  switch (inrrd->dim -1) {
  case 1:
    {
      // make sure size of dimensions match up
      unsigned int nx = inrrd->axis[1].size + off;
      if (nx != dims[0]) { return false; }
    }
    break;
  case 2:
    {
      unsigned int nx = inrrd->axis[1].size + off;
      unsigned int ny = inrrd->axis[2].size + off;
      if ((nx != dims[0]) || (ny != dims[1])) {
	return false;
      }
    }
    break;
  case 3:
    {
      unsigned int nx = inrrd->axis[1].size + off;
      unsigned int ny = inrrd->axis[2].size + off;
      unsigned int nz = inrrd->axis[3].size + off;
      if ((nx != dims[0]) || (ny != dims[1]) || (nz != dims[2])) {
	return false;
      }
    }
    break;
  default:   // anything else is invalid.
    return false;
  }
  // Things match up, create the new output field.
  out = new Fld(typename Fld::mesh_handle_type(mesh), fld->data_at());
  
  // Copy all of the non-transient properties from the original field.
  *((PropertyManager *)(out.get_rep()))=*((PropertyManager *)(fld.get_rep()));

  // Copy the data into the field.
  switch (fld->data_at()) {
  case Field::NODE :
    {
      typename Fld::mesh_type::Node::iterator iter, end;
      mesh->begin(iter);
      mesh->end(end);
      fill_data((Fld*)out.get_rep(), inrrd, iter, end);
    }
  break;
  case Field::EDGE :
    {
      typename Fld::mesh_type::Edge::iterator iter, end;
      mesh->begin(iter);
      mesh->end(end);
      fill_data((Fld*)out.get_rep(), inrrd, iter, end);
    }
  break;
  case Field::FACE :
    {
      typename Fld::mesh_type::Face::iterator iter, end;
      mesh->begin(iter);
      mesh->end(end);
      fill_data((Fld*)out.get_rep(), inrrd, iter, end);
    }
  break;
  case Field::CELL :
    {
      typename Fld::mesh_type::Cell::iterator iter, end;
      mesh->begin(iter);
      mesh->end(end);
      fill_data((Fld*)out.get_rep(), inrrd, iter, end);
    }
  break;
  default:
    return false;
  }

  return true;
}

} // end namespace SCIRun

#endif // ConvertToField_h
