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
				SCIRun::FieldHandle &, const int a0_size) = 0;
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

class ConvertToFieldEigenBase : public DynamicAlgoBase
{
public:
  virtual bool convert_to_field(SCIRun::FieldHandle, NrrdDataHandle, 
				SCIRun::FieldHandle &, const int a0_size) = 0;
  virtual ~ConvertToFieldEigenBase();

  static const string& get_h_file_path();
  static string dyn_file_name(const TypeDescription *td) {
    // add no extension.
    return template_class_name() + "." + td->get_filename() + ".";
  }

  static const string base_class_name() {
    static string name("ConvertToFieldEigenBase");
    return name;
  }

  static const string template_class_name() {
    static string name("ConvertToFieldEigen");
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
				SCIRun::FieldHandle &, const int a0_size);
};


template <class Fld>
class ConvertToFieldEigen : public ConvertToFieldEigenBase
{
public:
  //! virtual interface.
  virtual bool convert_to_field(SCIRun::FieldHandle, NrrdDataHandle in, 
				SCIRun::FieldHandle &, const int a0_size);
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
ConvertToField<Fld>::convert_to_field(SCIRun::FieldHandle fld, 
				      NrrdDataHandle      in,
				      SCIRun::FieldHandle &out,
				      const int a0_size)
{
  Nrrd *inrrd = in->nrrd;

  vector<unsigned int> dims;
  // The input fld in not neccessarily the exact type Fld, 
  // it will have the exact same mesh type however.
  //Fld *f = dynamic_cast<Fld*>(fld.get_rep());
  //ASSERT(f != 0);
  //typename Fld::mesh_handle_type m = f->get_typed_mesh(); 

  //const string data_name(fld->get_type_description(1)->get_name());
  //const string vec("Vector");
  //const string ten("Tensor");

  //if ((data_name == ten || data_name == vec) && a0_size == 1) {
  // Old field was tensor or vector data but is no longer
  //return false;
  //} 

  typedef typename Fld::mesh_type Msh;
  Msh *mesh = dynamic_cast<Msh*>(fld->mesh().get_rep());
  ASSERT(mesh != 0);
  int off = 0;
  bool uns = false;
  if (! mesh->get_dim(dims)) {
    // Unstructured fields fall into this category, for them we create nrrds
    // of dimension 1 (2 if vector or scalar data).
    uns = true;
    switch (fld->basis_order()) {
    case 1:
      {
	typename Fld::mesh_type::Node::size_type sz;
	mesh->size(sz);
	dims.push_back(sz);
      }
    break;
    case 0:
      {
	if (mesh->dimensionality() == 1) {
	  typename Fld::mesh_type::Edge::size_type sz;
	  mesh->size(sz);
	  dims.push_back(sz);
	} else if (mesh->dimensionality() == 2) {
	  typename Fld::mesh_type::Face::size_type sz;
	  mesh->size(sz);
	  dims.push_back(sz);
	} else if (mesh->dimensionality() == 3) {
	  typename Fld::mesh_type::Cell::size_type sz;
	  mesh->size(sz);
	  dims.push_back(sz);
	}
      }
    break;
    default:
      return false;
    }
    
    // if vector/tensor data store 3 or 7 at the end of dims vector
    if (a0_size > 1) 
      dims.push_back(a0_size);
  }
  if ((!uns) && fld->basis_order() == 0) {
    off = 1;
  }

  // If the data was vector or tensor it will have an extra axis.
  // It is axis 0.  Make sure sizes along each dim still match.
  if (inrrd->dim != (int)dims.size()) {
    return false;
  }

  // If a0_size equals 3 or 7 then the first axis contains
  // vector or tensor data and a ND nrrd would convert
  // to a (N-1)D type field. 

  int field_dim = inrrd->dim;
  if (a0_size > 1) // tensor or vector data in first dimension
    field_dim -= 1;
  switch (field_dim) {
  case 1:
    {
      // make sure size of dimensions match up
      unsigned int nx = 0;
      if (a0_size > 1) {
	nx = inrrd->axis[1].size + off;
      } else {
	nx = inrrd->axis[0].size + off;
      }
      if (nx != dims[0]) { return false; }
    }
    break;
  case 2:
    {
      unsigned int nx = 0, ny = 0;
      if (a0_size > 1) {
	nx = inrrd->axis[1].size + off;
	ny = inrrd->axis[2].size + off;
      } else {
	nx = inrrd->axis[0].size + off;
	ny = inrrd->axis[1].size + off;
      }
      if ((nx != dims[0]) || (ny != dims[1])) {
	return false;
      }
    }
    break;
  case 3:
    {
      unsigned int nx = 0, ny = 0, nz = 0;
      if (a0_size > 1) {
	nx = inrrd->axis[1].size + off;
	ny = inrrd->axis[2].size + off;
        nz = inrrd->axis[3].size + off;
      } else {
	nx = inrrd->axis[0].size + off;
	ny = inrrd->axis[1].size + off;
        nz = inrrd->axis[2].size + off;
      }
      if ((nx != dims[0]) || (ny != dims[1]) || (nz != dims[2])) {
	return false;
      }
    }
    break;
  default:   // anything else is invalid.
    return false;
  }

  // Things match up, create the new output field.
  out = new Fld(typename Fld::mesh_handle_type(mesh), fld->basis_order());
  
  // Copy all of the non-transient properties from the original field.
  *((PropertyManager *)(out.get_rep()))=*((PropertyManager *)(fld.get_rep()));

  // Copy the data into the field.
  switch (fld->basis_order()) {
  case 1:
    {
      typename Fld::mesh_type::Node::iterator iter, end;
      mesh->begin(iter);
      mesh->end(end);
      fill_data((Fld*)out.get_rep(), inrrd, iter, end);
    }
  break;
  case 0:
    {
      if (mesh->dimensionality() == 1) {
	typename Fld::mesh_type::Edge::iterator iter, end;
	mesh->begin(iter);
	mesh->end(end);
	fill_data((Fld*)out.get_rep(), inrrd, iter, end);	
      } else if (mesh->dimensionality() == 2) {
	typename Fld::mesh_type::Face::iterator iter, end;
	mesh->begin(iter);
	mesh->end(end);
	fill_data((Fld*)out.get_rep(), inrrd, iter, end);	
      } else if (mesh->dimensionality() == 3) {
	typename Fld::mesh_type::Cell::iterator iter, end;
	mesh->begin(iter);
	mesh->end(end);
	fill_data((Fld*)out.get_rep(), inrrd, iter, end);
      }
    }
    break;
  default:
    return false;
  }
  
  return true;
}

template <class Fld>
bool
ConvertToFieldEigen<Fld>::convert_to_field(SCIRun::FieldHandle fld, 
					   NrrdDataHandle      in,
					   SCIRun::FieldHandle &out,
					   const int a0_size)
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
    // Unstructured fields fall into this category, for them we create nrrds
    // of dimension 1 (2 with the tuple axis).
    uns = true;
    switch (fld->basis_order()) {
    case 1:
      {
	typename Fld::mesh_type::Node::size_type sz;
	mesh->size(sz);
	dims.push_back(sz);
      }
      break;
    case 0:
      {
	if (mesh->dimensionality() == 1) {
	  typename Fld::mesh_type::Edge::size_type sz;
	  mesh->size(sz);
	  dims.push_back(sz);
	} else if (mesh->dimensionality() == 2) {
	  typename Fld::mesh_type::Face::size_type sz;
	  mesh->size(sz);
	  dims.push_back(sz);
	} else if (mesh->dimensionality() == 3) {
	  typename Fld::mesh_type::Cell::size_type sz;
	  mesh->size(sz);
	  dims.push_back(sz);
	}
      }
      break;
    default:
      return false;
    }
    // if vector/tensor data store 3 or 7 at the end of dims vector
    if (a0_size > 1) 
      dims.push_back(a0_size);
  }
  if ((!uns) && fld->basis_order() == 0) {
    off = 1;
  }

  // If the data was vector or tensor it will have an extra axis.
  // It is axis 0.  Make sure sizes along each dim still match.
  if (inrrd->dim != (int)dims.size()) {
    return false;
  }

  // If a0_size equals 3 or 7 then the first axis contains
  // vector or tensor data and a ND nrrd would convert
  // to a (N-1)D type field. 

  int field_dim = inrrd->dim;
  if (a0_size > 1) // tensor or vector data in first dimension
    field_dim -= 1;
  switch (field_dim) {
  case 1:
    {
      // make sure size of dimensions match up
      unsigned int nx = inrrd->axis[1].size + off;
      if (nx != dims[0]) { return false; }
    }
    break;
  case 2:
    {
      unsigned int nx = inrrd->axis[0].size + off;
      unsigned int ny = inrrd->axis[1].size + off;
      if ((nx != dims[0]) || (ny != dims[1])) {
	return false;
      }
    }
    break;
  case 3:
    {
      unsigned int nx = inrrd->axis[0].size + off;
      unsigned int ny = inrrd->axis[1].size + off;
      unsigned int nz = inrrd->axis[2].size + off;
      if ((nx != dims[0]) || (ny != dims[1]) || (nz != dims[2])) {
	return false;
      }
    }
    break;
  default:   // anything else is invalid.
    return false;
  }

  // Things match up, create the new output field.
  out = new Fld(typename Fld::mesh_handle_type(mesh), fld->basis_order());
  
  // Copy all of the non-transient properties from the original field.
  *((PropertyManager *)(out.get_rep()))=*((PropertyManager *)(fld.get_rep()));

  // Copy the data into the field.
  switch (fld->basis_order()) {
  case 1:
    {
      typename Fld::mesh_type::Node::iterator iter, end;
      mesh->begin(iter);
      mesh->end(end);
      fill_eigen_data((Fld*)out.get_rep(), inrrd, iter, end);
    }
    break;
  case 0:
    {
      if (mesh->dimensionality() == 1) {
	typename Fld::mesh_type::Edge::iterator iter, end;
	mesh->begin(iter);
	mesh->end(end);
	fill_eigen_data((Fld*)out.get_rep(), inrrd, iter, end);
      } else if (mesh->dimensionality() == 2) {
	typename Fld::mesh_type::Face::iterator iter, end;
	mesh->begin(iter);
	mesh->end(end);
	fill_eigen_data((Fld*)out.get_rep(), inrrd, iter, end);
      } else if (mesh->dimensionality() == 3) {
	typename Fld::mesh_type::Cell::iterator iter, end;
	mesh->begin(iter);
	mesh->end(end);
	fill_eigen_data((Fld*)out.get_rep(), inrrd, iter, end);	  
      }
    }
    break;
  default:
    return false;
  }
  return true;
}

} // end namespace SCIRun

#endif // ConvertToField_h
