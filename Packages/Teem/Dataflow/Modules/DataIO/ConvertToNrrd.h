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
//    File   : ConvertToNrrd.h
//    Author : Martin Cole
//    Date   : Tue Jan  7 09:55:15 2003

#if !defined(ConvertToNrrd_h)
#define ConvertToNrrd_h


#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/QuadraticTetVolField.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Teem/Core/Datatypes/NrrdData.h>
#include <Core/Geometry/BBox.h>
#include <iostream>

namespace SCIRun {
using namespace SCITeem;
using namespace std;

//! ConvertToNrrdBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! ConvertToNrrdBase from the DynamicAlgoBase they will have a pointer to.
class ConvertToNrrdBase : public DynamicAlgoBase
{
public:
  virtual NrrdDataHandle convert_to_nrrd(SCIRun::FieldHandle in) = 0;
  virtual ~ConvertToNrrdBase();

  static const string& get_h_file_path();
  static string dyn_file_name(const TypeDescription *td) {
    // add no extension.
    return template_class_name() + "." + td->get_filename() + ".";
  }

  static const string base_class_name() {
    static string name("ConvertToNrrdBase");
    return name;
  }

  static const string template_class_name() {
    static string name("ConvertToNrrd");
    return name;
  }

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *td);
};

template <class Fld>
class ConvertToNrrd : public ConvertToNrrdBase
{
public:
  //! virtual interface.
  virtual NrrdDataHandle convert_to_nrrd(SCIRun::FieldHandle in);
};


// nrrd Types that we need to convert to:
//  nrrdTypeChar,          
//  nrrdTypeUChar,         
//  nrrdTypeShort,         
//  nrrdTypeUShort,        
//  nrrdTypeInt,           
//  nrrdTypeUInt,          
//  nrrdTypeLLong,         
//  nrrdTypeULLong,        
//  nrrdTypeFloat,         
//  nrrdTypeDouble,

template <class T>
unsigned int get_nrrd_type();

template <>
unsigned int get_nrrd_type<char>();

template <>
unsigned int get_nrrd_type<unsigned char>();

template <>
unsigned int get_nrrd_type<short>();

template <>
unsigned int get_nrrd_type<unsigned short>();

template <>
unsigned int get_nrrd_type<int>();

template <>
unsigned int get_nrrd_type<unsigned int>();

template <>
unsigned int get_nrrd_type<long long>();

template <>
unsigned int get_nrrd_type<unsigned long long>();

template <>
unsigned int get_nrrd_type<float>();

template <class T>
unsigned int get_nrrd_type() {
  return nrrdTypeDouble;
}

template <class T>
void fill_data(T &, double *) {
  ASSERTFAIL("should be only be called with Tensor or Vector types");
}

template <>
void fill_data<Tensor>(Tensor &t, double *p); 

template <>
void fill_data<Vector>(Vector &v, double *p);

template <class Fdata>
void* get_raw_data_ptr(Fdata &, int);

template <>
void* 
get_raw_data_ptr<FData2d<char> >(FData2d<char> &, int);

template <>
void* 
get_raw_data_ptr<FData2d<unsigned char> >(FData2d<unsigned char> &, int);

template <>
void* 
get_raw_data_ptr<FData2d<short> >(FData2d<short> &, int);

template <>
void* 
get_raw_data_ptr<FData2d<unsigned short> >(FData2d<unsigned short> &, int);

template <>
void* get_raw_data_ptr<FData2d<int> >(FData2d<int> &, int);

template <>
void* 
get_raw_data_ptr<FData2d<unsigned int> >(FData2d<unsigned int> &, int);

template <>
void* 
get_raw_data_ptr<FData2d<long long> >(FData2d<long long> &, int);

template <>
void* 
get_raw_data_ptr<FData2d<unsigned long long> >(FData2d<unsigned long long> &, int);

template <>
void* 
get_raw_data_ptr<FData2d<float> >(FData2d<float> &, int);

template <>
void* 
get_raw_data_ptr<FData2d<double> >(FData2d<double> &, int);

template <>
void* 
get_raw_data_ptr<FData3d<char> >(FData3d<char> &, int);

template <>
void* 
get_raw_data_ptr<FData3d<unsigned char> >(FData3d<unsigned char> &, int);

template <>
void* 
get_raw_data_ptr<FData3d<short> >(FData3d<short> &, int);

template <>
void* 
get_raw_data_ptr<FData3d<unsigned short> >(FData3d<unsigned short> &, int);

template <>
void* get_raw_data_ptr<FData3d<int> >(FData3d<int> &, int);

template <>
void* 
get_raw_data_ptr<FData3d<unsigned int> >(FData3d<unsigned int> &, int);

template <>
void* 
get_raw_data_ptr<FData3d<long long> >(FData3d<long long> &, int);

template <>
void* 
get_raw_data_ptr<FData3d<unsigned long long> >(FData3d<unsigned long long> &, int);

template <>
void* 
get_raw_data_ptr<FData3d<float> >(FData3d<float> &, int);

template <>
void* 
get_raw_data_ptr<FData3d<double> >(FData3d<double> &, int);

template <>
void* 
get_raw_data_ptr<FData3d<double> >(FData3d<double> &, int);

template <>
void* 
get_raw_data_ptr<FData3d<Vector> >(FData3d<Vector> &, int);

template <>
void* 
get_raw_data_ptr<FData3d<Tensor> >(FData3d<Tensor> &, int);

template <class Fdata>
void* get_raw_data_ptr(Fdata &data, int pad) {

  if (pad > 0) {
    int sz = data.size() * pad;
    double *new_data = new double[sz];
    double *p = new_data;
    typename Fdata::iterator iter = data.begin();
    while (iter != data.end()) {
      fill_data(*iter, p);
      ++iter;
      p += pad;
    }
    return new_data;
  }
  return &(data[0]); // no copy just wrap this pointer
}

template <class Fld>
NrrdDataHandle
convert(FieldHandle ifld) 
{
  typedef typename Fld::value_type val_t;
  Fld *f = dynamic_cast<Fld*>(ifld.get_rep());
  if (!f) { return 0; }

  typename Fld::mesh_handle_type m = f->get_typed_mesh(); 

  const string data_name(ifld->get_type_description(1)->get_name());
  const string vec("Vector");
  const string ten("Tensor");

  NrrdData *nout = 0;
  int pad_data = 0; // if 0 then no copy is made 
  if (data_name == ten) {
    pad_data = 6; // copy the data, and pad for tensor values
  } else if (data_name== vec) {
    pad_data = 3; // copy the data and pad for vector values
  }

  nout = scinew NrrdData(pad_data);

  vector<unsigned int> dims;
  Vector spc;
  Point minP, maxP;
  bool with_spacing = true;

  if (m->get_dim(dims)) {
    BBox bbox = m->get_bounding_box();
    minP = bbox.min();
    maxP = bbox.max();
    spc = maxP - minP;
    spc.x(spc.x() / (dims[0] - 1));
    spc.y(spc.y() / (dims[1] - 1));
    spc.z(spc.z() / (dims[2] - 1));
  } else {

    unsigned int sz = 0;
    switch(f->data_at()) {
    case Field::NODE :
      {
	typename Fld::mesh_type::Node::size_type size;
	m->synchronize(Mesh::NODES_E);
	m->size(size);
	sz = size;
      }
    break;
    case Field::EDGE :
      {
	typename Fld::mesh_type::Edge::size_type size;
	m->synchronize(Mesh::NODES_E);
	m->size(size);
	sz = size;
      }
    break;
    case Field::FACE :
      {
	typename Fld::mesh_type::Face::size_type size;
	m->synchronize(Mesh::NODES_E);
	m->size(size);
	sz = size;
      }
    break;
    case Field::CELL:
      {
	typename Fld::mesh_type::Cell::size_type size;
	m->synchronize(Mesh::CELLS_E);
	m->size(size);
	sz = size;
      }
      break;
    default:
      // error("wtf?");
      return 0;
    }
    dims.push_back(sz);
    with_spacing = false;
  }

  int dim = dims.size();

  switch(dim) {
  case 1: 
    {
      nrrdWrap(nout->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
	       get_nrrd_type<val_t>(), 1, dims[0]);
      if (f->data_at() == Field::NODE) {
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterNode);
      } else if (f->data_at() == Field::CELL) {
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterCell);
      } else  {
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter, nrrdCenterUnknown);
      }
    }
    break;
  case 2:
    {
      nrrdWrap(nout->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
	       get_nrrd_type<val_t>(), 2, dims[0], dims[1]);

      if (f->data_at() == Field::NODE) {
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter,
		    nrrdCenterNode, nrrdCenterNode);
      } else if (f->data_at() == Field::CELL) {
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter,
		    nrrdCenterCell, nrrdCenterCell);
      } else  {
	nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter,
		    nrrdCenterUnknown, nrrdCenterUnknown);
      }
    }
    break;
  case 3:
  default:
    nrrdWrap(nout->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
	     get_nrrd_type<val_t>(), 3, dims[0], dims[1], dims[2]);
    if (f->data_at() == Field::NODE) {
      nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter,
		  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);
    } else if (f->data_at() == Field::CELL) {
      nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter,
		  nrrdCenterCell, nrrdCenterCell, nrrdCenterCell);
    } else  {
      nrrdAxesSet(nout->nrrd, nrrdAxesInfoCenter,
		  nrrdCenterUnknown, nrrdCenterUnknown, nrrdCenterUnknown);
    }
  }

  int offset=0;
  if (data_name == vec || data_name == ten) offset=1;

  nout->nrrd->axis[0+offset].label = strdup("x");
  if (with_spacing) {
    nout->nrrd->axis[0+offset].min=minP.x();
    nout->nrrd->axis[0+offset].max=maxP.x();
    nout->nrrd->axis[0+offset].spacing=spc.x();
  }

  if (dim > 1) {
    nout->nrrd->axis[1+offset].min=minP.y();
    nout->nrrd->axis[1+offset].max=maxP.y();
    nout->nrrd->axis[1+offset].spacing=spc.y();
    nout->nrrd->axis[1+offset].label = strdup("y");
  }
  if (dim > 2) {
    nout->nrrd->axis[2+offset].min=minP.z();
    nout->nrrd->axis[2+offset].max=maxP.z();
    nout->nrrd->axis[2+offset].spacing=spc.z();
    nout->nrrd->axis[2+offset].label = strdup("z");
  }

  if (data_name == vec) {
    nout->nrrd->axis[0].label = strdup("v");
  } else if (data_name == ten) {
    nout->nrrd->axis[0].label = strdup("t");
  }

  return NrrdDataHandle(nout);
}

template <class Fld>
NrrdDataHandle
ConvertToNrrd<Fld>::convert_to_nrrd(FieldHandle ifh)
{
  Fld *fld = dynamic_cast<Fld*>(ifh.get_rep());
  ASSERT(fld != 0);
  //  typname FLD::mesh_handle_type mh = fld->get_typed_mesh();
  return convert<Fld>(ifh);
}




} // end namespace SCIRun

#endif // ConvertToNrrd_h
