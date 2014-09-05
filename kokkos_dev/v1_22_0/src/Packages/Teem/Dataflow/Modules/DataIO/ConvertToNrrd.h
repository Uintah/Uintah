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
  virtual bool convert_to_nrrd(SCIRun::FieldHandle, NrrdDataHandle &pointsH,
					 NrrdDataHandle &connectH, NrrdDataHandle &dataH,
					 string &) = 0;
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
  virtual bool convert_to_nrrd(SCIRun::FieldHandle, NrrdDataHandle &pointsH,
					 NrrdDataHandle &connectH, NrrdDataHandle &dataH,
					 string &);
};


template <class T>
void fill_data(T &, double *) {
  ASSERTFAIL("should be only be called with Tensor or Vector types");
}

template <class T>
void fill_data(T &, double *);

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
get_raw_data_ptr<FData2d<Vector> >(FData2d<Vector> &, int);

template <>
void* 
get_raw_data_ptr<FData2d<Tensor> >(FData2d<Tensor> &, int);

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

template <>
void* 
get_raw_data_ptr<vector<Vector> >(vector<Vector> &, int);

template <>
void* 
get_raw_data_ptr<vector<Tensor> >(vector<Tensor> &, int);

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
bool
ConvertToNrrd<Fld>::convert_to_nrrd(FieldHandle ifh, NrrdDataHandle &pointsH,
				    NrrdDataHandle &connectH, NrrdDataHandle &dataH,
				    string &label)
{
  typedef typename Fld::value_type val_t;
  Fld *f = dynamic_cast<Fld*>(ifh.get_rep());

  ASSERT(f != 0);

  // FIX ME : ASSUME FIELD HAS DATA

  typename Fld::mesh_handle_type m = f->get_typed_mesh(); 

  const string data_name(ifh->get_type_description(1)->get_name());
  const string vec("Vector");
  const string ten("Tensor");

  bool has_data = true;
  NrrdData *ndata = 0;
  NrrdData *npoints = 0;
  NrrdData *nconnect = 0;
  int pad_data = 0; // if 0 then no copy is made 
  int sink_size = 1;
  string sink_label = label + string(":Scalar");
  int kind = nrrdKindScalar;
  if (data_name == ten) {
    pad_data = 7; // copy the data, and pad for tensor values
    sink_size = 7;
    kind = nrrdKind3DMaskedSymTensor;
    sink_label = label + string(":Tensor");
  } else if (data_name== vec) {
    pad_data = 3; // copy the data and pad for vector values
    sink_size = 3;
    sink_label = label + string(":Vector");
    kind = nrrdKind3Vector;
  }
    
  vector<unsigned int> dims;
  Vector spc;
  Point minP, maxP;
  bool with_spacing = true;
  if (m->get_dim(dims)) {
    // structured
    BBox bbox = m->get_bounding_box();
    minP = bbox.min();
    maxP = bbox.max();
    spc = maxP - minP;
    spc.x(spc.x() / (dims[0] - 1));
    spc.y(spc.y() / (dims[1] - 1));
    spc.z(spc.z() / (dims[2] - 1));
  } else {
    // unstructured data so create a 1D nrrd (2D if vector or tensor)
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
      has_data = false;
    }
    dims.push_back(sz); 

    with_spacing = false;
  }

  // create the Points and Connections Nrrds
  // FIX ME : FINISH CASE STATEMENTS
  npoints = scinew NrrdData();
  nconnect = scinew NrrdData();

  switch(f->data_at()) {
  case Field::NODE:
  case Field::EDGE:
  case Field::CELL:
    {
      switch(dims.size()) {
      case 1:
	nrrdAlloc(npoints->nrrd, nrrdTypeDouble, 2, 3, dims[0]);
	break;
      case 2:
	nrrdAlloc(npoints->nrrd, nrrdTypeDouble, 3, 3, dims[0], dims[1]);
	break;
      case 3:
	nrrdAlloc(npoints->nrrd, nrrdTypeDouble, 4, 3, dims[0], dims[1], dims[2]);
	break;
      }

      typename Fld::mesh_type::Node::iterator iter, end;
      m->begin(iter);
      m->end(end);
      double *data = (double*)npoints->nrrd->data;
      while(iter != end) {
	Point p;
	m->get_point(p,*iter);
	data[0] = p.x();
	data[1] = p.y();
	data[2] = p.z();
	data += 3;
	++iter;
      }

      typename Fld::mesh_type::Elem::iterator iter2, end2;
      m->begin(iter2);
      m->end(end2);

      typename Fld::mesh_type::Node::array_type array;
      typename Fld::mesh_type::Elem::size_type nelems;  
      
      // get the number of elements and number of points per element
      // and allocate the nrrd
      m->size(nelems);
      m->get_nodes(array ,*iter2);
      if (array.size() == 1) 
	nrrdAlloc(nconnect->nrrd, nrrdTypeInt, 1, (int)nelems);
      else
	nrrdAlloc(nconnect->nrrd, nrrdTypeInt, 2, array.size(), (int)nelems);

      int* data2 = (int*)nconnect->nrrd->data;

      while(iter2 != end2) {
	m->get_nodes(array ,*iter2);
	
	// copy into nrrd
	int i = 0;
	for(i=0; i<(int)array.size(); i++) {
	  data2[i] = array[i];
	}	
	data2 += i;
	++iter2;
      }
    }
    break;
  default:
    return false;
  }
  pointsH = npoints;
  connectH = nconnect;

  
  // if vector/tensor data push to end of dims vector
  if (pad_data > 0) { dims.push_back(pad_data); }

  // create the Data Nrrd
  if (has_data) {
    ndata = scinew NrrdData(pad_data);
    // switch based on the dims size because that is the size
    // of nrrd to create
    int dim = dims.size();
    switch(dim) {
    case 1: 
      {
	// must be scalar data if only one dimension
	if(pad_data > 0) {
	  cerr << "Must be scalar data if only one dimension\n";
	  return false;
	}
	
	nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		 get_nrrd_type<val_t>(), 1, dims[0]);
	
	if (f->data_at() == Field::NODE) {
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			  nrrdCenterNode);
	} else if (f->data_at() == Field::CELL) {
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter, nrrdCenterCell,
			  nrrdCenterCell);
	} else  {
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter, nrrdCenterUnknown,
			  nrrdCenterUnknown);
	}
	ndata->nrrd->axis[0].label = strdup("x");
	
	if (with_spacing) {
	  ndata->nrrd->axis[0].min=minP.x();
	  ndata->nrrd->axis[0].max=maxP.x();
	  ndata->nrrd->axis[0].spacing=spc.x();
	}
	
	ndata->nrrd->axis[0].kind = nrrdKindDomain;
      }
      break;
    case 2:
      {
	// vector/tensor data stored as [x][3] or [x][7]
	if (pad_data > 0) {
	  nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		   get_nrrd_type<val_t>(), 2, pad_data, dims[0]);
	  ndata->nrrd->axis[0].kind = kind;
	} else {
	  nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		   get_nrrd_type<val_t>(), 2, dims[0], dims[1]);
	  ndata->nrrd->axis[0].kind = nrrdKindDomain;
	}
	ndata->nrrd->axis[1].kind = nrrdKindDomain;

	if (f->data_at() == Field::NODE) {
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter,
			  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);
	} else if (f->data_at() == Field::CELL) {
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter,
			  nrrdCenterCell, nrrdCenterCell, nrrdCenterCell);
	} else  {
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter, nrrdCenterUnknown,
			  nrrdCenterUnknown, nrrdCenterUnknown);
	}

	if (pad_data > 0) {
	  // 1D nrrd with vector/tenssor
	  ndata->nrrd->axis[0].label = strdup(sink_label.c_str());
	  ndata->nrrd->axis[1].label = strdup("x");

	  if (with_spacing) {
	    ndata->nrrd->axis[1].min=minP.x();
	    ndata->nrrd->axis[1].max=maxP.x();
	    ndata->nrrd->axis[1].spacing=spc.x();
	  }
	} else {
	  // 2D nrrd of scalars
	  ndata->nrrd->axis[0].label = strdup("x");
	  ndata->nrrd->axis[1].label = strdup("y");
	
	  if (with_spacing) {
	    ndata->nrrd->axis[0].min=minP.x();
	    ndata->nrrd->axis[0].max=maxP.x();
	    ndata->nrrd->axis[0].spacing=spc.x();
	    ndata->nrrd->axis[1].min=minP.y();
	    ndata->nrrd->axis[1].max=maxP.y();
	    ndata->nrrd->axis[1].spacing=spc.y();
	  }
	}
      }
      break;
    case 3:
      {
	if (f->data_at() == Field::NODE) {
	  if (pad_data > 0) {
	    // 2D nrrd with vector/tensor NODE
	    nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		     get_nrrd_type<val_t>(), 3, pad_data,
		     dims[0], dims[1]);
	    ndata->nrrd->axis[0].kind = kind;
	  } else {
	    // 3D nrrd of scalars NODE
	    nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		     get_nrrd_type<val_t>(), 3, dims[0], dims[1], dims[2]);
	    ndata->nrrd->axis[0].kind = nrrdKindDomain;
	  }
	  ndata->nrrd->axis[1].kind = nrrdKindDomain;
	  ndata->nrrd->axis[2].kind = nrrdKindDomain;

	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter,
			  nrrdCenterNode, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode);
	} else if (f->data_at() == Field::CELL) {
	  if (pad_data > 0) {
	    // 2D nrrd with vector/tensor CELL
	    nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		     get_nrrd_type<val_t>(), 3, pad_data, 
		     dims[0] - 1, dims[1] - 1);
	  } else {
	    // 3D nrrd of scalars CELL
	    nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		     get_nrrd_type<val_t>(), 3, 
		     dims[0] - 1, dims[1] - 1, dims[2] - 1);
	  }
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter,
			  nrrdCenterNode, nrrdCenterCell, 
			  nrrdCenterCell, nrrdCenterCell);
	} else  {
	  ASSERTFAIL("no support for edge or face centers");
	}

	// set labels
	if (pad_data > 0) {
	  ndata->nrrd->axis[0].label = strdup(sink_label.c_str());
	  ndata->nrrd->axis[1].label = strdup("x");
	  ndata->nrrd->axis[2].label = strdup("y");
	} else {
	  ndata->nrrd->axis[0].label = strdup("x");
	  ndata->nrrd->axis[1].label = strdup("y");
	  ndata->nrrd->axis[2].label = strdup("z");
	}

	// set min, max, and spacing
	if (with_spacing) {
	  if (pad_data > 0) {
	    // 2D nrrd with vector/tensor
	    ndata->nrrd->axis[1].min=minP.x();
	    ndata->nrrd->axis[1].max=maxP.x();
	    ndata->nrrd->axis[1].spacing=spc.x();
	    ndata->nrrd->axis[2].min=minP.y();
	    ndata->nrrd->axis[2].max=maxP.y();
	    ndata->nrrd->axis[2].spacing=spc.y();
	  } else {
	    // 3D nrrd with scalars
	    ndata->nrrd->axis[0].min=minP.x();
	    ndata->nrrd->axis[0].max=maxP.x();
	    ndata->nrrd->axis[0].spacing=spc.x();
	    ndata->nrrd->axis[1].min=minP.y();
	    ndata->nrrd->axis[1].max=maxP.y();
	    ndata->nrrd->axis[1].spacing=spc.y();
	    ndata->nrrd->axis[2].min=minP.z();
	    ndata->nrrd->axis[2].max=maxP.z();
	    ndata->nrrd->axis[2].spacing=spc.z();
	  }
	}
      }
      break;
    case 4:
      {
	// must be 3D vector/tensor data
	if (pad_data == 1) {
	  cerr << "Must be vector/tensor data\n";
	  return false;
	}

	ndata->nrrd->axis[0].kind = kind;
	ndata->nrrd->axis[1].kind = nrrdKindDomain;
	ndata->nrrd->axis[2].kind = nrrdKindDomain;
	ndata->nrrd->axis[3].kind = nrrdKindDomain;

	if (f->data_at() == Field::NODE) {
	  nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		   get_nrrd_type<val_t>(), 4, pad_data,
		   dims[0], dims[1], dims[2]);

	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter,
			  nrrdCenterNode, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode);
	} else if (f->data_at() == Field::CELL) {
	  nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		   get_nrrd_type<val_t>(), 4, pad_data, 
		   dims[0] - 1, dims[1] - 1, dims[2]-1);
	
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter,
			  nrrdCenterNode, nrrdCenterCell, 
			  nrrdCenterCell, nrrdCenterCell);
	} else  {
	  ASSERTFAIL("no support for edge or face centers");
	}

	// set labels
	ndata->nrrd->axis[0].label = strdup(sink_label.c_str());
	ndata->nrrd->axis[1].label = strdup("x");
	ndata->nrrd->axis[2].label = strdup("y");
	ndata->nrrd->axis[3].label = strdup("z");
      
	if (with_spacing) {
	  ndata->nrrd->axis[1].min=minP.x();
	  ndata->nrrd->axis[1].max=maxP.x();
	  ndata->nrrd->axis[1].spacing=spc.x();
	  ndata->nrrd->axis[2].min=minP.y();
	  ndata->nrrd->axis[2].max=maxP.y();
	  ndata->nrrd->axis[2].spacing=spc.y();
	  ndata->nrrd->axis[3].min=minP.z();
	  ndata->nrrd->axis[3].max=maxP.z();
	  ndata->nrrd->axis[3].spacing=spc.z();
	} 
      }
      break;
    default:
      break;
    }
    dataH = ndata;
  }
  return true;
}

} // end namespace SCIRun

#endif // ConvertToNrrd_h
