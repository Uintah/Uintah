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

#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/ImageField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/QuadraticTetVolField.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Geometry/BBox.h>
#include <iostream>

namespace SCIRun {
using namespace std;

//! ConvertToNrrdBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! ConvertToNrrdBase from the DynamicAlgoBase they will have a pointer to.
class ConvertToNrrdBase : public DynamicAlgoBase
{
public:
  virtual bool convert_to_nrrd(SCIRun::FieldHandle, NrrdDataHandle &pointsH,
			       NrrdDataHandle &connectH, NrrdDataHandle &dataH,
			       bool compute_points_p, bool compute_connects_p,
			       bool compute_data_p, const string &) = 0;
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
			       bool compute_points_p, bool compute_connects_p,
			       bool compute_data_p, const string &);
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
void* 
get_raw_data_ptr<FData2d<int> >(FData2d<int> &, int);

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
void* 
get_raw_data_ptr<FData3d<int> >(FData3d<int> &, int);

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

  if (pad > 3) {
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
				    NrrdDataHandle &connectH,
				    NrrdDataHandle &dataH,
				    bool compute_points_p,
				    bool compute_connects_p,
				    bool compute_data_p,
				    const string &label)
{
  typedef typename Fld::value_type val_t;
  Fld *f = dynamic_cast<Fld*>(ifh.get_rep());

  ASSERT(f != 0);

  typename Fld::mesh_handle_type m = f->get_typed_mesh(); 

  const string data_name(ifh->get_type_description(1)->get_name());
  const string vec("Vector");
  const string ten("Tensor");

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
    kind = nrrdKind3DMaskedSymMatrix;
    sink_label = label + string(":Tensor");
  } else if (data_name== vec) {
    pad_data = 3; // copy the data and pad for vector values
    sink_size = 3;
    sink_label = label + string(":Vector");
    kind = nrrdKind3Vector;
  }
    
  vector<unsigned int> ndims; // node dims
  vector<unsigned int> ddims; // data dims 
  Vector spc;
  Point minP, maxP;
  bool with_spacing = true;
  if (m->get_dim(ndims)) {
    // structured
    BBox bbox = m->get_bounding_box();
    minP = bbox.min();
    maxP = bbox.max();

    spc = maxP - minP;
    if (f->basis_order() == 0)
    {
      spc.x(spc.x() / ndims[0]);
      if (ndims.size() > 1)
      {
        spc.y(spc.y() / ndims[1]);
      }
      if (ndims.size() > 2)
      {
        spc.z(spc.z() / ndims[2]);
      }
    }
    else
    {
      spc.x(spc.x() / (ndims[0] - 1));
      if (ndims.size() > 1)
      {
        spc.y(spc.y() / (ndims[1] - 1));
      }
      if (ndims.size() > 2)
      {
        spc.z(spc.z() / (ndims[2] - 1));
      }
    }
    ddims = ndims;
    if (f->basis_order() == 0)
    {
      for (unsigned int i = 0; i < ddims.size(); i++)
      {
        ddims[i]--;
      }
    }
    else if (f->basis_order() == -1)
    {
      compute_data_p = false;
    }
  } else {
    // unstructured data so create a 1D nrrd (2D if vector or tensor)
    unsigned int nsz = 0;
    unsigned int sz = 0;
    typename Fld::mesh_type::Node::size_type size;
    m->synchronize(Mesh::NODES_E);
    m->size(size);
    nsz = size;
    switch(f->basis_order()) {
    case 1 :
      {
        sz = nsz;
      }
      break;
    case 0:
      {
	if (m->dimensionality() == 0) {
	  typename Fld::mesh_type::Node::size_type size;
	  m->synchronize(Mesh::NODES_E);
	  m->size(size);
	  sz = size;
	} else if (m->dimensionality() == 1) {
	  typename Fld::mesh_type::Edge::size_type size;
	  m->synchronize(Mesh::EDGES_E);
	  m->size(size);
	  sz = size;
	} else if (m->dimensionality() == 2) {
	  typename Fld::mesh_type::Face::size_type size;
	  m->synchronize(Mesh::FACES_E);
	  m->size(size);
	  sz = size;
	} else if (m->dimensionality() == 3) {
	  typename Fld::mesh_type::Cell::size_type size;
	  m->synchronize(Mesh::CELLS_E);
	  m->size(size);
	  sz = size;
	}

      }
      break;
    default:
      compute_data_p = false; // No data to compute.
    }
    ndims.push_back(nsz);
    ddims.push_back(sz); 

    with_spacing = false;
  }

  if (compute_points_p)
  {
    npoints = scinew NrrdData();
    switch(ndims.size()) {
    case 1:
      nrrdAlloc(npoints->nrrd, nrrdTypeDouble, 2, 3, ndims[0]);
      break;
    case 2:
      nrrdAlloc(npoints->nrrd, nrrdTypeDouble, 3, 3,
                ndims[0], ndims[1]);
      break;
    case 3:
      nrrdAlloc(npoints->nrrd, nrrdTypeDouble, 4, 3,
                ndims[0], ndims[1], ndims[2]);
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
  }

  if (compute_connects_p)
  {
    nconnect = scinew NrrdData();
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

  pointsH = npoints;
  connectH = nconnect;

  // if vector/tensor data push to end of dims vector
  if (pad_data > 0) { ddims.push_back(pad_data); }

  // create the Data Nrrd
  if (compute_data_p)
  {
    if (pad_data > 3)
    {
      ndata = scinew NrrdData();
    }
    else
    {
      ndata = scinew NrrdData(f);
    }

    // switch based on the dims size because that is the size
    // of nrrd to create
    int dim = ddims.size();
    switch(dim) {
    case 1: 
      {
	// must be scalar data if only one dimension
	if(pad_data > 0) {
	  cerr << "Must be scalar data if only one dimension\n";
	  return false;
	}
	
	nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		 get_nrrd_type<val_t>(), 1, ddims[0]);
	
	if (f->basis_order() == 1) {
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter, nrrdCenterNode, 
			  nrrdCenterNode);
	} else if (f->basis_order() == 0) {
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter, nrrdCenterCell,
			  nrrdCenterCell);
	} else  {
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter, nrrdCenterUnknown,
			  nrrdCenterUnknown);
	}
	ndata->nrrd->axis[0].label = airStrdup("x");
	
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
		   get_nrrd_type<val_t>(), 2, pad_data, ddims[0]);
	  ndata->nrrd->axis[0].kind = kind;
	} else {
	  nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		   get_nrrd_type<val_t>(), 2, ddims[0], ddims[1]);
	  ndata->nrrd->axis[0].kind = nrrdKindDomain;
	}
	ndata->nrrd->axis[1].kind = nrrdKindDomain;

	if (f->basis_order() == 1) {
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter,
			  nrrdCenterNode, nrrdCenterNode, nrrdCenterNode);
	} else if (f->basis_order() == 0) {
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter,
			  nrrdCenterCell, nrrdCenterCell, nrrdCenterCell);
	} else  {
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter, nrrdCenterUnknown,
			  nrrdCenterUnknown, nrrdCenterUnknown);
	}

	if (pad_data > 0) {
	  // 1D nrrd with vector/tensor
	  ndata->nrrd->axis[0].label = airStrdup(sink_label.c_str());
	  ndata->nrrd->axis[1].label = airStrdup("x");

	  if (with_spacing) {
	    ndata->nrrd->axis[1].min=minP.x();
	    ndata->nrrd->axis[1].max=maxP.x();
	    ndata->nrrd->axis[1].spacing=spc.x();
	  }
	} else {
	  // 2D nrrd of scalars
	  ndata->nrrd->axis[0].label = airStrdup("x");
	  ndata->nrrd->axis[1].label = airStrdup("y");

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
	if (f->basis_order() == 1) {
	  if (pad_data > 0) {
	    // 2D nrrd with vector/tensor NODE
	    nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		     get_nrrd_type<val_t>(), 3, pad_data,
		     ddims[0], ddims[1]);
	    ndata->nrrd->axis[0].kind = kind;
	  } else {
	    // 3D nrrd of scalars NODE
	    nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		     get_nrrd_type<val_t>(), 3, ddims[0], ddims[1], ddims[2]);
	    ndata->nrrd->axis[0].kind = nrrdKindDomain;
	  }
	  ndata->nrrd->axis[1].kind = nrrdKindDomain;
	  ndata->nrrd->axis[2].kind = nrrdKindDomain;

	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter,
			  nrrdCenterNode, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode);
	} else if (f->basis_order() == 0) {
	  if (pad_data > 0) {
	    // 2D nrrd with vector/tensor CELL
	    nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		     get_nrrd_type<val_t>(), 3, pad_data, 
		     ddims[0], ddims[1]);
	  } else {
	    // 3D nrrd of scalars CELL
	    nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		     get_nrrd_type<val_t>(), 3, 
		     ddims[0], ddims[1], ddims[2]);
	  }
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter,
			  nrrdCenterCell, nrrdCenterCell, 
			  nrrdCenterCell, nrrdCenterCell);
	} else  {
	  ASSERTFAIL("no support for edge or face centers");
	}

	// set labels
	if (pad_data > 0) {
	  ndata->nrrd->axis[0].label = airStrdup(sink_label.c_str());
	  ndata->nrrd->axis[1].label = airStrdup("x");
	  ndata->nrrd->axis[2].label = airStrdup("y");
	} else {
	  ndata->nrrd->axis[0].label = airStrdup("x");
	  ndata->nrrd->axis[1].label = airStrdup("y");
	  ndata->nrrd->axis[2].label = airStrdup("z");
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

	if (f->basis_order() == 1) {
	  nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		   get_nrrd_type<val_t>(), 4, pad_data,
		   ddims[0], ddims[1], ddims[2]);

	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter,
			  nrrdCenterNode, nrrdCenterNode, 
			  nrrdCenterNode, nrrdCenterNode);
	} else if (f->basis_order() == 0) {
	  nrrdWrap(ndata->nrrd, get_raw_data_ptr(f->fdata(), pad_data), 
		   get_nrrd_type<val_t>(), 4, pad_data, 
		   ddims[0], ddims[1], ddims[2]);
	
	  nrrdAxisInfoSet(ndata->nrrd, nrrdAxisInfoCenter,
			  nrrdCenterCell, nrrdCenterCell, 
			  nrrdCenterCell, nrrdCenterCell);
	} else  {
	  ASSERTFAIL("no support for edge or face centers");
	}

	// set labels
	ndata->nrrd->axis[0].label = airStrdup(sink_label.c_str());
	ndata->nrrd->axis[1].label = airStrdup("x");
	ndata->nrrd->axis[2].label = airStrdup("y");
	ndata->nrrd->axis[3].label = airStrdup("z");
      
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

    // check for transform
    const string meshstr =
      ifh->get_type_description(0)->get_name().substr(0, 6);
    
    if (!(ifh->mesh()->is_editable() && meshstr != "Struct"))
    {
      Transform t;

      // get the actual transform if possible for later use
      LatVolMesh *lv_m = dynamic_cast<LatVolMesh *> (ifh->mesh().get_rep());
      ImageMesh *i_m = dynamic_cast<ImageMesh *> (ifh->mesh().get_rep());
      ScanlineMesh *s_m = dynamic_cast<ScanlineMesh *> (ifh->mesh().get_rep());
      if(lv_m)
	t = lv_m->get_transform();
      else if (i_m)
	t = i_m->get_transform();
      else if (s_m)
	t = s_m->get_transform();
      else
      {
	cerr << "ERROR: Mesh type must be of type LatVol, Image, or Scanline to get correct transform information\n";
	return false;
      }

      double trans[16];
      t.get(trans);
      string trans_string = "";
      for(int i=0; i<16; i++) {
	trans_string += to_string(trans[i]);
	trans_string += " ";
      }

      dataH = ndata;

      // set the spaceDirection vectors if the transform 
      // matrix is not just a diagonal matrix
      bool axis_aligned = true;
      if( (Abs(trans[1] - 0.0) > 0.0001) ||
	  (Abs(trans[2] - 0.0) > 0.0001) ||
	  (Abs(trans[4] - 0.0) > 0.0001) ||
	  (Abs(trans[6] - 0.0) > 0.0001) ||
	  (Abs(trans[8] - 0.0) > 0.0001) ||
	  (Abs(trans[9] - 0.0) > 0.0001) ||
	  (Abs(trans[12] - 0.0) > 0.0001) ||
	  (Abs(trans[13] - 0.0) > 0.0001) ||
	  (Abs(trans[14] - 0.0) > 0.0001)) {
	axis_aligned = false;
      }

      if (!axis_aligned) {
	// Since we found a transform, set the appropriate
	// space information.
	Nrrd* n = dataH->nrrd;
	
	// set spaceDimension to always be 3
	nrrdSpaceSet(n, nrrdSpaceUnknown);

	// If axis aligned, set the space dimension
	// to be that of the actual number of axes with
	// domain information. But if there is the
	// data is not axis aligned, assume a space
	// of 3 (world space)
	if (axis_aligned) {
	  if (pad_data > 0)
	    nrrdSpaceDimensionSet(n, dim-1);
	  else
	    nrrdSpaceDimensionSet(n, dim);
	} else {
	  nrrdSpaceDimensionSet(n, 3);
	}
	
	// set the spaceOrigin which can be taken from the
	// 4th column of the transform 
	n->spaceOrigin[0] = trans[3];
	n->spaceOrigin[1] = trans[7];
	n->spaceOrigin[2] = trans[11];
	
	// set the spaceDirection to be the corresponding
	// column of the transform matrix
	double dir1[3], dir2[3], dir3[3];
	dir1[0] = trans[0];
	dir1[1] = trans[4];
	dir1[2] = trans[8];
	
	dir2[0] = trans[1];
	dir2[1] = trans[5];
	dir2[2] = trans[9];
	
	dir3[0] = trans[2];
	dir3[1] = trans[6];
	dir3[2] = trans[10];
	
	// vector/tensor data has a vector of AIR_NANs
	// for that axis
	if (pad_data > 0) {
	  double none[3];
	  none[0] = AIR_NAN;
	  none[1] = AIR_NAN;
	  none[2] = AIR_NAN;
	  nrrdAxisInfoSet(n, nrrdAxisInfoSpaceDirection,
			  none, dir1, dir2, dir3);
	}
	else {
	nrrdAxisInfoSet(n, nrrdAxisInfoSpaceDirection,
			dir1, dir2, dir3);
	}
	
	// set min/max and spacing and units to AIR_NAN now that 
	// direction vectors are being used 
	for(int a=0; a<dim; a++) {
	  n->axis[a].min = AIR_NAN;
	  n->axis[a].max = AIR_NAN;
	  n->axis[a].spacing = AIR_NAN;
	  n->axis[a].units = (char*)airFree(n->axis[a].units);
	}
      }
    } else {
      dataH = ndata;
    }
  }
  return true;
}

} // end namespace SCIRun

#endif // ConvertToNrrd_h
