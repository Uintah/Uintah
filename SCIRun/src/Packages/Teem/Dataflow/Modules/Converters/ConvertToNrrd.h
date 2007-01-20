/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <Core/Geometry/Transform.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/FData.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Geometry/BBox.h>
#include <iostream>

#include <Teem/Dataflow/Modules/Converters/share.h>
namespace SCIRun {
using namespace std;

//! ConvertToNrrdBase supports the dynamically loadable algorithm concept.
//! when dynamically loaded the user will dynamically cast to a 
//! ConvertToNrrdBase from the DynamicAlgoBase they will have a pointer to.
class SCISHARE ConvertToNrrdBase : public DynamicAlgoBase
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
inline void fill_data(T &, float *)
{
  ASSERTFAIL("should be only be called with Tensor or Vector types");
}


template <>
inline void
fill_data<Tensor>(Tensor &t, float *p)
{
  p[0] = 1.0;
  p[1] = t.mat_[0][0];
  p[2] = t.mat_[0][1];
  p[3] = t.mat_[0][2];
  p[4] = t.mat_[1][1];
  p[5] = t.mat_[1][2];
  p[6] = t.mat_[2][2];
}


template <class Fdata>
inline void *
get_raw_data_ptr(Fdata &data, int pad)
{
  if (pad > 3)
  {
    // We have to copy tensors from Tensor into masked symetric matrix.
    const size_t sz = data.size() * pad;
    float *new_data = new float[sz];
    float *p = new_data;
    typename Fdata::iterator iter = data.begin();
    while (iter != data.end())
    {
      fill_data(*iter, p);
      ++iter;
      p += pad;
    }
    return new_data;
  }
  return &(*data.begin()); // no copy just wrap this pointer
}


template <class Fld>
bool
ConvertToNrrd<Fld>::convert_to_nrrd(FieldHandle ifh,
                                    NrrdDataHandle &pointsH,
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

  TypeDescription::td_vec *tdv = 
    ifh->get_type_description(Field::FDATA_TD_E)->get_sub_type();
  const string data_name = (*tdv)[0]->get_name();

  NrrdData *ndata = 0;
  NrrdData *npoints = 0;
  NrrdData *nconnect = 0;
  int pad_data = 0; // if 0 then no copy is made 
  int sink_size = 1;
  string sink_label = label + string(":Scalar");
  int kind = nrrdKindScalar;
  if (data_name == "Tensor") {
    pad_data = 7; // copy the data, and pad for tensor values
    sink_size = 7;
    kind = nrrdKind3DMaskedSymMatrix;
    sink_label = label + string(":Tensor");
  } else if (data_name== "Vector") {
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
      {
	size_t size[NRRD_DIM_MAX];
	size[0] = 3;
	size[1] = ndims[0];
	nrrdAlloc_nva(npoints->nrrd_, nrrdTypeDouble, 2, size);
	break;
      }
    case 2:
      {
	size_t size[NRRD_DIM_MAX];
	size[0] = 3;
	size[1] = ndims[0];
	size[2] = ndims[1];
	nrrdAlloc_nva(npoints->nrrd_, nrrdTypeDouble, 3, size);
      break;
      }
    case 3:
      {
	size_t size[NRRD_DIM_MAX];
	size[0] = 3;
	size[1] = ndims[0];
	size[2] = ndims[1];
	size[3] = ndims[2];
	nrrdAlloc_nva(npoints->nrrd_, nrrdTypeDouble, 4, size);
	break;
      }
    }

    typename Fld::mesh_type::Node::iterator iter, end;
    m->begin(iter);
    m->end(end);
    double *data = (double*)npoints->nrrd_->data;
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
    if (array.size() == 1) {
      size_t size[NRRD_DIM_MAX];
      size[0] = nelems;
      nrrdAlloc_nva(nconnect->nrrd_, nrrdTypeInt, 1, size);
    }
    else {
      size_t size[NRRD_DIM_MAX];
      size[0] = array.size();
      size[1] = nelems;
      nrrdAlloc_nva(nconnect->nrrd_, nrrdTypeInt, 2, size);
    }

    int* data2 = (int*)nconnect->nrrd_->data;

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
	
	size_t size[NRRD_DIM_MAX];
	size[0] = ddims[0];
	nrrdWrap_nva(ndata->nrrd_, get_raw_data_ptr(f->fdata(), pad_data), 
		 get_nrrd_type<val_t>(), 1, size);
	
	if (f->basis_order() == 1) {
	  unsigned int centers[NRRD_DIM_MAX] = {nrrdCenterNode};
	  nrrdAxisInfoSet_nva(ndata->nrrd_, nrrdAxisInfoCenter, centers);
	} else if (f->basis_order() == 0) {
	  unsigned int centers[NRRD_DIM_MAX] = {nrrdCenterCell};
	  nrrdAxisInfoSet_nva(ndata->nrrd_, nrrdAxisInfoCenter, centers);
	} else  {
	  unsigned int centers[NRRD_DIM_MAX] = {nrrdCenterUnknown};
	  nrrdAxisInfoSet_nva(ndata->nrrd_, nrrdAxisInfoCenter, centers);
	}
	ndata->nrrd_->axis[0].label = airStrdup("x");
	
	if (with_spacing) {
	  ndata->nrrd_->axis[0].min=minP.x();
	  ndata->nrrd_->axis[0].max=maxP.x();
	  ndata->nrrd_->axis[0].spacing=spc.x();
	}
	
	ndata->nrrd_->axis[0].kind = nrrdKindDomain;
      }
      break;
    case 2:
      {
	// vector/tensor data stored as [x][3] or [x][7]
	if (pad_data > 0) {
	  size_t size[NRRD_DIM_MAX];
	  size[0] = pad_data;
	  size[1] = ddims[0];
	  nrrdWrap_nva(ndata->nrrd_, get_raw_data_ptr(f->fdata(), pad_data), 
		   get_nrrd_type<val_t>(), 2, size);
	  ndata->nrrd_->axis[0].kind = kind;
	} else {
	  size_t size[NRRD_DIM_MAX];
	  size[0] = ddims[0];
	  size[1] = ddims[1];
	  nrrdWrap_nva(ndata->nrrd_, get_raw_data_ptr(f->fdata(), pad_data), 
		   get_nrrd_type<val_t>(), 2, size);
	  ndata->nrrd_->axis[0].kind = nrrdKindDomain;
	}
	ndata->nrrd_->axis[1].kind = nrrdKindDomain;

	if (f->basis_order() == 1) {
	  unsigned int centers[NRRD_DIM_MAX];
	  centers[0] = nrrdCenterNode;
	  centers[1] = nrrdCenterNode;
	  nrrdAxisInfoSet_nva(ndata->nrrd_, nrrdAxisInfoCenter, centers);
	} else if (f->basis_order() == 0) {
	  unsigned int centers[NRRD_DIM_MAX];
	  centers[0] = nrrdCenterCell;
	  centers[1] = nrrdCenterCell;
	  nrrdAxisInfoSet_nva(ndata->nrrd_, nrrdAxisInfoCenter, centers);
	} else  {
	  unsigned int centers[NRRD_DIM_MAX];
	  centers[0] = nrrdCenterUnknown;
	  centers[1] = nrrdCenterUnknown;
	  nrrdAxisInfoSet_nva(ndata->nrrd_, nrrdAxisInfoCenter, centers);
	}

	if (pad_data > 0) {
	  // 1D nrrd with vector/tensor
	  ndata->nrrd_->axis[0].label = airStrdup(sink_label.c_str());
	  ndata->nrrd_->axis[1].label = airStrdup("x");

	  if (with_spacing) {
	    ndata->nrrd_->axis[1].min=minP.x();
	    ndata->nrrd_->axis[1].max=maxP.x();
	    ndata->nrrd_->axis[1].spacing=spc.x();
	  }
	} else {
	  // 2D nrrd of scalars
	  ndata->nrrd_->axis[0].label = airStrdup("x");
	  ndata->nrrd_->axis[1].label = airStrdup("y");

	  if (with_spacing) {
	    ndata->nrrd_->axis[0].min=minP.x();
	    ndata->nrrd_->axis[0].max=maxP.x();
	    ndata->nrrd_->axis[0].spacing=spc.x();
	    ndata->nrrd_->axis[1].min=minP.y();
	    ndata->nrrd_->axis[1].max=maxP.y();
	    ndata->nrrd_->axis[1].spacing=spc.y();
	  }
	}
      }
      break;
    case 3:
      {
	if (f->basis_order() == 1) {
	  if (pad_data > 0) {
	    // 2D nrrd with vector/tensor NODE
	    size_t size[NRRD_DIM_MAX];
	    size[0] = pad_data;
	    size[1] = ddims[0];
	    size[2] = ddims[1];
	    nrrdWrap_nva(ndata->nrrd_, get_raw_data_ptr(f->fdata(), pad_data), 
		     get_nrrd_type<val_t>(), 3, size);
	    ndata->nrrd_->axis[0].kind = kind;
	  } else {
	    // 3D nrrd of scalars NODE
	    size_t size[NRRD_DIM_MAX];
	    size[0] = ddims[0];
	    size[1] = ddims[1];
	    size[2] = ddims[2];
	    nrrdWrap_nva(ndata->nrrd_, get_raw_data_ptr(f->fdata(), pad_data), 
		     get_nrrd_type<val_t>(), 3, size);
	    ndata->nrrd_->axis[0].kind = nrrdKindDomain;
	  }
	  ndata->nrrd_->axis[1].kind = nrrdKindDomain;
	  ndata->nrrd_->axis[2].kind = nrrdKindDomain;

	  unsigned int centers[NRRD_DIM_MAX];
	  centers[0] = nrrdCenterNode; 	  
	  centers[1] = nrrdCenterNode;
	  centers[2] = nrrdCenterNode;
	  nrrdAxisInfoSet_nva(ndata->nrrd_, nrrdAxisInfoCenter, centers);
	} else if (f->basis_order() == 0) {
	  if (pad_data > 0) {
	    // 2D nrrd with vector/tensor CELL
	    size_t size[NRRD_DIM_MAX];
	    size[0] = pad_data;
	    size[1] = ddims[0];
	    size[2] = ddims[1];
	    nrrdWrap_nva(ndata->nrrd_, get_raw_data_ptr(f->fdata(), pad_data), 
		     get_nrrd_type<val_t>(), 3, size);
	  } else {
	    // 3D nrrd of scalars CELL
	    size_t size[NRRD_DIM_MAX];
	    size[0] = ddims[0];
	    size[1] = ddims[1];
	    size[2] = ddims[2];
	    nrrdWrap_nva(ndata->nrrd_, get_raw_data_ptr(f->fdata(), pad_data), 
		     get_nrrd_type<val_t>(), 3, size);
	  }
	  unsigned int centers[NRRD_DIM_MAX];
	  centers[0] = nrrdCenterCell;
	  centers[1] = nrrdCenterCell;
	  centers[2] = nrrdCenterCell;
	  nrrdAxisInfoSet_nva(ndata->nrrd_, nrrdAxisInfoCenter, centers);
	} else  {
	  ASSERTFAIL("no support for edge or face centers");
	}

	// set labels
	if (pad_data > 0) {
	  ndata->nrrd_->axis[0].label = airStrdup(sink_label.c_str());
	  ndata->nrrd_->axis[1].label = airStrdup("x");
	  ndata->nrrd_->axis[2].label = airStrdup("y");
	} else {
	  ndata->nrrd_->axis[0].label = airStrdup("x");
	  ndata->nrrd_->axis[1].label = airStrdup("y");
	  ndata->nrrd_->axis[2].label = airStrdup("z");
	}

	// set min, max, and spacing
	if (with_spacing) {
	  if (pad_data > 0) {
	    // 2D nrrd with vector/tensor
	    ndata->nrrd_->axis[1].min=minP.x();
	    ndata->nrrd_->axis[1].max=maxP.x();
	    ndata->nrrd_->axis[1].spacing=spc.x();
	    ndata->nrrd_->axis[2].min=minP.y();
	    ndata->nrrd_->axis[2].max=maxP.y();
	    ndata->nrrd_->axis[2].spacing=spc.y();
	  } else {
	    // 3D nrrd with scalars
	    ndata->nrrd_->axis[0].min=minP.x();
	    ndata->nrrd_->axis[0].max=maxP.x();
	    ndata->nrrd_->axis[0].spacing=spc.x();
	    ndata->nrrd_->axis[1].min=minP.y();
	    ndata->nrrd_->axis[1].max=maxP.y();
	    ndata->nrrd_->axis[1].spacing=spc.y();
	    ndata->nrrd_->axis[2].min=minP.z();
	    ndata->nrrd_->axis[2].max=maxP.z();
	    ndata->nrrd_->axis[2].spacing=spc.z();
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

	ndata->nrrd_->axis[0].kind = kind;
	ndata->nrrd_->axis[1].kind = nrrdKindDomain;
	ndata->nrrd_->axis[2].kind = nrrdKindDomain;
	ndata->nrrd_->axis[3].kind = nrrdKindDomain;

	size_t size[NRRD_DIM_MAX];
	size[0] = pad_data;
	size[1] = ddims[0];
	size[2] = ddims[1];
	size[3] = ddims[2];
	
	if (f->basis_order() == 1) {
	  nrrdWrap_nva(ndata->nrrd_, get_raw_data_ptr(f->fdata(), pad_data), 
		   get_nrrd_type<val_t>(), 4, size);

	  unsigned int centers[NRRD_DIM_MAX];
	  centers[0] = nrrdCenterNode; centers[1] = nrrdCenterNode;
	  centers[2] = nrrdCenterNode; centers[3] = nrrdCenterNode;
	  nrrdAxisInfoSet_nva(ndata->nrrd_, nrrdAxisInfoCenter, centers);
	} else if (f->basis_order() == 0) {
	  nrrdWrap_nva(ndata->nrrd_, get_raw_data_ptr(f->fdata(), pad_data), 
		   get_nrrd_type<val_t>(), 4, size);
	
	  unsigned int centers[NRRD_DIM_MAX];
	  centers[0] = nrrdCenterCell; centers[1] = nrrdCenterCell;
	  centers[2] = nrrdCenterCell; centers[3] = nrrdCenterCell;
	  nrrdAxisInfoSet_nva(ndata->nrrd_, nrrdAxisInfoCenter, centers);
	} else  {
	  ASSERTFAIL("no support for edge or face centers");
	}

	// set labels
	ndata->nrrd_->axis[0].label = airStrdup(sink_label.c_str());
	ndata->nrrd_->axis[1].label = airStrdup("x");
	ndata->nrrd_->axis[2].label = airStrdup("y");
	ndata->nrrd_->axis[3].label = airStrdup("z");
      
	if (with_spacing) {
	  ndata->nrrd_->axis[1].min=minP.x();
	  ndata->nrrd_->axis[1].max=maxP.x();
	  ndata->nrrd_->axis[1].spacing=spc.x();
	  ndata->nrrd_->axis[2].min=minP.y();
	  ndata->nrrd_->axis[2].max=maxP.y();
	  ndata->nrrd_->axis[2].spacing=spc.y();
	  ndata->nrrd_->axis[3].min=minP.z();
	  ndata->nrrd_->axis[3].max=maxP.z();
	  ndata->nrrd_->axis[3].spacing=spc.z();
	} 
      }
      break;
    default:
      break;
    }

    // check for transform
    const string meshstr =
      ifh->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name().substr(0, 6);
    
    if (!(ifh->mesh()->is_editable() && meshstr != "Struct"))
    {
      Transform t;
      m->get_canonical_transform(t);
      double trans[16];
      t.get(trans);
      string trans_string = "";
      for(int i=0; i<16; i++) {
	trans_string += to_string(trans[i]);
	trans_string += " ";
      }
      dataH = ndata;
      dataH->set_property("Transform", trans_string, false);
    } else {
      dataH = ndata;
    }
  }
  return true;
}

} // end namespace SCIRun

#endif // ConvertToNrrd_h
