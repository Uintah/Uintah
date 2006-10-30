//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : NrrdVolume.cc
//    Author : McKay Davis
//    Date   : Fri Oct 13 15:06:57 2006

#include <StandAlone/Apps/Painter/NrrdVolume.h>
#include <StandAlone/Apps/Painter/Painter.h>
#include <sci_comp_warn_fixes.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <typeinfo>
#include <iostream>
#include <sci_gl.h>
#include <sci_algorithm.h>
#include <Core/Datatypes/Field.h> 
#include <Core/Exceptions/GuiException.h>
#include <Core/Geom/OpenGLViewport.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/Environment.h>
#include <Core/Geom/FontManager.h>
#include <Core/Skinner/Variables.h>
#include <Core/Events/EventManager.h>
#include <Core/Events/SceneGraphEvent.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Skinner/GeomSkinnerVarSwitch.h>

namespace SCIRun {


NrrdVolume::NrrdVolume(Painter *painter,
                       const string &name,
                       NrrdDataHandle &nrrd) :
  painter_(painter),
  parent_(0),
  children_(0),
  nrrd_handle_(0),
  name_(name),
  filename_(name),
  mutex_(new Mutex(name.c_str())),
  opacity_(1.0),
  clut_min_(0.0),
  clut_max_(1.0),
  data_min_(0),
  data_max_(1.0),
  label_(0),
  colormap_(0),
  stub_axes_(),
  transform_(),
  keep_(true),
  visible_(painter->get_vars(), "FOO", true),
  expand_(true)
{
  set_nrrd(nrrd);
}



NrrdVolume::~NrrdVolume() {
  mutex_->lock();
  nrrd_handle_ = 0;
  mutex_->unlock();

}



int
nrrd_type_size(Nrrd *nrrd)
{
  int val = 0;
  switch (nrrd->type) {
  case nrrdTypeChar: val = sizeof(char); break;
  case nrrdTypeUChar: val = sizeof(unsigned char); break;
  case nrrdTypeShort: val = sizeof(short); break;
  case nrrdTypeUShort: val = sizeof(unsigned short); break;
  case nrrdTypeInt: val = sizeof(int); break;
  case nrrdTypeUInt: val = sizeof(unsigned int); break;
  case nrrdTypeLLong: val = sizeof(signed long long); break;
  case nrrdTypeULLong: val = sizeof(unsigned long long); break;
  case nrrdTypeFloat: val = sizeof(float); break;
  case nrrdTypeDouble: val = sizeof(double); break;
  default: throw "Unsupported data type: "+to_string(nrrd->type);
  }
  return val;
}


int
nrrd_size(Nrrd *nrrd)
{
  if (!nrrd->dim) return 0;
  unsigned int size = nrrd->axis[0].size;
  for (unsigned int a = 1; a < nrrd->dim; ++a)
    size *= nrrd->axis[a].size;
  return size;
}


int
nrrd_data_size(Nrrd *nrrd)
{
  return nrrd_size(nrrd) * nrrd_type_size(nrrd);
}



NrrdVolume::NrrdVolume(NrrdVolume *copy, 
                       const string &name,
                       int clear) :
  painter_(copy->painter_),
  parent_(0),
  children_(0),
  nrrd_handle_(0),
  name_(name),
  filename_(name),
  mutex_(new Mutex(name.c_str())),
  opacity_(copy->opacity_),
  clut_min_(copy->clut_min_),
  clut_max_(copy->clut_max_),
  data_min_(copy->data_min_),
  data_max_(copy->data_max_),
  colormap_(copy->colormap_),
  stub_axes_(copy->stub_axes_),
  transform_(),
  keep_(copy->keep_),
  visible_(),
  expand_(copy->expand_)
{
  visible_ = copy->visible_;
  copy->mutex_->lock();
  mutex_->lock();

  ASSERT(clear >= 0 && clear <= 2);
  
  switch (clear) {
  case 0: {
    nrrd_handle_ = scinew NrrdData();
    nrrdCopy(nrrd_handle_->nrrd_, copy->nrrd_handle_->nrrd_);
  } break;
  case 1: {
    nrrd_handle_ = scinew NrrdData();
    nrrdCopy(nrrd_handle_->nrrd_, copy->nrrd_handle_->nrrd_);
    memset(nrrd_handle_->nrrd_->data, 0, nrrd_data_size(nrrd_handle_->nrrd_));
  } break;
  default:
  case 2: {
    nrrd_handle_ = copy->nrrd_handle_;
  } break;
  }

  mutex_->unlock();
  //  set_nrrd(nrrd_);
  build_index_to_world_matrix();
  copy->mutex_->unlock();


}



void
NrrdVolume::set_nrrd(NrrdDataHandle &nrrd_handle) 
{
  mutex_->lock();
  nrrd_handle_ = nrrd_handle;
  //  nrrd_handle_.detach();
  //  nrrdBasicInfoCopy(nrrd_handle_->nrrd_, nrrd->nrrd,0);
  //  nrrdAxisInfoCopy(nrrd_handle_->nrrd_, nrrd->nrrd, 0,0);
  //  nrrdCopy(nrrd_handle_->nrrd_, nrrd->nrrd);
  Nrrd *n = nrrd_handle_->nrrd_;

  stub_axes_.clear();
  if (n->axis[0].size > 4) {
    nrrdAxesInsert(n, n, 0);
    n->axis[0].min = 0.0;
    n->axis[0].max = 1.0;
    n->axis[0].spacing = 1.0;
    stub_axes_.push_back(0);
  }

  if (n->dim == 3) {
    nrrdAxesInsert(n, n, 3);
    n->axis[3].min = 0.0;
    n->axis[3].max = 1.0;
    n->axis[3].spacing = 1.0;
    stub_axes_.push_back(3);
  }


  for (unsigned int a = 0; a < n->dim; ++a) {
    if (n->axis[a].center == nrrdCenterUnknown)
      n->axis[a].center = nrrdCenterNode;

#if 0
    if (airIsNaN(n->axis[a].min) && 
        airIsNaN(n->axis[a].max)) 
    {
      if (airIsNaN(n->axis[a].spacing)) {
        n->axis[0].spacing = 1.0;
      }
#endif

    if (n->axis[a].min > n->axis[a].max)
      SWAP(n->axis[a].min,n->axis[a].max);
    if (n->axis[a].spacing < 0.0)
      n->axis[a].spacing *= -1.0;
  }
#if 0
  NrrdRange range;
  nrrdRangeSet(&range, n, 0);
  if (data_min_ != range.min || data_max_ != range.max) {
    data_min_ = range.min;
    data_max_ = range.max;
    clut_min_ = range.min;
    clut_max_ = range.max;
    opacity_ = 1.0;
  }
#endif
  build_index_to_world_matrix();
  mutex_->unlock();
  reset_data_range();


}



void
NrrdVolume::reset_data_range() 
{
  mutex_->lock();
  NrrdRange range;
  nrrdRangeSet(&range, nrrd_handle_->nrrd_, 0);
  if (data_min_ != range.min || data_max_ != range.max) {
    data_min_ = range.min;
    data_max_ = range.max;
    clut_min_ = range.min;
    clut_max_ = range.max;
    opacity_ = 1.0;
  }
  mutex_->unlock();
}
  

NrrdDataHandle
NrrdVolume::get_nrrd() 
{
  NrrdDataHandle nrrd_handle = nrrd_handle_;
  nrrd_handle.detach();
  NrrdDataHandle nrrd2_handle = scinew NrrdData();

  //   nrrdBasicInfoCopy(nrrd->nrrd, nrrd_handle_->nrrd_,0);
  //   nrrdAxisInfoCopy(nrrd->nrrd, nrrd_handle_->nrrd_, 0,0);
  //   nrrd->nrrd->data = nrrd_handle_->nrrd_->data;

  for (int s = stub_axes_.size()-1; s >= 0 ; --s) {
    nrrdAxesDelete(nrrd2_handle->nrrd_, nrrd_handle->nrrd_, stub_axes_[s]);
    nrrd_handle = nrrd2_handle;
  }
  nrrdKeyValueCopy(nrrd_handle->nrrd_, nrrd_handle_->nrrd_);
  
  //  unsigned long ptr = (unsigned long)(&painter_);
  //  nrrdKeyValueAdd(nrrd_handle->nrrd_, 
  //                  "progress_ptr", to_string(ptr).c_str());

  return nrrd_handle;
}



Point
NrrdVolume::center(int axis, int slice) {
  vector<int> index(nrrd_handle_->nrrd_->dim,0);
  for (unsigned int a = 0; a < index.size(); ++a) 
    index[a] = nrrd_handle_->nrrd_->axis[a].size/2;
  if (axis >= 0 && axis < int(index.size()))
    index[axis] = Clamp(slice, 0, nrrd_handle_->nrrd_->axis[axis].size-1);
  ASSERT(index_valid(index));
  return index_to_world(index);
}


Point
NrrdVolume::min(int axis, int slice) {
  vector<int> index(nrrd_handle_->nrrd_->dim,0);
  if (axis >= 0 && axis < int(index.size()))
    index[axis] = Clamp(slice, 0, nrrd_handle_->nrrd_->axis[axis].size-1);
  ASSERT(index_valid(index));
  return index_to_world(index);
}

Point
NrrdVolume::max(int axis, int slice) {
  vector<int> index = max_index();
  if (axis >= 0 && axis < int(index.size()))
    index[axis] = Clamp(slice, 0, nrrd_handle_->nrrd_->axis[axis].size-1);
  ASSERT(index_valid(index));
  return index_to_world(index);
}



Vector
NrrdVolume::scale() {
  vector<int> index_zero(nrrd_handle_->nrrd_->dim,0);
  vector<int> index_one(nrrd_handle_->nrrd_->dim,1);
  return index_to_world(index_one) - index_to_world(index_zero);
}


double
NrrdVolume::scale(unsigned int axis) {
  ASSERT(axis >= 0 && (unsigned int) axis < nrrd_handle_->nrrd_->dim);
  return scale()[axis];
}



vector<int>
NrrdVolume::max_index() {
  vector<int> max_index(nrrd_handle_->nrrd_->dim,0);
  for (unsigned int a = 0; a < nrrd_handle_->nrrd_->dim; ++a)
    max_index[a] = nrrd_handle_->nrrd_->axis[a].size;
  return max_index;
}

int
NrrdVolume::max_index(unsigned int axis) {
  ASSERT(axis >= 0 && (unsigned int) axis < nrrd_handle_->nrrd_->dim);
  return max_index()[axis];
}

bool
NrrdVolume::inside_p(const Point &p) {
  return index_valid(world_to_index(p));
}



Point
NrrdVolume::index_to_world(const vector<int> &index) {
  unsigned int dim = index.size()+1;
  ColumnMatrix index_matrix(dim);
  ColumnMatrix world_coords(dim);
  for (unsigned int i = 0; i < dim-1; ++i)
    index_matrix[i] = index[i];
  index_matrix[dim-1] = 1.0;
  DenseMatrix transform = transform_;
  int tmp1, tmp2;
  transform.mult(index_matrix, world_coords, tmp1, tmp2);
  Point return_val;
  for (int i = 1; i < 4; ++i) 
    return_val(i-1) = world_coords[i];
  return return_val;
}


Point
NrrdVolume::index_to_point(const vector<double> &index) {
  unsigned int dim = index.size()+1;
  ColumnMatrix index_matrix(dim);
  ColumnMatrix world_coords(dim);
  for (unsigned int i = 0; i < dim-1; ++i)
    index_matrix[i] = index[i];
  index_matrix[dim-1] = 1.0;
  DenseMatrix transform = transform_;
  int tmp1, tmp2;
  transform.mult(index_matrix, world_coords, tmp1, tmp2);
  Point return_val;
  for (int i = 1; i < 4; ++i) 
    return_val(i-1) = world_coords[i];
  return return_val;
}


vector<int> 
NrrdVolume::world_to_index(const Point &p) {
  DenseMatrix transform = transform_;
  ColumnMatrix index_matrix(transform.ncols());
  ColumnMatrix world_coords(transform.nrows());
  for (int i = 0; i < transform.nrows(); ++i)
    if (i > 0 && i < 4) 
      world_coords[i] = p(i-1)-transform.get(i,transform.ncols()-1);
    else       
      world_coords[i] = 0.0;;
  transform.solve(world_coords, index_matrix, 1);
  vector<int> return_val(index_matrix.nrows()-1);
  for (unsigned int i = 0; i < return_val.size(); ++i) {
    return_val[i] = Floor(index_matrix[i]);
  }
  return return_val;
}

vector<double> 
NrrdVolume::point_to_index(const Point &p) {
  DenseMatrix transform = transform_;
  ColumnMatrix index_matrix(transform.ncols());
  ColumnMatrix world_coords(transform.nrows());
  for (int i = 0; i < transform.nrows(); ++i)
    if (i > 0 && i < 4) 
      world_coords[i] = p(i-1)-transform.get(i,transform.ncols()-1);
    else       
      world_coords[i] = 0.0;;
  transform.solve(world_coords, index_matrix, 1);
  vector<double> return_val(index_matrix.nrows()-1);
  for (unsigned int i = 0; i < return_val.size(); ++i) {
    return_val[i] = index_matrix[i];
  }
  return return_val;
}




vector<double> 
NrrdVolume::vector_to_index(const Vector &v) {
  Point zero(0,0,0);
  vector<double> zero_idx = point_to_index(zero);
  vector<double> idx = point_to_index(v.asPoint());
  for (unsigned int i = 0; i < zero_idx.size(); ++i) 
    idx[i] = idx[i] - zero_idx[i];
  return idx;
    
//   DenseMatrix transform = transform_;
//   ColumnMatrix index_matrix(transform.ncols());
//   ColumnMatrix world_coords(transform.nrows());
//   for (int i = 0; i < transform.nrows(); ++i)
//     if (i > 0 && i < 4) 
//       world_coords[i] = v[i-1];
//     else       
//       world_coords[i] = 0.0;;
//   int tmp, tmp2;
//   transform.mult_transpose(world_coords, index_matrix, tmp, tmp2);
//   vector<double> return_val(index_matrix.nrows()-1);
//   for (unsigned int i = 0; i < return_val.size(); ++i)
//     return_val[i] = index_matrix[i];
//   return return_val;
}


Vector 
NrrdVolume::index_to_vector(const vector<double> &index) {
  vector<double> zero_index(index.size(),0.0);
  return index_to_point(index) - index_to_point(zero_index);
}



void
NrrdVolume::build_index_to_world_matrix() {
  Nrrd *nrrd = nrrd_handle_->nrrd_;
  int dim = nrrd->dim+1;
  DenseMatrix matrix(dim, dim);
  matrix.zero();
  for (int i = 0; i < dim-1; ++i) {
    if (airExists(nrrd->axis[i].spacing)) {
      if (nrrd->axis[i].spacing == 0.0) {
        nrrd->axis[i].spacing = 1.0;
      }
      matrix.put(i,i,nrrd->axis[i].spacing);
    } else if (airExists(nrrd->axis[i].min) && airExists(nrrd->axis[i].max)) {
      if (nrrd->axis[i].min == nrrd->axis[i].max) {
        nrrd->axis[i].spacing = 1.0;
        matrix.put(i,i,1.0);
      } else {
        matrix.put(i,i,((nrrd->axis[i].max-nrrd->axis[i].min)/
                        nrrd->axis[i].size));
      }
    } else {
      matrix.put(i,i, 1.0);
    }

    if (airExists(nrrd->axis[i].min))
      matrix.put(i, nrrd->dim, nrrd->axis[i].min);
  }

  if (nrrd->axis[0].size != 1) {
    matrix.put(2,nrrd->dim, nrrd->axis[2].min+nrrd->axis[2].size*matrix.get(2,2));
    matrix.put(2,2,-matrix.get(2,2));
  }


  matrix.put(dim-1, dim-1, 1.0);
    
  transform_ = matrix;
}

bool
NrrdVolume::index_valid(const vector<int> &index) {
  unsigned int dim = nrrd_handle_->nrrd_->dim;
  if (index.size() != dim) return false;
  for (unsigned int a = 0; a < dim; ++a) 
    if (index[a] < 0 ||
	(unsigned int) index[a] >= nrrd_handle_->nrrd_->axis[a].size) {
      return false;
    }
  return true;
}


NrrdVolume *
NrrdVolume::create_label_volume()
{
  NrrdDataHandle nrrd_handle = new NrrdData();
  nrrdBasicInfoCopy(nrrd_handle->nrrd_, nrrd_handle_->nrrd_, 0);
  nrrdAxisInfoCopy(nrrd_handle->nrrd_, nrrd_handle_->nrrd_, 0, 0);
  nrrd_handle->nrrd_->type = nrrdTypeUInt;

  nrrd_handle->nrrd_->data = new unsigned int[nrrd_size(nrrd_handle_->nrrd_)];
  ASSERT(nrrd_handle->nrrd_->data);

  memset(nrrd_handle->nrrd_->data, 0, nrrd_data_size(nrrd_handle->nrrd_));
  NrrdVolume *vol = new NrrdVolume(painter_, name_ + " Label", nrrd_handle);
  vol->label_ = 1;
  delete vol->mutex_;
  vol->mutex_ = mutex_;
  return vol;
}


unsigned int
NrrdVolume::compute_label_mask(unsigned int label)
{
  ASSERT((label & label_) == 0);
  label |= label_;
  for (unsigned int i = 0; i < children_.size(); ++i) {
    label |= children_[i]->compute_label_mask(label);
  }
  return label;
}


NrrdVolume *
NrrdVolume::create_child_label_volume(unsigned int label)
{
  NrrdVolume *anchor_volume = this;
  while (anchor_volume->parent_) anchor_volume = anchor_volume->parent_;

  if (!label) {
    const unsigned char max_bit = sizeof(unsigned int)*8;
    unsigned char bit = 0;
    unsigned int used_labels = anchor_volume->compute_label_mask();
    while (bit < max_bit && (used_labels & (1 << bit))) ++bit;
    if (bit == max_bit) {
      cerr << "Cannot create child label volume!\n";
      return 0;
    }
    label = 1 << bit;
  }

  NrrdVolume *vol = 
    new NrrdVolume(painter_, 
                   anchor_volume->name_+" "+to_string(label), 
                   nrrd_handle_);
  delete vol->mutex_;
  vol->mutex_ = mutex_;
  vol->label_ = label;
  vol->parent_ = this;
  children_.push_back(vol);
  return vol;
}



#ifdef HAVE_INSIGHT
template <class T>
bool
write_itk_image(ITKDatatypeHandle &itk_image_h, const string &filename)
{
  // create a new writer
  typedef itk::Image < T, 3 > ImageType;
  typedef itk::ImageFileWriter< ImageType > FileWriterType; 
  typename FileWriterType::Pointer writer = FileWriterType::New();
  ImageType *img = 
    dynamic_cast<ImageType *>(itk_image_h->data_.GetPointer());
  ASSERT(img);

  // set writer
  writer->SetFileName( filename.c_str() );
  writer->SetInput(img);
  
  try {
    writer->Update();  
  } catch  ( itk::ExceptionObject & err ) {
    cerr << "NrrdVolume::write tik::ExceptionObject caught" << std::endl;
    cerr << err.GetDescription() << std::endl;
    return false;
  }

  return true;
}
#endif

bool
NrrdVolume::write(string fname) {
#ifndef HAVE_INSIGHT
  return false;
#else
  fname = substituteTilde(fname);
  
  ITKDatatypeHandle img = get_itk_image();
  
  switch (nrrd_handle_->nrrd_->type) {
  case nrrdTypeUInt: return write_itk_image<unsigned int>(img, fname); break;
  default:
  case nrrdTypeFloat: return write_itk_image<float>(img, fname); break;
  }
  return false;
#endif
}

#if 0
NrrdHandle
NrrdVolume::create_float_nrrd_from_label()
{

  NrrdVolume *anchor_volume = this;
  while (anchor_volume->parent_) anchor_volume = anchor_volume->parent_;

  const unsigned char max_bit = sizeof(unsigned int)*8;
  unsigned char bit = 0;
  unsigned int used_labels = anchor_volume->compute_label_mask();
  while (bit < max_bit && (used_labels & (1 << bit))) ++bit;
  if (bit == max_bit) {
    cerr << "Cannot create child label volume!\n";
    return 0;
  }

  unsigned int label = 1 << bit;

  NrrdVolume *vol = 
    new NrrdVolume(painter_, 
                   anchor_volume->name_+" "+to_string(label), 
                   nrrd_handle_);
  delete vol->mutex_;
  vol->mutex_ = mutex_;
  vol->label_ = label;
  vol->parent_ = this;
  children_.push_back(vol);
  return vol;
}

#endif


VolumeSliceHandle
NrrdVolume::get_volume_slice(const Plane &plane) {
  NrrdVolume *parent = this;
  VolumeSlices_t::iterator siter = parent->all_slices_.begin();
  VolumeSlices_t::iterator send = parent->all_slices_.end();  
  for (; siter != send; ++siter) {
    if ((*siter)->get_plane() == plane) {
      return (*siter);
    }
  }

  while (parent->parent_) parent = parent->parent_;
  if (this != parent) {
    VolumeSlices_t::iterator siter = parent->all_slices_.begin();
    VolumeSlices_t::iterator send = parent->all_slices_.end();  
    for (; siter != send; ++siter) {
      if ((*siter)->get_plane() == plane) {
        all_slices_.push_back(new VolumeSlice(this, plane, (*siter)->nrrd_handle_));
        return all_slices_.back();
        //        return (*siter);
      }
    }
  } 
  all_slices_.push_back(new VolumeSlice(this, plane));
  return all_slices_.back();
}

void
NrrdVolume::purge_unused_slices()
{
  NrrdVolume *parent = this;
  while (parent->parent_) parent = parent->parent_;

  VolumeSlices_t new_all_slices;
  for (unsigned int j = 0; j < parent->all_slices_.size(); ++j) {      
    if (parent->all_slices_[j]->ref_cnt > 1) {
      new_all_slices.push_back(parent->all_slices_[j]);
    }
  }
  parent->all_slices_ = new_all_slices;
}


ColorMapHandle
NrrdVolume::get_colormap() {
  if (!colormap_ && label_) 
    return ColorMap::create_pseudo_random(5);

  return painter_->get_colormap(colormap_);
}    


GeomIndexedGroup *
NrrdVolume::get_geom_group() 
{
  if (!geom_group_.get_rep()) {
    geom_group_ = new GeomIndexedGroup();
    geom_switch_ = new GeomSkinnerVarSwitch(geom_group_, visible_);
    event_handle_t add_geom_switch_event = 
      new SceneGraphEvent(geom_switch_, name_);
    EventManager::add_event(add_geom_switch_event);
  }
  GeomIndexedGroup *ret_val = 
    dynamic_cast<GeomIndexedGroup *>(geom_group_.get_rep());

  return ret_val;
}


}
