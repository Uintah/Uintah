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
#include <StandAlone/Apps/Painter/VolumeOps.h>
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
                       NrrdDataHandle &nrrd,
                       unsigned int label) :
  lock(("NrrdVolume "+name).c_str()),
  ref_cnt(0),
  painter_(painter),
  parent_(0),
  children_(0),
  nrrd_handle_(0),
  name_(name),
  filename_(name),
  opacity_(1.0),
  clut_min_(0.0),
  clut_max_(1.0),
  data_min_(0),
  data_max_(1.0),
  label_(label),
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
  nrrd_handle_ = 0;
}



void
NrrdVolume::set_nrrd(NrrdDataHandle &nrrd_handle) 
{
  nrrd_handle_ = nrrd_handle;
  nrrd_handle_->lock.lock();
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

  nrrd_handle_->lock.unlock();
  build_index_to_world_matrix();
  dirty_ = true;
  reset_data_range();


}



void
NrrdVolume::reset_data_range() 
{
  NrrdRange range;
  nrrdRangeSet(&range, nrrd_handle_->nrrd_, 0);
  if (data_min_ != range.min || data_max_ != range.max) {
    data_min_ = range.min;
    data_max_ = range.max;
    clut_min_ = range.min;
    clut_max_ = range.max;
    opacity_ = 1.0;
  }
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
}


Vector 
NrrdVolume::index_to_vector(const vector<double> &index) {
  vector<double> zero_index(index.size(),0.0);
  return index_to_point(index) - index_to_point(zero_index);
}



void
NrrdVolume::build_index_to_world_matrix() {
  nrrd_handle_->lock.lock();
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
  nrrd_handle_->lock.unlock();
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



NrrdVolumeHandle
NrrdVolume::create_child_label_volume(unsigned int label)
{
  NrrdVolumeHandle parent = this;
  while (parent->parent_.get_rep()) parent = parent->parent_;

  if (!label) {
    const unsigned char max_bit = sizeof(unsigned int)*8;
    unsigned char bit = 0;
    unsigned int used_labels = parent->compute_label_mask();
    while (bit < max_bit && (used_labels & (1 << bit))) ++bit;
    if (bit == max_bit) {
      cerr << "Cannot create child label volume!\n";
      return 0;
    }
    label = 1 << bit;
  }

  NrrdVolumeHandle vol = 
    new NrrdVolume(painter_, 
                   parent->name_+" "+to_string(label), 
                   nrrd_handle_, label);
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

VolumeSliceHandle
NrrdVolume::get_volume_slice(const Plane &plane) {
  //  return new VolumeSlice(this, plane);
  NrrdVolumeHandle parent = this;
  VolumeSlices_t::iterator siter = parent->all_slices_.begin();
  VolumeSlices_t::iterator send = parent->all_slices_.end();  
  for (; siter != send; ++siter) {
    if ((*siter)->get_plane() == plane) {
      return (*siter);
    }
  }

  while (parent->parent_.get_rep()) parent = parent->parent_;
  if (this != parent.get_rep()) {
    VolumeSlices_t::iterator siter = parent->all_slices_.begin();
    VolumeSlices_t::iterator send = parent->all_slices_.end();  
    for (; siter != send; ++siter) {
      if ((*siter)->get_plane() == plane) {
        lock.lock();
        all_slices_.push_back
          (new VolumeSlice(this, plane, (*siter)->nrrd_handle_, label_));
        lock.unlock();
        return all_slices_.back();
        //        return (*siter);
      }
    }
  } 
  lock.lock();
  all_slices_.push_back(new VolumeSlice(this, plane,0,label_));
  lock.unlock();
  return all_slices_.back();
}

void
NrrdVolume::purge_unused_slices()
{
  NrrdVolumeHandle parent = this;
  while (parent->parent_.get_rep()) parent = parent->parent_;

  VolumeSlices_t new_all_slices;
  if (!dirty_) {
    for (unsigned int j = 0; j < parent->all_slices_.size(); ++j) {      
      if (parent->all_slices_[j]->ref_cnt > 1) {
        new_all_slices.push_back(parent->all_slices_[j]);
      }
    }
  }
  dirty_ = false;
  parent->all_slices_ = new_all_slices;
}


ColorMapHandle
NrrdVolume::get_colormap() {
  if (!colormap_ && label_) 
    return ColorMap::create_rainbow(20.0);

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


NrrdDataHandle
NrrdVolume::extract_label_as_bit() 
{
  Nrrd *n = nrrd_handle_->nrrd_;
  if (!label_ || !n || n->type != nrrdTypeUInt) return 0;

  NrrdData *newnrrdh = new NrrdData();
  nrrdCopy(newnrrdh->nrrd_, n);
  unsigned int *data = (unsigned int *)newnrrdh->nrrd_->data;
  ASSERT(data);
  unsigned int count = 1;
  for(unsigned int i=0; i < n->dim-1; i++) {
    count *= n->axis[i+1].size;
  }
  
  unsigned int bit = 0;
  while (!(label_ & (1 << bit))) bit++;
  
  for (unsigned int i = 0; i < count; ++i) {
    data[i] = (data[i] >> bit) & 1;
  }

  return newnrrdh;
}

int
NrrdVolume::bit() {
  if (!label_) return -1;
  int lbit = 0;
  while (!(label_ & (1 << lbit))) ++lbit;
  return lbit;
}
    


void
NrrdVolume::change_type_from_float_to_bit() {
  NrrdDataHandle nrrdh = VolumeOps::float_to_bit(nrrd_handle_, 0, label_);
  set_nrrd(nrrdh);  
}

void
NrrdVolume::change_type_from_bit_to_float(float val) {
  nrrd_handle_ = VolumeOps::bit_to_float(nrrd_handle_, label_, val);
  dirty_ = true;
}

void
NrrdVolume::clear() {
  VolumeOps::clear_nrrd(nrrd_handle_);
}
  
 
int
NrrdVolume::numbytes() {
  return VolumeOps::nrrd_data_size(nrrd_handle_);
}




}
