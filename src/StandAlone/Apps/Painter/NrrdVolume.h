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
//    File   : NrrdVolume.h
//    Author : McKay Davis
//    Date   : Fri Oct 13 16:03:50 2006


#ifndef LEXOV_NrrdVolume
#define LEXOV_NrrdVolume

#include <vector>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/NrrdData.h>
#include <Core/Skinner/Variables.h>
#include <StandAlone/Apps/Painter/NrrdToITK.h>
#include <StandAlone/Apps/Painter/VolumeSlice.h>
#include <StandAlone/Apps/Painter/VolumeOps.h>
#include <sci_defs/insight_defs.h>
#include <Core/Geom/IndexedGroup.h>

using std::vector;


namespace SCIRun {

class Painter;
class VolumeSlice;
class NrrdVolume;
typedef LockingHandle<NrrdVolume> NrrdVolumeHandle;
typedef vector<NrrdVolumeHandle>	NrrdVolumes;

class NrrdVolume { 
public:
  // For LockingHandle<NrrdVolume> NrrdVolumeHandle typedef after this class
  Mutex               lock;
  unsigned int        ref_cnt;


  // Constructor
  NrrdVolume	      (Painter *painter, 
                       const string &name,
                       NrrdDataHandle &nrrdh,
                       const unsigned int label = 0);

  ~NrrdVolume();


#ifdef HAVE_INSIGHT
  ITKDatatypeHandle   get_itk_image() 
  { return nrrd_to_itk_image(nrrd_handle_); }
#endif
    
  bool                write(string filename);

  void                set_nrrd(NrrdDataHandle &);
  NrrdDataHandle      get_nrrd();
  void                set_dirty(bool d = true) { dirty_ = d; }

  NrrdVolumeHandle    create_label_volume(unsigned int label=1, 
                                          NrrdDataHandle nrrdh = 0);
  NrrdVolumeHandle    create_child_label_volume(unsigned int label=0);
  unsigned int        compute_label_mask(unsigned int label = 0);

  // Generates a VolumeSlice class if Plane intersects the volume,
  // Returns 0 if the Plane does not intersect the volume
  VolumeSliceHandle   get_volume_slice(const Plane &);

  void                reset_data_range();

  // Methods to transform between index and world space
  void                build_index_to_world_matrix();
  Point               index_to_world(const vector<int> &index);
  Point               index_to_point(const vector<double> &index);
  vector<int>         world_to_index(const Point &p);
  vector<double>      point_to_index(const Point &p);
  vector<double>      vector_to_index(const Vector &v);
  Vector              index_to_vector(const vector<double> &);
  bool                index_valid(const vector<int> &index);
  Point               center(int axis = -1, int slice = -1);
  Point               min(int axis = -1, int slice = -1);
  Point               max(int axis = -1, int slice = -1);


  // Voxel getter/setters
  template<class T>
  void                get_value(const vector<int> &index, T &value);
  template<class T>
  void                set_value(const vector<int> &index, T value);


  Vector              scale();
  double              scale(unsigned int axis);

  vector<int>         max_index();
  int                 max_index(unsigned int axis);

  bool                inside_p(const Point &p);
  ColorMapHandle      get_colormap();
  GeomIndexedGroup*   get_geom_group();
  NrrdDataHandle      extract_label_as_bit();

  void                change_type_from_float_to_bit();
  void                change_type_from_bit_to_float();
  void                clear();
  int                 numbytes();
  int                 bit();
  


  Painter *           painter_;
  NrrdVolumeHandle    parent_;
  NrrdVolumes         children_;
  NrrdDataHandle      nrrd_handle_;
  string              name_;
  string              filename_;

  double	      opacity_;
  double              clut_min_;
  double              clut_max_;
  double              data_min_;
  double              data_max_;

  unsigned int        label_;
  int                 colormap_;
  vector<int>         stub_axes_;
  DenseMatrix         transform_;
  bool                keep_;
  Skinner::Var<bool>  visible_;
  bool                expand_;
  GeomHandle          geom_switch_;
  GeomHandle          geom_group_;


  // This next function must be called inside an opengl context, becuase
  // it may call a geometry class desctructor that is deletes GL elements
  // that are bound to a gl context, and thus the context must be current
  void                purge_unused_slices();
  bool                dirty_;
  VolumeSlices_t      all_slices_;
  //  VolumeSliceGroups_t slice_groups_;
};





template<class T>
void
NrrdVolume::get_value(const vector<int> &index, T &value) 
{
  ASSERT(index_valid(index));
  VolumeOps::nrrd_get_value(nrrd_handle_->nrrd_, index, value);
}


template <class T>
void
NrrdVolume::set_value(const vector<int> &index, T value) {
  ASSERT(index_valid(index));
  VolumeOps::nrrd_set_value(nrrd_handle_->nrrd_, index, value);
}








}


#endif
