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
#include <sci_defs/insight_defs.h>
#include <Core/Geom/IndexedGroup.h>

using std::vector;


namespace SCIRun {

class Painter;
class VolumeSlice;

class NrrdVolume { 
public:
  // Constructor
  NrrdVolume	      (Painter *painter, 
                       const string &name,
                       NrrdDataHandle &);
  // Copy Constructor
  NrrdVolume          (NrrdVolume *copy, 
                       const string &name, 
                       int mode = 0); // if 1, clears out volume
  ~NrrdVolume();


#ifdef HAVE_INSIGHT
  ITKDatatypeHandle   get_itk_image() 
  { return nrrd_to_itk_image(nrrd_handle_); }
#endif
    
  bool                write(string filename);

  void                set_nrrd(NrrdDataHandle &);
  NrrdDataHandle      get_nrrd();
  NrrdVolume *        create_label_volume();
  NrrdVolume *        create_child_label_volume(unsigned int label=0);
  unsigned int        compute_label_mask(unsigned int label = 0);

  // Generates a VolumeSlice class if Plane intersects the volume,
  // Returns 0 if the Plane does not intersect the volume
  VolumeSliceHandle   get_volume_slice(const Plane &);

  void                reset_data_range();
  Point               index_to_world(const vector<int> &index);
  Point               index_to_point(const vector<double> &index);
  vector<int>         world_to_index(const Point &p);
  vector<double>      point_to_index(const Point &p);
  vector<double>      vector_to_index(const Vector &v);
  Vector              index_to_vector(const vector<double> &);
  void                build_index_to_world_matrix();
  bool                index_valid(const vector<int> &index);
  template<class T>
  void                get_value(const vector<int> &index, T &value);
  template<class T>
  void                set_value(const vector<int> &index, T value);

  Point               center(int axis = -1, int slice = -1);
  Point               min(int axis = -1, int slice = -1);
  Point               max(int axis = -1, int slice = -1);

  Vector              scale();
  double              scale(unsigned int axis);

  vector<int>         max_index();
  int                 max_index(unsigned int axis);

  bool                inside_p(const Point &p);
  ColorMapHandle      get_colormap();
  GeomIndexedGroup*   get_geom_group();
  NrrdDataHandle      extract_label_as_bit();
  NrrdDataHandle      extract_bit_as_float(float value=1.0);


  Painter *           painter_;
  NrrdVolume *        parent_;
  vector<NrrdVolume*> children_;
  NrrdDataHandle      nrrd_handle_;

  string              name_;
  string              filename_;
  Mutex *             mutex_;


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
  VolumeSlices_t      all_slices_;
  //  VolumeSliceGroups_t slice_groups_;
};

typedef vector<NrrdVolume *>		NrrdVolumes;
typedef map<string, NrrdVolume *>	NrrdVolumeMap;
typedef list<string>                    NrrdVolumeOrder;



template<class T>
static void
nrrd_get_value(const Nrrd *nrrd, 
               const vector<int> &index, 
               T &value)
{
  ASSERT((unsigned int)(index.size()) == nrrd->dim);
  int position = index[0];
  int mult_factor = nrrd->axis[0].size;
  for (unsigned int a = 1; a < nrrd->dim; ++a) {
    position += index[a] * mult_factor;
    mult_factor *= nrrd->axis[a].size;
  }

  switch (nrrd->type) {
  case nrrdTypeChar: {
    char *slicedata = (char *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeUChar: {
    unsigned char *slicedata = (unsigned char *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeShort: {
    short *slicedata = (short *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeUShort: {
    unsigned short *slicedata = (unsigned short *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeInt: {
    int *slicedata = (int *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeUInt: {
    unsigned int *slicedata = (unsigned int *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeLLong: {
    signed long long *slicedata = (signed long long *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeULLong: {
    unsigned long long *slicedata = (unsigned long long *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeFloat: {
    float *slicedata = (float *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  case nrrdTypeDouble: {
    double *slicedata = (double *)nrrd->data;
    value = (T)slicedata[position];
  } break;
  default: {
    throw "Unsupported data type: "+to_string(nrrd->type);
  } break;
  }
}



template <class T>
static void
nrrd_set_value(Nrrd *nrrd, 
               const vector<int> &index, 
               T val, 
               unsigned int label_mask = 0)
{
  ASSERT((unsigned int)(index.size()) == nrrd->dim);
  int position = index[0];
  int mult_factor = nrrd->axis[0].size;
  for (unsigned int a = 1; a < nrrd->dim; ++a) {
    position += index[a] * mult_factor;
    mult_factor *= nrrd->axis[a].size;
  }

  switch (nrrd->type) {
  case nrrdTypeChar: {
    char *slicedata = (char *)nrrd->data;
    slicedata[position] = (char)val;
  } break;
  case nrrdTypeUChar: {
    unsigned char *slicedata = (unsigned char *)nrrd->data;
    slicedata[position] = (unsigned char)val;
    } break;
  case nrrdTypeShort: {
    short *slicedata = (short *)nrrd->data;

    slicedata[position] = (short)val;
    } break;
  case nrrdTypeUShort: {
    unsigned short *slicedata = (unsigned short *)nrrd->data;
    slicedata[position] = (unsigned short)val;
    } break;
  case nrrdTypeInt: {
    int *slicedata = (int *)nrrd->data;
    slicedata[position] = (int)val;
    } break;
  case nrrdTypeUInt: {
    unsigned int *slicedata = (unsigned int *)nrrd->data;
    if (slicedata[position] == label_mask) 
      slicedata[position] = (unsigned int)val;
    } break;
  case nrrdTypeLLong: {
    signed long long *slicedata = (signed long long *)nrrd->data;
    slicedata[position] = (signed long long)val;
    } break;
  case nrrdTypeULLong: {
    unsigned long long *slicedata = (unsigned long long *)nrrd->data;
    slicedata[position] = (unsigned long long)val;
    } break;
  case nrrdTypeFloat: {
    float *slicedata = (float *)nrrd->data;
    slicedata[position] = (float)val;
    } break;
  case nrrdTypeDouble: {
    double *slicedata = (double *)nrrd->data;
    slicedata[position] = (double)val;
    } break;
  default: { 
    throw "Unsupported data type: "+to_string(nrrd->type);
    } break;
  }
}


template<class T>
void
NrrdVolume::get_value(const vector<int> &index, T &value) {
  ASSERT(index_valid(index));
  nrrd_get_value(nrrd_handle_->nrrd_, index, value);
}


template <class T>
void
NrrdVolume::set_value(const vector<int> &index, T value) {
  ASSERT(index_valid(index));
  nrrd_set_value(nrrd_handle_->nrrd_, index, value);
}








}


#endif
