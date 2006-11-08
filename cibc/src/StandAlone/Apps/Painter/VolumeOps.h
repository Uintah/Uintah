//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
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
//    File   : VolumeOps.h
//    Author : McKay Davis
//    Date   : Tue Oct 31 17:41:48 2006 (spooky!)

#ifndef LeXoV_VolumeOps_H
#define LeXoV_VolumeOps_H

#include <Core/Datatypes/NrrdData.h>
#include <Core/Util/Assert.h>
#include <Core/Containers/StringUtil.h>

namespace SCIRun {

class VolumeOps {
public:
  static unsigned int           nrrd_elem_count(NrrdDataHandle &);
  static unsigned int           nrrd_data_size(NrrdDataHandle &);
  static unsigned int           nrrd_type_size(NrrdDataHandle &);

  static NrrdDataHandle         float_to_bit(NrrdDataHandle &,
                                             float ref = 0.0,
                                             unsigned int mask = 0);


  static NrrdDataHandle         bit_to_float(NrrdDataHandle &,
                                             unsigned int mask = 0,
                                             float ref = 0.0);
                                            


  static NrrdDataHandle         clear_nrrd(NrrdDataHandle &);
  static NrrdDataHandle         create_clear_nrrd(NrrdDataHandle &, 
                                                  unsigned int type = 0);
  static NrrdDataHandle         create_nrrd(NrrdDataHandle &, 
                                            unsigned int type = 0);
  template <class T>
  static
  void                          nrrd_get_value(Nrrd *,
                                               const vector<int> &index,
                                               T &value);
  
  template <class T>
  static 
  void                           nrrd_set_value(Nrrd *,
                                                const vector<int> &index,
                                                T value,
                                                unsigned int mask=0);
};




template<class T>
void
VolumeOps::nrrd_get_value(Nrrd *nrrd, 
                          const vector<int> &index, 
                          T &value)
{
  //  Nrrd *nrrd = nrrdh->nrrd_;
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
void
VolumeOps::nrrd_set_value(Nrrd *nrrd,
                          const vector<int> &index, 
                          T val, 
                          unsigned int label_mask)
{
  //  Nrrd *nrrd = nrrdh->nrrd_;
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


}
#endif
