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
//    File   : VolumeOps.cc
//    Author : McKay Davis
//    Date   : Tue Oct 31 17:42:53 2006


#include <StandAlone/Apps/Painter/VolumeOps.h>
#include <Core/Containers/StringUtil.h>

namespace SCIRun {


unsigned int
VolumeOps::nrrd_type_size(NrrdDataHandle &nrrdh)
{
  switch (nrrdh->nrrd_->type) {
  case nrrdTypeChar: return sizeof(char); break;
  case nrrdTypeUChar: return sizeof(unsigned char); break;
  case nrrdTypeShort: return sizeof(short); break;
  case nrrdTypeUShort: return sizeof(unsigned short); break;
  case nrrdTypeInt: return sizeof(int); break;
  case nrrdTypeUInt: return sizeof(unsigned int); break;
  case nrrdTypeLLong: return sizeof(signed long long); break;
  case nrrdTypeULLong: return sizeof(unsigned long long); break;
  case nrrdTypeFloat: return sizeof(float); break;
  case nrrdTypeDouble: return sizeof(double); break;
  default: 
    throw "Unsupported data type: "+to_string(nrrdh->nrrd_->type); break;
  }
  return 0;
}


unsigned int
VolumeOps::nrrd_elem_count(NrrdDataHandle &nrrdh)
{
  if (!nrrdh->nrrd_->dim) return 0;
  unsigned int size = nrrdh->nrrd_->axis[0].size;
  for (unsigned int a = 1; a < nrrdh->nrrd_->dim; ++a)
    size *= nrrdh->nrrd_->axis[a].size;
  return size;
}


unsigned int
VolumeOps::nrrd_data_size(NrrdDataHandle &nrrdh)
{
  return nrrd_elem_count(nrrdh) * nrrd_type_size(nrrdh);
}



NrrdDataHandle
VolumeOps::float_to_bit(NrrdDataHandle &inh, float ref, unsigned int mask)
{
  NrrdDataHandle nrrdh = VolumeOps::create_clear_nrrd(inh, nrrdTypeUInt);
  unsigned int n = VolumeOps::nrrd_elem_count(inh);
  float *src = (float *)inh->nrrd_->data;
  unsigned int *dst = (unsigned int *)nrrdh->nrrd_->data;
  for (unsigned int i = 0; i < n; ++i, ++src, ++dst) {
    if (*src >= ref) {
      *dst |= mask; 
    }
  }
 
  return nrrdh;
}

NrrdDataHandle
VolumeOps::clear_nrrd(NrrdDataHandle &inh) 
{
  memset(inh->nrrd_->data, 0, nrrd_data_size(inh));
  return inh;
}


NrrdDataHandle
VolumeOps::create_clear_nrrd(NrrdDataHandle &inh,
                                    unsigned int type) 
{
  NrrdDataHandle nout = create_nrrd(inh, type);
  return clear_nrrd(nout);
}

NrrdDataHandle
VolumeOps::create_nrrd(NrrdDataHandle &inh,
                              unsigned int type)
{
  Nrrd *src = inh->nrrd_;

  NrrdDataHandle nrrdh = new NrrdData();
  Nrrd *dst = nrrdh->nrrd_;
  
  nrrdBasicInfoCopy(dst, src, 0);
  nrrdAxisInfoCopy(dst, src, 0, 0);

  unsigned int num = nrrd_elem_count(inh);

  if (!type) type = src->type;
  dst->type = type;

  switch (type) {
  case nrrdTypeChar: dst->data = new signed char[num]; break;
  case nrrdTypeUChar: dst->data = new unsigned char[num]; break;
  case nrrdTypeShort: dst->data = new signed short[num]; break;
  case nrrdTypeUShort: dst->data = new unsigned short[num]; break;
  case nrrdTypeInt: dst->data = new signed int[num]; break;
  case nrrdTypeUInt: dst->data = new unsigned int[num]; break;
  case nrrdTypeLLong: dst->data = new signed long long[num]; break;
  case nrrdTypeULLong: dst->data = new unsigned long long[num]; break;
  case nrrdTypeFloat: dst->data = new float[num]; break;
  case nrrdTypeDouble: dst->data = new double[num]; break;
  default: throw "unhandled type in NrrdVolume::get_clear_nrrd"; break;
  }

  return nrrdh;
}


NrrdDataHandle
VolumeOps::bit_to_float(NrrdDataHandle &ninh, 
                        unsigned int mask, float value) {
  Nrrd *src = ninh->nrrd_;
  NrrdDataHandle nrrdh = create_nrrd(ninh, nrrdTypeFloat);
  Nrrd *dst = nrrdh->nrrd_;

  unsigned int *srcdata = (unsigned int *)src->data;
  float *dstdata = (float *)dst->data;  
  unsigned int num = nrrd_elem_count(ninh);
  for (unsigned int i = 0; i < num; ++i, ++srcdata, ++dstdata) {
    *dstdata = (*srcdata & mask) ? value : 0.0f;
  }

  return nrrdh;
}



}
