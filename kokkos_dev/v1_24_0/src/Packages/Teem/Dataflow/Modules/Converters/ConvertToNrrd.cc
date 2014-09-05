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

//    File   : ConvertToNrrd.cc
//    Author : Martin Cole
//    Date   : Tue Jan  7 10:24:59 2003

#include <Teem/Dataflow/Modules/Converters/ConvertToNrrd.h>

namespace SCIRun {

ConvertToNrrdBase::~ConvertToNrrdBase()
{
}

CompileInfoHandle
ConvertToNrrdBase::get_compile_info(const TypeDescription *td) 
{
  CompileInfo *rval = scinew CompileInfo(dyn_file_name(td), 
					 base_class_name(), 
					 template_class_name(), 
					 td->get_name());
  rval->add_include(get_h_file_path());
  td->fill_compile_info(rval);
  return rval;
}

const string& 
ConvertToNrrdBase::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

template <>
void fill_data<Vector>(Vector &v, double *p) {
  p[0] = v.x();
  p[1] = v.y();
  p[2] = v.z();
}

template <>
void fill_data<Tensor>(Tensor &t, double *p) {
  p[0] = 1.0;
  p[1] = t.mat_[0][0];
  p[2] = t.mat_[0][1];
  p[3] = t.mat_[0][2];
  p[4] = t.mat_[1][1];
  p[5] = t.mat_[1][2];
  p[6] = t.mat_[2][2];
  p += 7;
}


template <>
void* 
get_raw_data_ptr<FData2d<char> >(FData2d<char> &data, int) {
  return &(data(0,0));
}

template <>
void* 
get_raw_data_ptr<FData2d<unsigned char> >(FData2d<unsigned char> &data, int) {
  return &(data(0,0));
}

template <>
void* 
get_raw_data_ptr<FData2d<short> >(FData2d<short> &data, int) {
  return &(data(0,0));
}

template <>
void* 
get_raw_data_ptr<FData2d<unsigned short> >(FData2d<unsigned short> &data, int) {
  return &(data(0,0));
}

template <>
void* get_raw_data_ptr<FData2d<int> >(FData2d<int> &data, int) {
  return &(data(0,0));
}

template <>
void* 
get_raw_data_ptr<FData2d<unsigned int> >(FData2d<unsigned int> &data, int) {
  return &(data(0,0));
}

template <>
void* 
get_raw_data_ptr<FData2d<long long> >(FData2d<long long> &data, int) {
  return &(data(0,0));
}

template <>
void* 
get_raw_data_ptr<FData2d<unsigned long long> >(FData2d<unsigned long long> &data, int) {
  return &(data(0,0));
}

template <>
void* 
get_raw_data_ptr<FData2d<float> >(FData2d<float> &data, int) {
  return &(data(0,0));
}

template <>
void* 
get_raw_data_ptr<FData2d<double> >(FData2d<double> &data, int) {
  return &(data(0,0));
}


template <>
void* 
get_raw_data_ptr<FData3d<char> >(FData3d<char> &data, int) {
  return &(data(0,0,0));
}

template <>
void* 
get_raw_data_ptr<FData3d<unsigned char> >(FData3d<unsigned char> &data, int) {
  return &(data(0,0,0));
}

template <>
void* 
get_raw_data_ptr<FData3d<short> >(FData3d<short> &data, int) {
  return &(data(0,0,0));
}

template <>
void* 
get_raw_data_ptr<FData3d<unsigned short> >(FData3d<unsigned short> &data, int) {
  return &(data(0,0,0));
}

template <>
void* get_raw_data_ptr<FData3d<int> >(FData3d<int> &data, int) {
  return &(data(0,0,0));
}

template <>
void* 
get_raw_data_ptr<FData3d<unsigned int> >(FData3d<unsigned int> &data, int) {
  return &(data(0,0,0));
}

template <>
void* 
get_raw_data_ptr<FData3d<long long> >(FData3d<long long> &data, int) {
  return &(data(0,0,0));
}

template <>
void* 
get_raw_data_ptr<FData3d<unsigned long long> >(FData3d<unsigned long long> &data, int) {
  return &(data(0,0,0));
}

template <>
void* 
get_raw_data_ptr<FData3d<float> >(FData3d<float> &data, int) {
  return &(data(0,0,0));
}

template <>
void* 
get_raw_data_ptr<FData3d<double> >(FData3d<double> &data, int)  {
  return &(data(0,0,0));
}

// Vector and Tensor data gets a copy to a newly allocated chunk of memory.
template <>
void* 
get_raw_data_ptr<FData3d<Vector> >(FData3d<Vector> &data, int)  {
  ASSERT(sizeof(Vector) == sizeof(double) * 3);
  return &(data(0,0,0));
}

template <>
void* 
get_raw_data_ptr<FData3d<Tensor> >(FData3d<Tensor> &data, int)  {
  int nx = data.dim3();
  int ny = data.dim2();
  int nz = data.dim1();
  float *new_data = new float[nx * ny * nz * 7];
  float *p = new_data;
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
	Tensor &t = data(k,j,i);
	p[0] = 1.0;
	p[1] = t.mat_[0][0];
	p[2] = t.mat_[0][1];
	p[3] = t.mat_[0][2];
	p[4] = t.mat_[1][1];
	p[5] = t.mat_[1][2];
	p[6] = t.mat_[2][2];
	p += 7;
      }
    }
  }
  return new_data;
}

template <>
void* 
get_raw_data_ptr<FData2d<Vector> >(FData2d<Vector> &data, int)  {
  ASSERT(sizeof(Vector) == sizeof(double) * 3);
  return &(data(0,0));
}

template <>
void* 
get_raw_data_ptr<FData2d<Tensor> >(FData2d<Tensor> &data, int)  {
  int nx = data.dim2();
  int ny = data.dim1();
  float *new_data = new float[nx * ny * 7];
  float *p = new_data;
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      Tensor &t = data(j,i);
      p[0] = 1.0;
      p[1] = t.mat_[0][0];
      p[2] = t.mat_[0][1];
      p[3] = t.mat_[0][2];
      p[4] = t.mat_[1][1];
      p[5] = t.mat_[1][2];
      p[6] = t.mat_[2][2];
      p += 7;
    }
  }
  return new_data;
}

template <>
void*
get_raw_data_ptr<vector<Vector> >(vector<Vector> &data, int)  {
  ASSERT(sizeof(Vector) == sizeof(double) * 3);
  return &(data[0]);
}

template <>
void* 
get_raw_data_ptr<vector<Tensor> >(vector<Tensor> &data, int)  {
  int sz = data.size();
  float *new_data = new float[sz * 7];
  float *p = new_data;
  for (int i = 0; i < sz; i++) {
    Tensor &t = data[i];
    p[0] = 1.0;
    p[1] = t.mat_[0][0];
    p[2] = t.mat_[0][1];
    p[3] = t.mat_[0][2];
    p[4] = t.mat_[1][1];
    p[5] = t.mat_[1][2];
    p[6] = t.mat_[2][2];
    p += 7;
  }
  return new_data;
}


} // end namespace SCIRun

