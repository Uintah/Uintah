//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : ConvertToNrrd.cc
//    Author : Martin Cole
//    Date   : Tue Jan  7 10:24:59 2003

#include <Teem/Dataflow/Modules/DataIO/ConvertToNrrd.h>

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
  int nx = data.dim3();
  int ny = data.dim2();
  int nz = data.dim1();
  double *new_data = new double[nx * ny * nz * 3];
  double *p = new_data;
  for (int k = 0; k < nz; k++) {
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
	fill_data(data(k,j,i), p);
	p += 3;
      }
    }
  }
  
  return new_data;
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


} // end namespace SCIRun

