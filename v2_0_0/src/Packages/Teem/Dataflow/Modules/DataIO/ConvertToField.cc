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
//    File   : ConvertToField.cc
//    Author : Martin Cole
//    Date   : Wed Jan 22 13:41:24 2003

#include <Teem/Dataflow/Modules/DataIO/ConvertToField.h>
#include <teem/ten.h>

namespace SCIRun {

ConvertToFieldBase::~ConvertToFieldBase()
{
}

CompileInfoHandle
ConvertToFieldBase::get_compile_info(const TypeDescription *td) 
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
ConvertToFieldBase::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

template <class T>
void do_vector(Vector&v, void *&ptr) {
  T *&p = (T*&)ptr;
  v.x(*p);
  ++p;
  v.y(*p);
  ++p;
  v.z(*p);
  ++p;
}

template <>
void get_val_and_inc_nrrdptr<Vector>(Vector &v, void *&ptr, unsigned type)
{
  
  switch (type) {
    
  case nrrdTypeChar :
    do_vector<char>(v, ptr);
    break;
  case nrrdTypeUChar :
    do_vector<unsigned char>(v, ptr);
    break;
  case nrrdTypeShort :
    do_vector<short>(v, ptr);
    break;
  case nrrdTypeUShort :
    do_vector<unsigned short>(v, ptr);
    break;
  case nrrdTypeInt :
    do_vector<int>(v, ptr);
    break;
  case nrrdTypeUInt :
    do_vector<unsigned int>(v, ptr);
    break;
  case nrrdTypeLLong :
    do_vector<long long>(v, ptr);
    break;
  case nrrdTypeULLong :
    do_vector<unsigned long long>(v, ptr);
    break;
  case nrrdTypeFloat :
    do_vector<float>(v, ptr);
    break;
  case nrrdTypeDouble :
    do_vector<double>(v, ptr);
    break;
  }
}

template <class T>
void do_tensor(Tensor &t, void *&ptr) 
{
  T *&p = (T*&)ptr;
  ++p; // skip first value (confidence)
  t.mat_[0][0] = (*p);
  ++p;
  t.mat_[0][1] = (*p);
  ++p;
  t.mat_[0][2] = (*p);
  ++p;
  t.mat_[1][1] = (*p);
  ++p;
  t.mat_[1][2] = (*p);
  ++p;
  t.mat_[2][2] = (*p);
  ++p;
}

void get_val_and_eigens_and_inc_nrrdptr(Tensor &t, void *&ptr)
{
  float *&p = (float*&)ptr;
  float eval[3], evec[9], eval_scl[3], evec_scl[9];
  tenEigensolve(eval, evec, p);
  float scl = p[0] > 0;
  for (int cc=0; cc<3; cc++) {
    ELL_3V_SCALE(evec_scl+3*cc, scl, evec+3*cc);
    eval_scl[cc] = scl*eval[cc];
  }
  t.set_outside_eigens(Vector(evec_scl[0], evec_scl[1], evec_scl[2]),
		       Vector(evec_scl[3], evec_scl[4], evec_scl[5]),
		       Vector(evec_scl[6], evec_scl[7], evec_scl[8]),
		       eval_scl[0], eval_scl[1], eval_scl[2]);
  ++p; // skip first value (confidence)
  t.mat_[0][0] = (*p);
  ++p;
  t.mat_[0][1] = (*p);
  ++p;
  t.mat_[0][2] = (*p);
  ++p;
  t.mat_[1][1] = (*p);
  ++p;
  t.mat_[1][2] = (*p);
  ++p;
  t.mat_[2][2] = (*p);
  ++p;
}

template <>
void get_val_and_inc_nrrdptr<Tensor>(Tensor &t, void *&ptr, unsigned type)
{
  switch (type) {
    
  case nrrdTypeChar :
    do_tensor<char>(t, ptr);
    break;
  case nrrdTypeUChar :
    do_tensor<unsigned char>(t, ptr);
    break;
  case nrrdTypeShort :
    do_tensor<short>(t, ptr);
    break;
  case nrrdTypeUShort :
    do_tensor<unsigned short>(t, ptr);
    break;
  case nrrdTypeInt :
    do_tensor<int>(t, ptr);
    break;
  case nrrdTypeUInt :
    do_tensor<unsigned int>(t, ptr);
    break;
  case nrrdTypeLLong :
    do_tensor<long long>(t, ptr);
    break;
  case nrrdTypeULLong :
    do_tensor<unsigned long long>(t, ptr);
    break;
  case nrrdTypeFloat :
    do_tensor<float>(t, ptr);
    break;
  case nrrdTypeDouble :
    do_tensor<double>(t, ptr);
    break;
  }
}

} // end namespace SCIRun

