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

//    File   : ConvertToField.cc
//    Author : Martin Cole
//    Date   : Wed Jan 22 13:41:24 2003

#include <Teem/Dataflow/Modules/DataIO/ConvertToField.h>
#include <teem/ten.h>

namespace SCIRun {

ConvertToFieldBase::~ConvertToFieldBase()
{
}

ConvertToFieldEigenBase::~ConvertToFieldEigenBase()
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

CompileInfoHandle
ConvertToFieldEigenBase::get_compile_info(const TypeDescription *td) 
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
ConvertToFieldEigenBase::get_h_file_path() {
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
  //  float scl = (p[0] > 0.5);
  ++p; // skip first value (confidence)
  t.mat_[0][0] = (*p);//*scl;
  ++p;
  t.mat_[0][1] = (*p);//*scl;
  ++p;
  t.mat_[0][2] = (*p);//*scl;
  ++p;
  t.mat_[1][1] = (*p);//*scl;
  ++p;
  t.mat_[1][2] = (*p);//*scl;
  ++p;
  t.mat_[2][2] = (*p);//*scl;
  ++p;
}

void get_val_and_eigens_and_inc_nrrdptr(Tensor &t, void *&ptr)
{
  float *&p = (float*&)ptr;
  float eval[3], evec[9], eval_scl[3], evec_scl[9];
  tenEigensolve(eval, evec, p);

//  cerr << "p= ["<<p[0]<<", "<<p[1]<<", "<<p[2]<<", "<<p[3]<<", "<<p[4]<<", "<<p[5]<<", "<<p[6]<<"]\n";
//  cerr << "evals = "<<eval[0]<<", "<<eval[1]<<", "<<eval[2]<<"\n";
//  cerr << "evecs= ["<<evec[0]<<", "<<evec[1]<<", "<<evec[2]<<"]\n";
//  cerr << "       ["<<evec[3]<<", "<<evec[4]<<", "<<evec[5]<<"]\n";
//  cerr << "       ["<<evec[6]<<", "<<evec[7]<<", "<<evec[8]<<"]\n";

  float scl = p[0] > 0.5;
  for (int cc=0; cc<3; cc++) {
    ELL_3V_SCALE(evec_scl+3*cc, scl, evec+3*cc);
    eval_scl[cc] = scl*eval[cc];
  }
  Vector e1(evec_scl[0], evec_scl[1], evec_scl[2]);
  Vector e2(evec_scl[3], evec_scl[4], evec_scl[5]);
  Vector e3(evec_scl[6], evec_scl[7], evec_scl[8]);
//  cerr << "e1="<<e1<<"  e2="<<e2<<"  e3="<<e3<<"  l1="<<eval_scl[0]<<" l2="<<eval_scl[1]<<" l3="<<eval_scl[2]<<"\n";
  t.set_outside_eigens(e1, e2, e3, eval_scl[0], eval_scl[1], eval_scl[2]);
  ++p; // skip first value (confidence)
  t.mat_[0][0] = (*p);//*scl;
  ++p;
  t.mat_[0][1] = (*p);//*scl;
  ++p;
  t.mat_[0][2] = (*p);//*scl;
  ++p;
  t.mat_[1][1] = (*p);//*scl;
  ++p;
  t.mat_[1][2] = (*p);//*scl;
  ++p;
  t.mat_[2][2] = (*p);//*scl;
  ++p;
//  cerr << "tensor="<<t<<"\n";
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

