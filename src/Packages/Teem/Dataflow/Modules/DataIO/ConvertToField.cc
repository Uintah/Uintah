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

template <>
void get_val_and_inc_nrrdptr<Vector>(Vector &v, void *&ptr)
{
  double *&p = (double*&)ptr;
  v.x(*p);
  ++p;
  v.y(*p);
  ++p;
  v.z(*p);
  ++p;
}

template <>
void get_val_and_inc_nrrdptr<Tensor>(Tensor &t, void *&ptr)
{
  double *&p = (double*&)ptr;
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

} // end namespace SCIRun

