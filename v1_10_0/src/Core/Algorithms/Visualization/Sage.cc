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
//    File   : Sage.cc
//    Author : Martin Cole
//    Date   : Fri Jun 15 17:09:17 2001

#include <Core/Algorithms/Visualization/Sage.h>

namespace SCIRun {

SageAlg::SageAlg()
{}

SageAlg::~SageAlg() 
{}

const string& 
SageAlg::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

CompileInfoHandle
SageAlg::get_compile_info(const TypeDescription *td) {

  string fname("Sage." + td->get_name(".", "."));
  CompileInfo *rval = scinew CompileInfo(fname, "SageAlg", 
					 "Sage", td->get_name());
  rval->add_include(get_h_file_path());
  td->fill_compile_info(rval);
  return rval;
}

}
