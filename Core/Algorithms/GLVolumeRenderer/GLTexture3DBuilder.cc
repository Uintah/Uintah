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
//    File   : GLTexture3DBuilder.cc
//    Author : Martin Cole
//    Date   : Fri Jun 15 17:09:17 2001

#include <Core/Algorithms/GLVolumeRenderer/GLTexture3DBuilder.h>
#include <iostream>

namespace SCIRun {

using namespace std;

GLTexture3DBuilderAlg::GLTexture3DBuilderAlg()
{}

GLTexture3DBuilderAlg::~GLTexture3DBuilderAlg() 
{}

const string& 
GLTexture3DBuilderAlg::get_h_file_path() 
{
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

CompileInfoHandle
GLTexture3DBuilderAlg::get_compile_info(const TypeDescription *td )
{
  string subname;
  string subinc;
  string sname = td->get_name("", "");

  //Test for LatVolField inheritance...
  if (sname.find("LatVol") != string::npos ) {
    // we are dealing with a lattice vol or inherited version
    //subname.append("GLTexture3DBuilder<" + td->get_name() + "> ");
    subname.append(td->get_name());
    subinc.append(get_h_file_path());
  } else {
    cerr << "Unsupported Geometry, needs to be of Lattice type." << endl;
    subname.append("Cannot compile this unsupported type");
  }

  string fname("GLTexture3DBuilder." + td->get_filename() + ".");
  CompileInfo *rval = scinew CompileInfo(fname, "GLTexture3DBuilderAlg", 
					 "GLTexture3DBuilder",
					 subname);
  rval->add_include(get_h_file_path());
  rval->add_include(subinc);
  td->fill_compile_info(rval);
  return rval;
}

}
