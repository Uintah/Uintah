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
//    File   : Noise.cc
//    Author : Martin Cole
//    Date   : Fri Jun 15 17:09:17 2001

#include <Core/Algorithms/Visualization/Noise.h>
#include <Core/Algorithms/Visualization/HexMC.h>
#include <Core/Algorithms/Visualization/TetMC.h>
#include <iostream>

namespace SCIRun {

using namespace std;

NoiseAlg::NoiseAlg()
{}

NoiseAlg::~NoiseAlg() 
{}

const string& 
NoiseAlg::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

CompileInfo *
NoiseAlg::get_compile_info(const TypeDescription *td) {
  string subname;
  string subinc;
  string sname = td->get_name("", "");
  
  //Test for LatticeVol inheritance...
  if (sname.find("Lattice") != string::npos) {
    // we are dealing with a lattice vol or inherited version
    subname.append("HexMC<" + td->get_name() + "> ");
    subinc.append(HexMCBase::get_h_file_path());
  } else if (sname.find("TetVol") != string::npos) {
    subname.append("TetMC<" + td->get_name() + "> ");
    subinc.append(TetMCBase::get_h_file_path());
  } else {
    cerr << "Unsupported Field, needs to be of Lattice or Tet type." << endl;
    subname.append("Cannot compile this unsupported type");
  }

  string fname("Noise." + td->get_name(".", "."));
  CompileInfo *rval = scinew CompileInfo(fname, "NoiseAlg", 
					 "Noise", subname);
  rval->add_include(get_h_file_path());
  rval->add_include(subinc);
  td->fill_compile_info(rval);
  return rval;
}

}
