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

//    File   : Noise.cc
//    Author : Martin Cole
//    Date   : Fri Jun 15 17:09:17 2001

#include <Core/Algorithms/Visualization/Noise.h>
#include <Core/Algorithms/Visualization/HexMC.h>
#include <Core/Algorithms/Visualization/UHexMC.h>
#include <Core/Algorithms/Visualization/TetMC.h>
#include <Core/Algorithms/Visualization/TriMC.h>
#include <Core/Algorithms/Visualization/QuadMC.h>
#include <iostream>

namespace SCIRun {

using namespace std;

PersistentTypeID SpanSpaceBase::type_id("SpanSpaceBase", "Datatype", 0);

NoiseAlg::NoiseAlg()
{}

NoiseAlg::~NoiseAlg() 
{}

const string& 
NoiseAlg::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

FieldHandle
NoiseAlg::get_field()
{
  return trisurf_;
}


MatrixHandle
NoiseAlg::get_interpolant()
{
  return interpolant_;
}


CompileInfoHandle
NoiseAlg::get_compile_info(const TypeDescription *td,
			   bool cell_centered_p,
			   bool face_centered_p)
{
  string subname;
  string subinc;
  string sname = td->get_name("", "");

  //Test for LatVolField inheritance...
  if (sname.find("LatVolField") != string::npos ||
      sname.find("StructHexVolField") != string::npos) {
    // we are dealing with a lattice vol or inherited version
    subname.append("HexMC<" + td->get_name() + "> ");
    subinc.append(HexMCBase::get_h_file_path());
    face_centered_p = false;
  } else if (sname.find("TetVolField") != string::npos) {
    subname.append("TetMC<" + td->get_name() + "> ");
    subinc.append(TetMCBase::get_h_file_path());
    face_centered_p = false;
  } else if (sname.find("HexVolField") != string::npos) {
    subname.append("UHexMC<" + td->get_name() + "> ");
    subinc.append(UHexMCBase::get_h_file_path());
    face_centered_p = false;
  } else if (sname.find("TriSurfField") != string::npos) {
    subname.append("TriMC<" + td->get_name() + "> ");
    subinc.append(TriMCBase::get_h_file_path());
    cell_centered_p = false;
  } else if (sname.find("QuadSurfField") != string::npos) {
    subname.append("QuadMC<" + td->get_name() + "> ");
    subinc.append(QuadMCBase::get_h_file_path());
    cell_centered_p = false;
  } else if (sname.find("ImageField") != string::npos) {
    subname.append("QuadMC<" + td->get_name() + "> ");
    subinc.append(QuadMCBase::get_h_file_path());
    cell_centered_p = false;
  } else {
    cerr << "Unsupported Field, needs to be of Lattice or Tet type." << endl;
    subname.append("Cannot compile this unsupported type");
  }

  string dloc = "";
  if (cell_centered_p) { dloc = "Cell"; }
  if (face_centered_p) { dloc = "Face"; }

  string fname("Noise" + dloc + "." + td->get_filename() + ".");
  CompileInfo *rval = scinew CompileInfo(fname, "NoiseAlg", 
					 "Noise" + dloc,
					 subname);
  rval->add_include(get_h_file_path());
  rval->add_include(subinc);
  td->fill_compile_info(rval);
  return rval;
}

}
