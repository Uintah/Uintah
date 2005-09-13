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
