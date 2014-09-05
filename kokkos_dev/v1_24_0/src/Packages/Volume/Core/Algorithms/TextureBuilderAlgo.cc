//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : TextureBuilderAlgo.cc
//    Author : Milan Ikits
//    Date   : Wed Jul 14 23:54:35 2004

#include <Packages/Volume/Core/Algorithms/TextureBuilderAlgo.h>
#include <Core/Util/DebugStream.h>

#include <iostream>

namespace Volume {

using namespace SCIRun;
using namespace std;

static DebugStream dbg("TextureBuilderAlgo", false);

TextureBuilderAlgoBase::TextureBuilderAlgoBase()
{}

TextureBuilderAlgoBase::~TextureBuilderAlgoBase() 
{}

const string& 
TextureBuilderAlgoBase::get_h_file_path() 
{
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}

CompileInfoHandle
TextureBuilderAlgoBase::get_compile_info(const TypeDescription* td)
{
  string subname;
  string subinc;
  string sname = td->get_name("", "");

  //Test for LatVolField inheritance...
  if(sname.find("LatVol") != string::npos) {
    // we are dealing with a lattice vol or inherited version
    //subname.append("TextureBuilderAlgo<" + td->get_name() + "> ");
    subname.append(td->get_name());
    subinc.append(get_h_file_path());
  } else {
    cerr << "Unsupported Geometry, needs to be of Lattice type." << endl;
    subname.append("Cannot compile this unsupported type");
  }
  string fname("TextureBuilderAlgo." + td->get_filename() + ".");
  CompileInfo *rval = scinew CompileInfo(fname, "TextureBuilderAlgoBase", 
					 "TextureBuilderAlgo",
					 subname);
  rval->add_include(get_h_file_path());
  rval->add_include(subinc);
  rval->add_namespace("Volume");
  td->fill_compile_info(rval);
  return rval;
}

} // namespace Volume
