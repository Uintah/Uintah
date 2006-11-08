/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  
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

/*
 * FILE: matlabconverter.cc
 * AUTH: Jeroen G Stinstra
 * DATE: 18 MAR 2004
 */

#include <Core/Matlab/fieldtomatlab.h>

namespace MatlabIO {

using namespace std;
using namespace SCIRun;

CompileInfoHandle
FieldToMatlabAlgo::get_compile_info(SCIRun::FieldHandle field)
{
  const TypeDescription *fieldTD = field->get_type_description();

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FieldToMatlabAlgoT");
  static const string base_class_name("FieldToMatlabAlgo");
  
  // Everything we did is in the MatlabIO namespace so we need to add this
  // Otherwise the dynamically code generator will not sgenerate a "using namespace ...."
  static const string name_space_name("MatlabIO");

  // Supply the dynamic compiler with enough information to build a file in the
  // on-the-fly libs which will have the templated function in there
  string filename = template_class_name + "." + fieldTD->get_filename() + ".";
  CompileInfoHandle ci = 
    scinew CompileInfo(filename,base_class_name,template_class_name,fieldTD->get_name());

  // Add in the include path to compile this obj
  ci->add_include(include_path); 
  // Add the MatlabIO namespace
  ci->add_namespace(name_space_name);
  // Fill out any other default values
  fieldTD->fill_compile_info(ci.get_rep());
  return(ci);
}


} // end namespace
