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

#include <Packages/ModelCreation/Core/Fields/MergeFields.h>

namespace ModelCreation {

using namespace SCIRun;

CompileInfoHandle
MergeFieldsAlgo::get_compile_info(SCIRun::FieldHandle field)
{
  const TypeDescription *fsrc = field->get_type_description();
  const TypeDescription *msrc = field->mesh()->get_type_description();
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  std::string include_path(TypeDescription::cc_to_h(__FILE__));
  std::string algo_name("MergeFieldsAlgoT");
  std::string base_name("MergeFieldsAlgo");

  CompileInfoHandle ci = 
    scinew CompileInfo(algo_name + "." +
                       fsrc->get_filename() + ".",
                       base_name, 
                       algo_name, 
                       fsrc->get_name()+","+msrc->get_name());

  // Add in the include path to compile this obj
  ci->add_include(include_path);
  ci->add_namespace("ModelCreation");   
  fsrc->fill_compile_info(ci.get_rep());
  return(ci);
}

} // namespace ModelCreation