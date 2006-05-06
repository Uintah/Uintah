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

#include <Packages/ModelCreation/Core/Fields/FieldDataElemToNode.h>

namespace ModelCreation {

using namespace SCIRun;

bool FieldDataElemToNodeAlgo::FieldDataElemToNode(ProgressReporter *pr,
                              FieldHandle input,
                              FieldHandle& output,
                              std::string method)
{
  if (input.get_rep() == 0)
  {
    pr->error("FieldDataNodeToElem: No input source field");
    return (false);
  }

  FieldInformation fi(input);
  FieldInformation fo(output);
  
  fo.make_lineardata();

  if (fi.is_lineardata())
  {
    pr->remark("FieldDataElemToNode: Skipping conversion data is already at nodes");
    output = input;
    return (true);
  }

  if (!(fi.is_constantdata()))
  {
    pr->error("FieldDataElemToNode: This function needs to have data at the elements");
    return (false);  
  }

  if (fi.is_nonlinear())
  {
    pr->error("FieldDataNodeToElem: This function has not been implemented for non linear elements");
    return (false);
  }

  CompileInfoHandle ci = scinew CompileInfo("ALGOFieldDataElemToNodeAlgo." +
                       fi.get_field_filename() + "." + fo.get_field_filename() + ".",
                       "FieldDataElemToNodeAlgo","FieldDataElemToNodeAlgoT",
                       fi.get_field_name() + "," + fo.get_field_name());


  // Add in the include path to compile this obj
  ci->add_data_include(SCIRun::TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("ModelCreation");  
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  fo.fill_compile_info(ci);  
  
  SCIRun::Handle<FieldDataElemToNodeAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->FieldDataElemToNode(pr,input,output,method));  
}

} // namespace ModelCreation