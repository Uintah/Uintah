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

#include <Core/Algorithms/Fields/MapFieldDataFromElemToNode.h>

namespace SCIRunAlgo {

using namespace SCIRun;

bool MapFieldDataFromElemToNodeAlgo::MapFieldDataFromElemToNode(ProgressReporter *pr,
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
  FieldInformation fo(input);
  
  fo.make_lineardata();

  if (fi.is_lineardata())
  {
    pr->remark("MapFieldDataFromElemToNode: Skipping conversion data is already at nodes");
    output = input;
    return (true);
  }

  if (!(fi.is_constantdata()))
  {
    pr->error("MapFieldDataFromElemToNode: This function needs to have data at the elements");
    return (false);  
  }

  if (fi.is_nonlinear())
  {
    pr->error("FieldDataNodeToElem: This function has not been implemented for non linear elements");
    return (false);
  }

  CompileInfoHandle ci = scinew CompileInfo("ALGOMapFieldDataFromElemToNodeAlgo." +
                       fi.get_field_filename() + "." + fo.get_field_filename() + ".",
                       "MapFieldDataFromElemToNodeAlgo","MapFieldDataFromElemToNodeAlgoT",
                       fi.get_field_name() + "," + fo.get_field_name());


  // Add in the include path to compile this obj
  ci->add_data_include(SCIRun::TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");  
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  fo.fill_compile_info(ci);  
  
  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;    
  
  SCIRun::Handle<MapFieldDataFromElemToNodeAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
//    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->MapFieldDataFromElemToNode(pr,input,output,method));  
}

} // namespace SCIRunAlgo
