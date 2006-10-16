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

#include <Core/Algorithms/Fields/SplitFieldByDomain.h>

namespace SCIRunAlgo {

using namespace SCIRun;

bool SplitFieldByDomainAlgo::SplitFieldByDomain(ProgressReporter *pr, FieldHandle input, FieldHandle& output)
{
  if (input.get_rep() == 0)
  {
    pr->error("SplitFieldByDomain: No input field");
    return (false);
  }

  FieldInformation fi(input);
  if (fi.is_nonlinear())
  {
    pr->error("SplitFieldByDomain: This function has not yet been defined for non-linear elements.");
    return (false);
  }

  if (!(fi.is_constantdata()))
  {
    pr->error("SplitFieldByDomain: This function only works for data located at the elements");
    return (false);
  }

  if (!(fi.is_unstructuredmesh()))
  {
    pr->error("SplitFieldByDomain: This function only works for unstructured meshes");
    return (false);
  }

  
  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGOSplitFieldByDomainAlgoT." + fi.get_field_filename() + ".",
    "SplitFieldByDomainAlgo","SplitFieldByDomainAlgoT",
    fi.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");

  fi.fill_compile_info(ci);

  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;  
  
  // Handle dynamic compilation
  SCIRun::Handle<SplitFieldByDomainAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }


  return(algo->SplitFieldByDomain(pr,input,output));    
}

} // namespace SCIRunAlgo

