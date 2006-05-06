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

#include <Packages/ModelCreation/Core/Fields/ScaleField.h>

namespace ModelCreation {

using namespace SCIRun;

bool ScaleFieldAlgo::ScaleField(ProgressReporter *pr, FieldHandle input, FieldHandle& output,double datascale, double meshscale)
{
  if (input.get_rep() == 0)
  {
    pr->error("ScaleField: No input field");
    return (false);
  }

  // no precompiled version available, so compile one

  FieldInformation fi(input);
  
  if (fi.is_nonlinear())
  {
    pr->error("ScaleField: This function has not yet been defined for non-linear elements");
    return (false);
  }
  
  for(size_t p =0; p< precompiled_.size(); p++)
  {
    if (precompiled_[p]->testinput(input)) return(precompiled_[p]->ScaleField(pr,input,output,datascale,meshscale));
  }

  // Setup dynamic files

  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGOScaleField."+fi.get_field_filename()+".",
    "ScaleFieldAlgo","ScaleFieldAlgoT",
    fi.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("ModelCreation");
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  
  // Handle dynamic compilation
  SCIRun::Handle<ScaleFieldAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->ScaleField(pr,input,output,datascale,meshscale));
}

bool ScaleFieldAlgo::testinput(FieldHandle input)
{
  return (false);
}

AlgoList<ScaleFieldAlgo> ScaleFieldAlgo::precompiled_;

} // End namespace ModelCreation
