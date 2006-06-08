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

#include <Core/Algorithms/Fields/ClipBySelectionMask.h>

namespace SCIRunAlgo {

using namespace SCIRun;

bool
ClipBySelectionMaskAlgo::ClipBySelectionMask(ProgressReporter *pr,
                       FieldHandle input,
                       FieldHandle& output,
                       MatrixHandle selmask,
                       MatrixHandle &interpolant,
                       int nodeclipmode)
{
  if (input.get_rep() == 0)
  {
    pr->error("ClipBySelectionMask: No input field");
    return (false);
  }

  if (selmask.get_rep() == 0)
  {
    pr->error("ClipBySelectionMask: No input selection mask");
    return (false);
  }

  FieldInformation fi(input);
  
  CompileInfoHandle ci = 
    scinew CompileInfo("ALGOClipBySelectionMask." + fi.get_field_filename() + ".",
                       "ClipBySelectionMaskAlgo","ClipBySelectionMaskAlgoT",
                       fi.get_field_name());

  // Add in the include path to compile this obj
  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  fi.fill_compile_info(ci);

  // Handle dynamic compilation
  SCIRun::Handle<ClipBySelectionMaskAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->ClipBySelectionMask(pr,input,output,selmask,interpolant,nodeclipmode));

}

} // namespace SCIRunAlgo
