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

#include <Core/Algorithms/Fields/ApplyMappingMatrix.h>

namespace SCIRunAlgo {

using namespace SCIRun;

bool ApplyMappingMatrixAlgo::ApplyMappingMatrix(ProgressReporter *pr, FieldHandle fsrc, FieldHandle fdst, FieldHandle& output,MatrixHandle mapping)
{
  if (fsrc.get_rep() == 0)
  {
    pr->error("ApplyMappingMatrix: No input source field");
    return (false);
  }

  if (fdst.get_rep() == 0)
  {
    pr->error("ApplyMappingMatrix: No input destination field");
    return (false);
  }


  // no precompiled version available, so compile one

  FieldInformation fi_src(fsrc);
  FieldInformation fi_dst(fdst);
  FieldInformation fi_out(fdst);
  
  if ((fi_src.is_nonlinear())||(fi_dst.is_nonlinear()))
  {
    pr->error("ApplyMappingMatrix: This function has not yet been defined for non-linear elements");
    return (false);
  }

  fi_out.set_data_type(fi_src.get_data_type());

  // Setup dynamic files

  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGOApplyMappingMatrix."+fi_src.get_field_filename()+"."+fi_dst.get_field_filename()+"."+fi_out.get_field_filename()+".",
    "ApplyMappingMatrixAlgo","ApplyMappingMatrixAlgoT",
    fi_src.get_field_name() + "," + fi_dst.get_field_name() + "," + fi_out.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi_src.fill_compile_info(ci);
  fi_dst.fill_compile_info(ci);
  fi_out.fill_compile_info(ci);
  
  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;
  // Handle dynamic compilation
  
  SCIRun::Handle<ApplyMappingMatrixAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->ApplyMappingMatrix(pr,fsrc,fdst,output,mapping)); 
}


} // End namespace SCIRunAlgo

