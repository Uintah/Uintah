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

#include <Core/Algorithms/Fields/ConvertIndicesToFieldData.h>

namespace SCIRunAlgo {

using namespace SCIRun;

bool 
ConvertIndicesToFieldDataAlgo::ConvertIndicesToFieldData(ProgressReporter *pr, 
							 FieldHandle input, 
							 FieldHandle& output,
							 MatrixHandle data)
{
  if (input.get_rep() == 0)
  {
    pr->error("ConvertIndicesToFieldData: No input field");
    return (false);
  }

  // no precompiled version available, so compile one

  FieldInformation fi(input);
  FieldInformation fo(input);
  
  if (fi.is_nonlinear())
  {
    pr->error("ConvertIndicesToFieldData: This function has not yet been defined for non-linear elements");
    return (false);
  }

  if (fi.is_nodata())
  {
    pr->error("ConvertIndicesToFieldData: This function has not yet been defined for fields with no data");
    return (false);
  }
  
  int nrows = data->nrows();
  int ncols = data->ncols();

  std::string algotype;
  
  if (ncols == 1)
  {
    algotype = "Scalar";
  }
  else if (ncols == 3)
  {
    algotype = "Vector";
  }
  else if (ncols == 6 || ncols == 9)
  {
    algotype = "Tensor";
  }
  else
  {
    if (nrows == 1)
    {
      algotype = "Scalar";
    }
    else if (nrows == 3)
    {
      algotype = "Vector";
    }
    else if (nrows == 6 || nrows == 9)
    {
      algotype = "Tensor";
    }
    else
    {
      pr->error("ConvertIndicesToFieldData: Data does not have dimension of 1, 3, 6, or 9");
      return (false);      
    }
  }
  
  if (algotype == "Scalar") fo.make_double();
  if (algotype == "Vector") fo.make_vector();
  if (algotype == "Tensor") fo.make_tensor();

  // Setup dynamic files

  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGOConvertIndicesToFieldData."+fi.get_field_filename()+"."+fo.get_field_filename()+".",
    "ConvertIndicesToFieldDataAlgo","IndicesTo"+algotype+"AlgoT",
    fi.get_field_name() + "," + fo.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  fo.fill_compile_info(ci);
  
  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;  
  
  // Handle dynamic compilation
  SCIRun::Handle<ConvertIndicesToFieldDataAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->ConvertIndicesToFieldData(pr,input,output,data));
}

} // End namespace SCIRunAlgo
