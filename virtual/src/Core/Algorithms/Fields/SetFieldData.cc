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

#include <Core/Algorithms/Fields/SetFieldData.h>
#include <Core/Algorithms/Fields/GetFieldInfo.h>

namespace SCIRunAlgo {

using namespace SCIRun;

bool SetFieldDataAlgo::SetFieldData(SCIRun::ProgressReporter *pr,SCIRun::FieldHandle input, SCIRun::FieldHandle& output, SCIRun::MatrixHandle matrix, bool keepscalartype)
{

  GetFieldInfoAlgo falgo;
  int numnodes, numelems;
  if(!(falgo.GetFieldInfo(pr,input,numnodes,numelems)))
  {
    pr->error("Could not obtain dimensions of field");
    return (false);
  }

  FieldInformation fi(input);

  std::string algo_type = "";   
  int numvals;

  if ((matrix->nrows() == numnodes)||(matrix->nrows() == numelems))
  {
    if (matrix->ncols() == 1) { algo_type = "Scalar"; if (keepscalartype == false) if (algo_type ==  "Scalar") fi.make_double(); }
    if (matrix->ncols() == 3) { algo_type = "Vector"; if (algo_type ==  "Vector") fi.make_vector(); }
    if ((matrix->ncols() == 6)||(matrix->ncols() == 9)) { algo_type = "Tensor"; if (algo_type ==  "Tensor") fi.make_tensor(); }
    numvals = matrix->nrows();
    if (numnodes != numelems)
    {
      if (numvals == numnodes) fi.make_lineardata();
      if (numvals == numelems) fi.make_constantdata();
    }
  }
  else if ((matrix->ncols() == numnodes)||(matrix->ncols() == numelems))
  {
    if (matrix->nrows() == 1) { algo_type = "Scalar"; if (keepscalartype == false) if (algo_type ==  "Scalar") fi.make_double(); }
    if (matrix->nrows() == 3) { algo_type = "Vector"; if (algo_type ==  "Vector") fi.make_vector(); }
    if ((matrix->nrows() == 6)||(matrix->nrows() == 9)) { algo_type = "Tensor"; if (algo_type ==  "Tensor") fi.make_tensor(); }
    numvals = matrix->ncols();
    if (numnodes != numelems)
    {
      if (numvals == numnodes) fi.make_lineardata();
      if (numvals == numelems) fi.make_constantdata();
    }
  }
  else
  {
    if (matrix->nrows() == 1)
    {
      if (matrix->ncols() == 1) { algo_type = "Scalar"; if (keepscalartype == false) if (algo_type ==  "Scalar") fi.make_double(); }
      if (matrix->ncols() == 3) { algo_type = "Vector"; if (algo_type ==  "Vector") fi.make_vector(); }
      if ((matrix->ncols() == 6)||(matrix->ncols() == 9)) { algo_type = "Tensor"; if (algo_type ==  "Tensor") fi.make_tensor(); }    
    }
    else if (matrix->ncols() == 1)
    {
      if (matrix->nrows() == 1) { algo_type = "Scalar"; if (keepscalartype == false) if (algo_type ==  "Scalar") fi.make_double(); }
      if (matrix->nrows() == 3) { algo_type = "Vector"; if (algo_type ==  "Vector") fi.make_vector(); }
      if ((matrix->nrows() == 6)||(matrix->nrows() == 9)) { algo_type = "Tensor"; if (algo_type ==  "Tensor") fi.make_tensor(); }
    }
  }
  
  if (algo_type == "")
  {
    pr->error("SetFieldData: Matrix dimensions do not match any of the fields dimensions");
    return (false);
  }
  
  CompileInfoHandle ci = scinew CompileInfo("ALGOSetField" + algo_type + "DataAlgoT." + fi.get_field_filename() + ".",
                       "SetFieldDataAlgo", "SetField" + algo_type + "DataAlgoT", fi.get_field_name());

  // Add in the include path to compile this obj
  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");   
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  
  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;    

  SCIRun::Handle<SetFieldDataAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->SetFieldData(pr,input,output,matrix,keepscalartype));
}

} // namespace SCIRunAlgo

