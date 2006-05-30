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

#include <Core/Algorithms/Regression/CompareFields.h>

namespace SCIRunAlgo {

using namespace SCIRun;

bool CompareFieldsAlgo::CompareFields(ProgressReporter *pr, FieldHandle field1, FieldHandle field2)
{
  if (field1.get_rep() == field2.get_rep()) return (true);

  if (field1.get_rep() == 0)
  {
    pr->remark("CompareFields: Field 1 is empty");
    return (false);
  }

  if (field2.get_rep() == 0)
  {
    pr->remark("CompareFields: Field 2 is empty");
    return (false);
  }

  // no precompiled version available, so compile one

  FieldInformation fi1(field1);
  FieldInformation fi2(field2);

  if (!(fi1 == fi2))
  {
    pr->remark("CompareFields: Field types are not equal");
    return (false);  
  }
  
  // Setup dynamic files

  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGOCompareFields."+fi1.get_field_filename()+".",
    "CompareFieldsAlgo","CompareFielsAlgoT",
    fi1.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi1.fill_compile_info(ci);
  
  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;
  
  SCIRun::Handle<CompareFieldsAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->CompareFields(pr,field1,field2)); 
}


} // End namespace SCIRunAlgo

