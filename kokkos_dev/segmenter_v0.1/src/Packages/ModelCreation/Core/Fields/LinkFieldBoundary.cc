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

#include <Packages/ModelCreation/Core/Fields/LinkFieldBoundary.h>

namespace ModelCreation {

using namespace SCIRun;

bool LinkFieldBoundaryAlgo::LinkFieldBoundary(ProgressReporter *pr, FieldHandle input, MatrixHandle& NodeLink, MatrixHandle& ElemLink, double tol, bool linkx, bool linky, bool linkz)
{
  if (input.get_rep() == 0)
  {
    pr->error("LinkFieldBoundary: No input field");
    return (false);
  }

  FieldInformation fi(input);
  
  if (fi.is_nonlinear())
  {
    pr->error("LinkFieldBoundary: This function has not yet been defined for non-linear elements");
    return (false);
  }
   
  if (!(fi.is_volume()||fi.is_surface()||fi.is_curve()))
  {
    pr->error("LinkFieldBoundary: this function is only defined for curve, surface and volume data");
    return (false);
  }

  std::string algotype = "";  
  if (fi.is_volume()) algotype = "Volume";
  if (fi.is_surface()) algotype = "Surface";
  if (fi.is_curve()) algotype = "Curve";
  
  // Setup dynamic files

  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "LinkFieldBoundary."+fi.get_field_filename()+".",
    "LinkFieldBoundaryAlgo","LinkFieldBoundary"+algotype+"AlgoT",
    fi.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("ModelCreation");
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  
  // Handle dynamic compilation
  SCIRun::Handle<LinkFieldBoundaryAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->LinkFieldBoundary(pr,input,NodeLink,ElemLink,tol,linkx,linky,linkz));
}


} // End namespace ModelCreation
