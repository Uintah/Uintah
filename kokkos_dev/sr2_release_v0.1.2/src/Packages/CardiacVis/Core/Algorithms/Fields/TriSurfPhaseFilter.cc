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

#include <Packages/CardiacVis/Core/Algorithms/Fields/TriSurfPhaseFilter.h>

namespace CardiacVis {

using namespace SCIRun;

bool TriSurfPhaseFilterAlgo::TriSurfPhaseFilter(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle& phaseline, FieldHandle& phasepoint)
{
  if (input.get_rep() == 0)
  {
    pr->error("TriSurfPhaseFilter: No input field");
    return (false);
  }

  // no precompiled version available, so compile one

  FieldInformation fi(input);
  FieldInformation fo(input);
  FieldInformation fo2(input);
  FieldInformation fo3(input);
  
  if (!(fi.is_linear()))
  {
    pr->error("TriSurfPhaseFilter: This function has not yet been defined for non-linear elements");
    return (false);
  }
  
  std::string mesh_type = fi.get_mesh_type();
  std::string data_type = fi.get_data_type();
  if (mesh_type != "TriSurfMesh" || data_type != "double")
  {
    pr->error("TriSurfPhaseFilter: This function works only for TriSurf Meshes with a double as input");
    return (false);
  }
  
  fo.set_mesh_type("TriSurfMesh");
  fo.set_data_type("double");
  fo2.set_mesh_type("CurveMesh");
  fo2.set_data_type("double");
  fo3.set_mesh_type("PointCloudMesh");
  fo3.set_data_type("double");

  // Setup dynamic files

  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGOTriSurfPhaseFilter."+fi.get_field_filename()+"."+fo.get_field_filename()+".",
    "TriSurfPhaseFilterAlgo","TriSurfPhaseFilterAlgoT",
    fi.get_field_name() + "," + fo.get_field_name() + "," + fo2.get_field_name() + "," + fo3.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("CardiacVis");
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  fo.fill_compile_info(ci);
  fo2.fill_compile_info(ci);
  fo3.fill_compile_info(ci);
  
  // Handle dynamic compilation
  SCIRun::Handle<TriSurfPhaseFilterAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
//    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->TriSurfPhaseFilter(pr,input,output,phaseline,phasepoint));
}


} // End namespace ModelCreation
