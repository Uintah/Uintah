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

#include <Core/Algorithms/Fields/DomainBoundary.h>

namespace SCIRunAlgo {

using namespace SCIRun;

bool DomainBoundaryAlgo::DomainBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly)
{
  if (input.get_rep() == 0)
  {
    pr->error("DomainBoundary: No input field");
    return (false);
  }

  FieldInformation fi(input);
  FieldInformation fo(input);
  
  if (fi.is_nonlinear())
  {
    pr->error("DomainBoundary: This function has not yet been defined for non-linear elements");
    return (false);
  }
  
  if (!(fi.is_constantdata()))
  {
    pr->error("DomainBoundary: This function needs a compartment definition on the elements (constant element data)");
    return (false);    
  }
  
  if (!(fi.is_volume()||fi.is_surface()))
  {
    pr->error("DomainBoundary: THis function is only defined for surface and volume data");
    return (false);
  }

  std::string mesh_type = fi.get_mesh_type();
  if ((mesh_type == "LatVolMesh")||(mesh_type == "StructHexVolMesh")||(mesh_type == "HexVolMesh"))
  {
    fo.set_mesh_type("QuadSurfMesh");
  }
  else if ((mesh_type == "ImageMesh")||(mesh_type == "StructQuadSurfMesh")||(mesh_type == "QuadSurfMesh")||(mesh_type == "TriSurfMesh"))
  {
    fo.set_mesh_type("CurveMesh");
  }
  else if (mesh_type == "TetVolMesh")
  {
    fo.set_mesh_type("TriSurfMesh");
  }
  else
  {
    pr->error("No method available for mesh: " + mesh_type);
    return (false);
  }

  fo.make_nodata();

  // Setup dynamic files

  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGODomainBoundary."+fi.get_field_filename()+"."+fo.get_field_filename()+".",
    "DomainBoundaryAlgo","DomainBoundaryAlgoT",
    fi.get_field_name() + "," + fo.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  fo.fill_compile_info(ci);
  
  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;
    
  // Handle dynamic compilation
  SCIRun::Handle<DomainBoundaryAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->DomainBoundary(pr,input,output,DomainLink,minrange,maxrange,userange,addouterboundary,innerboundaryonly));
}



bool DomainBoundary2Algo::DomainBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly)
{
  if (input.get_rep() == 0)
  {
    pr->error("DomainBoundary: No input field");
    return (false);
  }

  FieldInformation fi(input);
  FieldInformation fo(input);
  
  if (fi.is_nonlinear())
  {
    pr->error("DomainBoundary: This function has not yet been defined for non-linear elements");
    return (false);
  }
  
  if (!(fi.is_constantdata()))
  {
    pr->error("DomainBoundary: This function needs a compartment definition on the elements (constant element data)");
    return (false);    
  }
  
  if (!(fi.is_volume()||fi.is_surface()))
  {
    pr->error("DomainBoundary: THis function is only defined for surface and volume data");
    return (false);
  }

  std::string mesh_type = fi.get_mesh_type();
  if ((mesh_type == "LatVolMesh")||(mesh_type == "StructHexVolMesh")||(mesh_type == "HexVolMesh"))
  {
    fo.set_mesh_type("QuadSurfMesh");
  }
  else if ((mesh_type == "ImageMesh")||(mesh_type == "StructQuadSurfMesh")||(mesh_type == "QuadSurfMesh")||(mesh_type == "TriSurfMesh"))
  {
    fo.set_mesh_type("CurveMesh");
  }
  else if (mesh_type == "TetVolMesh")
  {
    fo.set_mesh_type("TriSurfMesh");
  }
  else
  {
    pr->error("No method available for mesh: " + mesh_type);
    return (false);
  }

  fo.make_nodata();

  // Setup dynamic files

  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGODomainBoundary2."+fi.get_field_filename()+"."+fo.get_field_filename()+".",
    "DomainBoundary2Algo","DomainBoundary2AlgoT",
    fi.get_field_name() + "," + fo.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  fo.fill_compile_info(ci);
  
  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;    
  
  // Handle dynamic compilation
  SCIRun::Handle<DomainBoundary2Algo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->DomainBoundary(pr,input,output,DomainLink,minrange,maxrange,userange,addouterboundary,innerboundaryonly));
}


} // End namespace SCIRunAlgo
