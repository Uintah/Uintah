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

#include <Core/Algorithms/Fields/GetFieldBoundary.h>

namespace SCIRunAlgo {

using namespace SCIRun;

bool GetFieldBoundaryAlgo::GetFieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle& mapping)
{
  // Check whether we have an input field
  if (input.get_rep() == 0)
  {
    pr->error("FieldBoundary: No input field");
    return (false);
  }

  // Figure out what the input type and output type have to be
  FieldInformation fi(input);
  FieldInformation fo(input);
  
  if (fi.is_nonlinear())
  {
    pr->error("FieldBoundary: This function has not yet been defined for non-linear elements");
    return (false);
  }
  
  if (!(fi.is_volume()||fi.is_surface()||fi.is_curve()))
  {
    pr->error("FieldBoundary: this function is only defined for curve, surface and volume data");
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
  else if ((mesh_type == "CurveMesh")||(mesh_type == "StructCurveMesh")||(mesh_type == "ScanlineMesh"))
  {
    fo.set_mesh_type("PointCloudMesh");
  }
  else
  {
    pr->error("No method available for mesh: " + mesh_type);
    return (false);
  }

  ////////////////////////////////////////////////
  // VIRTUAL VERSION OF THIS ALGORITHM

  std::cout << "fi.interface="<< fi.has_virtual_interface() <<"\n";
  std::cout << "fo.interface="<< fo.has_virtual_interface() <<"\n";

  if (fi.has_virtual_interface() && fo.has_virtual_interface())
  {
    output = Create_Field(fo);
    
    if (UseScalarInterface(fi,fo)) return (GetFieldBoundaryV<double>(pr,input,output,mapping));
    if (UseVectorInterface(fi,fo)) return (GetFieldBoundaryV<Vector>(pr,input,output,mapping));
    if (UseTensorInterface(fi,fo)) return (GetFieldBoundaryV<Tensor>(pr,input,output,mapping));
  }

  ////////////////////////////////////////////////
  // DYNAMIC COMPILATION VERSION OF THIS ALGORITHM

  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGOGetFieldBoundary."+fi.get_field_filename()+"."+fo.get_field_filename()+".",
    "GetFieldBoundaryAlgo","GetFieldBoundaryAlgoT",
    fi.get_field_name() + "," + fo.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  fo.fill_compile_info(ci);
  
  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;
  
  // Handle dynamic compilation
  SCIRun::Handle<GetFieldBoundaryAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }
  
  std::cout << "Finished dynamic compilation\n";
  return(algo->GetFieldBoundary(pr,input,output,mapping));

}

} // End namespace SCIRunAlgo
