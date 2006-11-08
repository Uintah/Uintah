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

#include <Core/Algorithms/Fields/ConvertMeshToTetVol.h>

namespace SCIRunAlgo {

using namespace SCIRun;

bool ConvertMeshToTetVolAlgo::ConvertMeshToTetVol(ProgressReporter *pr, FieldHandle input, FieldHandle& output)
{
  if (input.get_rep() == 0)
  {
    pr->error("ConvertMeshToTetVol: No input field");
    return (false);
  }

  // no precompiled version available, so compile one

  FieldInformation fi(input);
  FieldInformation fo(input);
  
  if (fi.is_nonlinear())
  {
    pr->error("ConvertMeshToTetVol: This function has not yet been defined for non-linear elements");
    return (false);
  }
  
  std::string algotype;
  std::string mesh_type = fi.get_mesh_type();
  if ((mesh_type == "LatVolMesh")||(mesh_type == "StructHexVolMesh"))
  {
    fo.set_mesh_type("TetVolMesh");
    fo.set_container_type("vector");
    algotype = "ConvertLatVolToTetVolAlgoT";
  }
  else if (mesh_type == "HexVolMesh")
  {
    fo.set_mesh_type("TetVolMesh");
    fo.set_container_type("vector");
    algotype = "ConvertHexVolToTetVolAlgoT";
  }
  else
  {
    pr->error("No method available for mesh: " + mesh_type);
    return (false);
  }


  ////////////////////////////////////////////////
  // VIRTUAL VERSION OF THIS ALGORITHM

  if (fi.has_virtual_interface() && fo.has_virtual_interface())
  {
    output = Create_Field(fo);
    
    if (algotype == "ConvertLatVolToTetVolAlgoT")
    {
      if (UseScalarInterface(fi,fo)) return (ConvertLatVolToTetVolV<double>(pr,input,output));
      if (UseVectorInterface(fi,fo)) return (ConvertLatVolToTetVolV<Vector>(pr,input,output));
      if (UseTensorInterface(fi,fo)) return (ConvertLatVolToTetVolV<Tensor>(pr,input,output));
    }
    else
    {
      if (UseScalarInterface(fi,fo)) return (ConvertHexVolToTetVolV<double>(pr,input,output));
      if (UseVectorInterface(fi,fo)) return (ConvertHexVolToTetVolV<Vector>(pr,input,output));
      if (UseTensorInterface(fi,fo)) return (ConvertHexVolToTetVolV<Tensor>(pr,input,output));    
    }
  }

  ////////////////////////////////////////////////
  // DYNAMIC COMPILATION VERSION OF THIS ALGORITHM


  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGOConvertMeshToTetVol."+fi.get_field_filename()+"."+fo.get_field_filename()+".",
    "ConvertMeshToTetVolAlgo",algotype,
    fi.get_field_name() + "," + fo.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  fo.fill_compile_info(ci);
  
  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;  
  
  // Handle dynamic compilation
  SCIRun::Handle<ConvertMeshToTetVolAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->ConvertMeshToTetVol(pr,input,output));
}

} // End namespace SCIRunAlgo
