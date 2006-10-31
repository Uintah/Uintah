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

#include <Packages/CardioWaveInterface/Core/Model/BuildMembraneTable.h>

namespace CardioWave {

using namespace SCIRun;

bool BuildMembraneTableAlgo::BuildMembraneTable(ProgressReporter *pr, FieldHandle ElementType, FieldHandle MembraneModel, MatrixHandle CompToGeom, MatrixHandle NodeLink, MatrixHandle ElemLink, MembraneTable& membranetable, MatrixHandle& MappingMatrix)
{
  if (ElementType.get_rep() == 0)
  {
    pr->error("BuildMembraneTable: No element type field");
    return (false);
  }

  if (MembraneModel.get_rep() == 0)
  {
    pr->error("BuildMembraneTable: No membrane model field");
    return (false);
  }

  // no precompiled version available, so compile one

  FieldInformation fi(ElementType);
  FieldInformation fi2(MembraneModel);
  
  if (!(fi.is_constantdata()))
  {
    pr->error("BuildMembraneTable: The element type field needs to have one data value assigned to each element");
    return (false);
  }
  
  if (!fi.is_unstructuredmesh())
  {
    pr->error("BuildMembraneTable: This function is not defined for structured meshes");
    return (false);
  }  

  if (!((fi.is_volume()&&fi2.is_surface())||(fi.is_surface()&&fi2.is_curve())))
  {
    pr->error("BuildMembraneTable: The element type field and the MembraneModelace needs to be a volumea and a surface OR a surface and a curve");
    return (false);
  }  
  
  if (!fi2.is_unstructuredmesh())
  {
    pr->error("BuildMembraneTable: This function is not defined for structured meshes");
    return (false);
  }  

  // Setup dynamic files

  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "BuildMembraneTable."+fi.get_field_filename()+"."+fi2.get_field_filename()+".",
    "BuildMembraneTableAlgo","BuildMembraneTableAlgoT",
    fi.get_field_name()+","+fi2.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("CardioWave");
  ci->add_namespace("SCIRun");

  fi.fill_compile_info(ci);
  fi2.fill_compile_info(ci);
  
  // Handle dynamic compilation
  SCIRun::Handle<BuildMembraneTableAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
 //   SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->BuildMembraneTable(pr,ElementType,MembraneModel,CompToGeom,NodeLink,ElemLink,membranetable,MappingMatrix));
}


} // End namespace ModelCreation
