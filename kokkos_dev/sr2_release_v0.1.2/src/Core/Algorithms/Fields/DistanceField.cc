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

#include <Core/Algorithms/Fields/DistanceField.h>

namespace SCIRunAlgo {

using namespace SCIRun;

bool DistanceFieldCellAlgo::DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle objfield, FieldHandle dobjfield)
{
  if (input.get_rep() == 0)
  {
    pr->error("DistanceField: No input field");
    return (false);
  }

  if (objfield.get_rep() == 0)
  {
    pr->error("DistanceField: No object field");
    return (false);
  }

  if (dobjfield.get_rep() == 0)
  {
    pr->error("DistanceField: No object field boundary");
    return (false);
  }

  // no precompiled version available, so compile one

  FieldInformation fi(input);
  FieldInformation fo(input);
  FieldInformation fobj(objfield);
  FieldInformation fdobj(dobjfield);
  
  if (fi.is_nonlinear())
  {
    pr->error("DistanceField: This function has not yet been defined for non-linear elements");
    return (false);
  }
  
  if (!(fobj.is_volume()))
  {
    pr->error("DistanceField: This function is only defined for volume meshes");
    return (false);    
  }
  
  fo.set_data_type("double");
  if (fo.is_nodata()) fo.make_lineardata();
  
  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGODistanceFieldCell."+fi.get_field_filename()+"."+fobj.get_field_filename()+".",
    "DistanceFieldCellAlgo","DistanceFieldCellAlgoT",
    fi.get_field_name() + "," + fo.get_field_name() + "," + fobj.get_field_name() + "," + fdobj.get_field_name() );

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  fo.fill_compile_info(ci);
  fobj.fill_compile_info(ci);
  fdobj.fill_compile_info(ci);
  
  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;

  // Handle dynamic compilation
  SCIRun::Handle<DistanceFieldCellAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->DistanceField(pr,input,output,objfield,dobjfield));
}

bool DistanceFieldFaceAlgo::DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle objfield)
{
  if (input.get_rep() == 0)
  {
    pr->error("DistanceField: No input field");
    return (false);
  }

  if (objfield.get_rep() == 0)
  {
    pr->error("DistanceField: No object field");
    return (false);
  }

  // no precompiled version available, so compile one

  FieldInformation fi(input);
  FieldInformation fo(input);
  FieldInformation fobj(objfield);
  
  if (fi.is_nonlinear())
  {
    pr->error("DistanceField: This function has not yet been defined for non-linear elements");
    return (false);
  }
  
  if (!(fi.is_surface()))
  {
    pr->error("DistanceField: This function is only defined for surface meshes");
    return (false);    
  }

  if (!(fi.is_unstructuredmesh()))
  {
    pr->error("DistanceField: This function is only defined for unstructured surface meshes");
    return (false);    
  }
  
  fo.set_data_type("double");
  if (fo.is_nodata()) fo.make_lineardata();
  
  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGODistanceFieldFace."+fi.get_field_filename()+"."+fobj.get_field_filename()+".",
    "DistanceFieldFaceAlgo","DistanceFieldFaceAlgoT",
    fi.get_field_name() + "," + fo.get_field_name() + "," + fobj.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  fo.fill_compile_info(ci);
  fobj.fill_compile_info(ci);

  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;
  
  // Handle dynamic compilation
  SCIRun::Handle<DistanceFieldFaceAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->DistanceField(pr,input,output,objfield));
}

bool DistanceFieldEdgeAlgo::DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle objfield)
{
  if (input.get_rep() == 0)
  {
    pr->error("DistanceField: No input field");
    return (false);
  }

  if (objfield.get_rep() == 0)
  {
    pr->error("DistanceField: No object field");
    return (false);
  }

  // no precompiled version available, so compile one

  FieldInformation fi(input);
  FieldInformation fo(input);
  FieldInformation fobj(objfield);
  
  if (fi.is_nonlinear())
  {
    pr->error("DistanceField: This function has not yet been defined for non-linear elements");
    return (false);
  }
  
  if (!(fobj.is_curve()))
  {
    pr->error("DistanceField: This function is only defined for curve meshes");
    return (false);    
  }

  fo.set_data_type("double");
  if (fo.is_nodata()) fo.make_lineardata();
  
  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGODistanceFieldEdge."+fi.get_field_filename()+"."+fobj.get_field_filename()+".",
    "DistanceFieldEdgeAlgo","DistanceFieldEdgeAlgoT",
    fi.get_field_name() + "," + fo.get_field_name() + "," + fobj.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  fo.fill_compile_info(ci);
  fobj.fill_compile_info(ci);

  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;
  
  // Handle dynamic compilation
  SCIRun::Handle<DistanceFieldEdgeAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->DistanceField(pr,input,output,objfield));
}

bool DistanceFieldNodeAlgo::DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle objfield)
{
  if (input.get_rep() == 0)
  {
    pr->error("DistanceField: No input field");
    return (false);
  }

  if (objfield.get_rep() == 0)
  {
    pr->error("DistanceField: No object field");
    return (false);
  }

  // no precompiled version available, so compile one

  FieldInformation fi(input);
  FieldInformation fo(input);
  FieldInformation fobj(objfield);
  
  if (fi.is_nonlinear())
  {
    pr->error("DistanceField: This function has not yet been defined for non-linear elements");
    return (false);
  }
  
  if (!(fobj.is_pointcloud()))
  {
    pr->error("DistanceField: This function is only defined for point cloud meshes");
    return (false);    
  }
  
  fo.set_data_type("double");
  if (fo.is_nodata()) fo.make_lineardata();  
  
  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGODistanceFieldNode."+fi.get_field_filename()+"."+fobj.get_field_filename()+".",
    "DistanceFieldNodeAlgo","DistanceFieldNodeAlgoT",
    fi.get_field_name() + "," + fo.get_field_name() + "," + fobj.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  fo.fill_compile_info(ci);
  fobj.fill_compile_info(ci);
  
  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;
    
  // Handle dynamic compilation
  SCIRun::Handle<DistanceFieldNodeAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->DistanceField(pr,input,output,objfield));
}


bool SignedDistanceFieldFaceAlgo::DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle objfield)
{
  if (input.get_rep() == 0)
  {
    pr->error("SignedDistanceField: No input field");
    return (false);
  }

  if (objfield.get_rep() == 0)
  {
    pr->error("SignedDistanceField: No object field");
    return (false);
  }

  // no precompiled version available, so compile one

  FieldInformation fi(input);
  FieldInformation fo(input);
  FieldInformation fobj(objfield);
  
  if (fi.is_nonlinear())
  {
    pr->error("SignedDistanceField: This function has not yet been defined for non-linear elements");
    return (false);
  }
  
  if (!(fi.is_surface()))
  {
    pr->error("SignedDistanceField: This function is only defined for surface meshes");
    return (false);    
  }

  if (!(fi.is_unstructuredmesh()))
  {
    pr->error("SignedDistanceField: This function is only defined for unstructured surface meshes");
    return (false);    
  }
  
  fo.set_data_type("double");
  if (fo.is_nodata()) fo.make_lineardata();
  
  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGOSignedDistanceFieldFace."+fi.get_field_filename()+"."+fobj.get_field_filename()+".",
    "SignedDistanceFieldFaceAlgo","SignedDistanceFieldFaceAlgoT",
    fi.get_field_name() + "," + fo.get_field_name() + "," + fobj.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi.fill_compile_info(ci);
  fo.fill_compile_info(ci);
  fobj.fill_compile_info(ci);
  
  // Handle dynamic compilation
  SCIRun::Handle<SignedDistanceFieldFaceAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->DistanceField(pr,input,output,objfield));
}

} // End namespace SCIRunAlgo
