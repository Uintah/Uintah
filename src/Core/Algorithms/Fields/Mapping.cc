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

#include <Core/Algorithms/Fields/Mapping.h>

namespace SCIRunAlgo {

using namespace SCIRun;

bool NodalMappingAlgo::NodalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle fsrc,
                       FieldHandle fdst, FieldHandle& fout, 
                       std::string mappingmethod, double def_value)
{
  if (fsrc.get_rep() == 0)
  {
    pr->error("NodalMapping: No input source field");
    return (false);
  }

  if (fdst.get_rep() == 0)
  {
    pr->error("NodalMapping: No input destination field");
    return (false);
  }


  // no precompiled version available, so compile one

  FieldInformation fi_src(fsrc);
  FieldInformation fi_dst(fdst);
  FieldInformation fi_out(fdst);
  
  if ((fi_src.is_nonlinear())||(fi_dst.is_nonlinear()))
  {
    pr->error("NodalMapping: This function has not yet been defined for non-linear elements");
    return (false);
  }

  // we need to adapt the output type
  fi_out.set_data_type(fi_src.get_data_type());
  // We are doing nodal mapping
  fi_out.make_lineardata();

  std::string mapping = "";

  if (mappingmethod == "ClosestNodalData")
  {
    if (fi_src.is_constantdata())
    {
      mapping = "ClosestModalData<"+ fi_src.get_field_name() +" >";    
    }
    else
    {
      mapping = "ClosestNodalData<"+ fi_src.get_field_name() +" >";
    }
  }
  else if (mappingmethod == "ClosestInterpolatedData")
  {
    mapping = "ClosestInterpolatedData<"+fi_src.get_field_name() + " >";
    pr->error("NodalMapping: This method has not yet been fully implemented");
    return (false);
  }
  else if (mappingmethod == "InterpolatedData")
  {
    mapping = "InterpolatedData<"+fi_src.get_field_name() + " >";    
  }
  else
  {
    pr->error("NodalMapping: This method has not yet been fully implemented");
    return (false);  
  }
  
  // Setup dynamic files

  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGONodalMapping."+mappingmethod+"."+fi_src.get_field_filename()+"."+fi_dst.get_field_filename()+".",
    "NodalMappingAlgo","NodalMappingAlgoT",
    mapping+ "," + fi_src.get_field_name() + "," + fi_dst.get_field_name() + "," + fi_out.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi_src.fill_compile_info(ci);
  fi_dst.fill_compile_info(ci);
  fi_out.fill_compile_info(ci);
  
  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;
  // Handle dynamic compilation
  
  SCIRun::Handle<NodalMappingAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
//    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->NodalMapping(pr,numproc,fsrc,fdst,fout,mappingmethod,def_value)); 
}


bool ModalMappingAlgo::ModalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle fsrc,
                       FieldHandle fdst, FieldHandle& fout, 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter,
                       double def_value)
{
  if (fsrc.get_rep() == 0)
  {
    pr->error("ModalMapping: No input source field");
    return (false);
  }

  if (fdst.get_rep() == 0)
  {
    pr->error("ModalMapping: No input destination field");
    return (false);
  }


  // no precompiled version available, so compile one

  FieldInformation fi_src(fsrc);
  FieldInformation fi_dst(fdst);
  FieldInformation fi_out(fdst);
  
  if ((fi_src.is_nonlinear())||(fi_dst.is_nonlinear()))
  {
    pr->error("ModalMapping: This function has not yet been defined for non-linear elements");
    return (false);
  }

  // we need to adapt the output type
  fi_out.set_data_type(fi_src.get_data_type());
  // We are doing nodal mapping
  fi_out.make_constantdata();

  std::string mapping = "";

  if (mappingmethod == "ClosestNodalData")
  {
    if (fi_src.is_constantdata())
    {
      mapping = "ClosestModalData<"+ fi_src.get_field_name() +" >";    
    }
    else
    {
      mapping = "ClosestNodalData<"+ fi_src.get_field_name() +" >";
    }
  }
  else if (mappingmethod == "ClosestInterpolatedData")
  {
    mapping = "ClosestInterpolatedData<"+fi_src.get_field_name() + " >";
    pr->error("ModalMapping: This method has not yet been fully implemented");
    return (false);
  }
  else if (mappingmethod == "InterpolatedData")
  {
    mapping = "InterpolatedData<"+fi_src.get_field_name() + " >";    
  }
  else
  {
    pr->error("ModalMapping: This method has not yet been fully implemented");
    return (false);  
  }
  
  if (fi_out.is_pnt_element())
  {
    pr->error("ModalMapping: The output mesh is an point cloud. Use NodalMapping to map onto a pointcloud");
    return (false);
  }
  
  std::string integrator;
 
 
  if (integrationmethod == "Gaussian1")
  {
    if (fi_out.is_crv_element()) integrator = "GaussianIntegration<CrvGaussian1<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_tri_element()) integrator = "GaussianIntegration<TriGaussian1<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_quad_element()) integrator = "GaussianIntegration<QuadGaussian1<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_tet_element()) integrator = "GaussianIntegration<TetGaussian1<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_prism_element()) integrator = "GaussianIntegration<PrismGaussian1<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_hex_element()) integrator = "GaussianIntegration<HexGaussian1<double>,"+fi_out.get_field_name()+" >";
  }
  else if (integrationmethod == "Gaussian2")
  {
    if (fi_out.is_crv_element()) integrator = "GaussianIntegration<CrvGaussian2<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_tri_element()) integrator = "GaussianIntegration<TriGaussian2<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_quad_element()) integrator = "GaussianIntegration<QuadGaussian2<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_tet_element()) integrator = "GaussianIntegration<TetGaussian2<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_prism_element()) integrator = "GaussianIntegration<PrismGaussian2<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_hex_element()) integrator = "GaussianIntegration<HexGaussian2<double>,"+fi_out.get_field_name()+" >";
  }
  else if (integrationmethod == "Gaussian3")
  {
    if (fi_out.is_crv_element()) integrator = "GaussianIntegration<CrvGaussian3<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_tri_element()) integrator = "GaussianIntegration<TriGaussian3<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_quad_element()) integrator = "GaussianIntegration<QuadGaussian3<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_tet_element()) integrator = "GaussianIntegration<TetGaussian3<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_prism_element()) integrator = "GaussianIntegration<PrismGaussian3<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_hex_element()) integrator = "GaussianIntegration<HexGaussian3<double>,"+fi_out.get_field_name()+" >";
  }
  else if (integrationmethod == "Regular1")
  {
    if (fi_out.is_tri_element()||fi_out.is_tet_element()||fi_out.is_prism_element())
    {
      pr->error("No Regular set of nodes has been defined for the output field type. Use Gaussian nodes for now");
      return (false);
    }
    integrator = "RegularIntegration<"+fi_out.get_field_name()+",1>";
  }
  else if (integrationmethod == "Regular2")
  {
    if (fi_out.is_tri_element()||fi_out.is_tet_element()||fi_out.is_prism_element())
    {
      pr->error("No Regular set of nodes has been defined for the output field type. Use Gaussian nodes for now");
      return (false);
    }
    integrator = "RegularIntegration<"+fi_out.get_field_name()+",2>";
  }
  else if (integrationmethod == "Regular3")
  {
    if (fi_out.is_tri_element()||fi_out.is_tet_element()||fi_out.is_prism_element())
    {
      pr->error("No Regular set of nodes has been defined for the output field type. Use Gaussian nodes for now");
      return (false);
    }
    integrator = "RegularIntegration<"+fi_out.get_field_name()+",3>";
  }
  else if (integrationmethod == "Regular4")
  {
    if (fi_out.is_tri_element()||fi_out.is_tet_element()||fi_out.is_prism_element())
    {
      pr->error("No Regular set of nodes has been defined for the output field type. Use Gaussian nodes for now");
      return (false);
    }
    integrator = "RegularIntegration<"+fi_out.get_field_name()+",4>";
  }
  else if (integrationmethod == "Regular5")
  {
    if (fi_out.is_tri_element()||fi_out.is_tet_element()||fi_out.is_prism_element())
    {
      pr->error("No Regular set of nodes has been defined for the output field type. Use Gaussian nodes for now");
      return (false);
    }
    integrator = "RegularIntegration<"+fi_out.get_field_name()+",5>";
  }
  else
  {
    pr->error("Integration/Interpolation sample node definition is unknown");
    return (false);
  }
  
  // Setup dynamic files

  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGOModalMapping."+mappingmethod+"."+integrationmethod+"."+fi_src.get_field_filename()+"."+fi_dst.get_field_filename()+".",
    "ModalMappingAlgo","ModalMappingAlgoT",
    mapping+ "," + integrator + "," + fi_src.get_field_name() + "," + fi_dst.get_field_name() + "," + fi_out.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi_src.fill_compile_info(ci);
  fi_dst.fill_compile_info(ci);
  fi_out.fill_compile_info(ci);
  
  // Add these for the Gaussian integration schemes
  ci->add_basis_include("Core/Basis/CrvLinearLgn.h");
  ci->add_basis_include("Core/Basis/TriLinearLgn.h");
  ci->add_basis_include("Core/Basis/QuadBilinearLgn.h");
  ci->add_basis_include("Core/Basis/TetLinearLgn.h");
  ci->add_basis_include("Core/Basis/PrismLinearLgn.h");
  ci->add_basis_include("Core/Basis/HexTrilinearLgn.h");

  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;
  // Handle dynamic compilation
  
  SCIRun::Handle<ModalMappingAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->ModalMapping(pr,numproc,fsrc,fdst,fout,mappingmethod,integrationmethod,integrationfilter,def_value)); 
}



bool GradientModalMappingAlgo::GradientModalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle fsrc,
                       FieldHandle fdst, FieldHandle& fout, 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter,
                       bool calcnorm)
{
  if (fsrc.get_rep() == 0)
  {
    pr->error("GradientModalMapping: No input source field");
    return (false);
  }

  if (fdst.get_rep() == 0)
  {
    pr->error("GradientModalMapping: No input destination field");
    return (false);
  }


  // no precompiled version available, so compile one

  FieldInformation fi_src(fsrc);
  FieldInformation fi_dst(fdst);
  FieldInformation fi_out(fdst);
  
  if ((fi_src.is_nonlinear())||(fi_dst.is_nonlinear()))
  {
    pr->error("GradientModalMapping: This function has not yet been defined for non-linear elements");
    return (false);
  }

  if (!(fi_src.is_scalar()))
  {
    pr->error("GradientModalMapping: SCIRun does currently not have a class to store gradients of Vectors and Tensors");
    return (false);
  }
  
  if (!(fi_src.is_linearmesh()))
  {
    pr->error("GradientModalMapping: This function calculates the gradient per element. As the element has constant basis. All gradients are zero");
    return (false);
  }
  
  // we need to adapt the output type
  fi_out.set_data_type("Vector");
  
  if (calcnorm) fi_out.set_data_type("double");

  // We are doing nodal mapping
  fi_out.make_constantdata();

  std::string mapping = "";

  if (mappingmethod == "InterpolatedData")
  {
    mapping = "InterpolatedGradient<"+fi_src.get_field_name() + " >";    
  }
  else
  {
    pr->error("GradientModalMapping: This method has not yet been fully implemented");
    return (false);  
  }
  
  if (fi_out.is_pnt_element())
  {
    pr->error("GradientModalMapping: The output mesh is a point cloud, this module has not yet been implemented for pointclouds");
    return (false);
  }
  
  std::string integrator;
 
 
  if (integrationmethod == "Gaussian1")
  {
    if (fi_out.is_crv_element()) integrator = "GaussianIntegration<CrvGaussian1<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_tri_element()) integrator = "GaussianIntegration<TriGaussian1<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_quad_element()) integrator = "GaussianIntegration<QuadGaussian1<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_tet_element()) integrator = "GaussianIntegration<TetGaussian1<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_prism_element()) integrator = "GaussianIntegration<PrismGaussian1<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_hex_element()) integrator = "GaussianIntegration<HexGaussian1<double>,"+fi_out.get_field_name()+" >";
  }
  else if (integrationmethod == "Gaussian2")
  {
    if (fi_out.is_crv_element()) integrator = "GaussianIntegration<CrvGaussian2<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_tri_element()) integrator = "GaussianIntegration<TriGaussian2<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_quad_element()) integrator = "GaussianIntegration<QuadGaussian2<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_tet_element()) integrator = "GaussianIntegration<TetGaussian2<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_prism_element()) integrator = "GaussianIntegration<PrismGaussian2<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_hex_element()) integrator = "GaussianIntegration<HexGaussian2<double>,"+fi_out.get_field_name()+" >";
  }
  else if (integrationmethod == "Gaussian3")
  {
    if (fi_out.is_crv_element()) integrator = "GaussianIntegration<CrvGaussian3<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_tri_element()) integrator = "GaussianIntegration<TriGaussian3<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_quad_element()) integrator = "GaussianIntegration<QuadGaussian3<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_tet_element()) integrator = "GaussianIntegration<TetGaussian3<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_prism_element()) integrator = "GaussianIntegration<PrismGaussian3<double>,"+fi_out.get_field_name()+" >";
    if (fi_out.is_hex_element()) integrator = "GaussianIntegration<HexGaussian3<double>,"+fi_out.get_field_name()+" >";
  }
  else if (integrationmethod == "Regular1")
  {
    if (fi_out.is_tri_element()||fi_out.is_tet_element()||fi_out.is_prism_element())
    {
      pr->error("No Regular set of nodes has been defined for the output field type. Use Gaussian nodes for now");
      return (false);
    }
    integrator = "RegularIntegration<"+fi_out.get_field_name()+",1>";
  }
  else if (integrationmethod == "Regular2")
  {
    if (fi_out.is_tri_element()||fi_out.is_tet_element()||fi_out.is_prism_element())
    {
      pr->error("No Regular set of nodes has been defined for the output field type. Use Gaussian nodes for now");
      return (false);
    }
    integrator = "RegularIntegration<"+fi_out.get_field_name()+",2>";
  }
  else if (integrationmethod == "Regular3")
  {
    if (fi_out.is_tri_element()||fi_out.is_tet_element()||fi_out.is_prism_element())
    {
      pr->error("No Regular set of nodes has been defined for the output field type. Use Gaussian nodes for now");
      return (false);
    }
    integrator = "RegularIntegration<"+fi_out.get_field_name()+",3>";
  }
  else if (integrationmethod == "Regular4")
  {
    if (fi_out.is_tri_element()||fi_out.is_tet_element()||fi_out.is_prism_element())
    {
      pr->error("No Regular set of nodes has been defined for the output field type. Use Gaussian nodes for now");
      return (false);
    }
    integrator = "RegularIntegration<"+fi_out.get_field_name()+",4>";
  }
  else if (integrationmethod == "Regular5")
  {
    if (fi_out.is_tri_element()||fi_out.is_tet_element()||fi_out.is_prism_element())
    {
      pr->error("No Regular set of nodes has been defined for the output field type. Use Gaussian nodes for now");
      return (false);
    }
    integrator = "RegularIntegration<"+fi_out.get_field_name()+",5>";
  }
  else
  {
    pr->error("GradientModalMapping: Integration/Interpolation sample node definition is unknown");
    return (false);
  }
  
  // Setup dynamic files

  std::string algotype = "";
  if (calcnorm) algotype = "Norm";

  if (calcnorm) std::cout << "calcnorm\n";

  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGOGradientModalMapping"+algotype+"."+mappingmethod+"."+integrationmethod+"."+fi_src.get_field_filename()+"."+fi_dst.get_field_filename()+".",
    "GradientModalMappingAlgo","GradientModalMapping"+algotype+"AlgoT",
    mapping+ "," + integrator + "," + fi_src.get_field_name() + "," + fi_dst.get_field_name() + "," + fi_out.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi_src.fill_compile_info(ci);
  fi_dst.fill_compile_info(ci);
  fi_out.fill_compile_info(ci);
  
  // Add these for the Gaussian integration schemes
  ci->add_basis_include("Core/Basis/CrvLinearLgn.h");
  ci->add_basis_include("Core/Basis/TriLinearLgn.h");
  ci->add_basis_include("Core/Basis/QuadBilinearLgn.h");
  ci->add_basis_include("Core/Basis/TetLinearLgn.h");
  ci->add_basis_include("Core/Basis/PrismLinearLgn.h");
  ci->add_basis_include("Core/Basis/HexTrilinearLgn.h");

  if (dynamic_cast<RegressionReporter *>(pr)) ci->keep_library_ = false;
  // Handle dynamic compilation
  
  SCIRun::Handle<GradientModalMappingAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
//    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->GradientModalMapping(pr,numproc,fsrc,fdst,fout,mappingmethod,integrationmethod,integrationfilter,calcnorm)); 
}



} // End namespace SCIRunAlgo

