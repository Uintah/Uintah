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

#include <Core/Algorithms/Fields/CurrentDensityMapping.h>

namespace SCIRunAlgo {

using namespace SCIRun;


bool CurrentDensityMappingAlgo::CurrentDensityMapping(ProgressReporter *pr,
                       int numproc, FieldHandle pot, FieldHandle con,
                       FieldHandle fdst, FieldHandle& fout, 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter,
                       bool multiply_with_normal,
                       bool calcnorm)
{
  if (calcnorm && multiply_with_normal)
  {
    pr->error("CurrentDensityMapping: Cannot multiply with normal and calculate norm at the same time");
    return (false);
  }

  if (pot.get_rep() == 0)
  {
    pr->error("CurrentDensityMapping: No input potential field");
    return (false);
  }

  if (con.get_rep() == 0)
  {
    pr->error("CurrentDensityMapping: No input conductivity field");
    return (false);
  }

  if (fdst.get_rep() == 0)
  {
    pr->error("CurrentDensityMapping: No input destination field");
    return (false);
  }


  // no precompiled version available, so compile one

  FieldInformation fi_pot(pot);
  FieldInformation fi_con(con);
  FieldInformation fi_dst(fdst);
  FieldInformation fi_out(fdst);
  
  if ((fi_pot.is_nonlinear())||(fi_con.is_nonlinear())||(fi_dst.is_nonlinear()))
  {
    pr->error("CurrentDensityMapping: This function has not yet been defined for non-linear elements");
    return (false);
  }

  if (!(fi_pot.is_scalar()))
  {
    pr->error("CurrentDensityMapping: Potential needs to be a scalar field");
    return (false);
  }

  if (!(fi_pot.is_scalar())&&!(fi_pot.is_tensor()))
  {
    pr->error("CurrentDensityMapping: Conductivity needs to be a scalar or tensor");
    return (false);
  }

  if (fi_pot.is_constantdata())
  {
    pr->error("CurrentDensityMapping: Potential is located on the elements: hence the potential gradient in every element is zero");
    return (false);
  }

  // we need to adapt the output type
  if (multiply_with_normal)
  {
    if (!fi_out.is_surface())
    {
      pr->error("CurrentDensityMapping: Output field is not a surface and thus is the multiplication with the surface normal not available");
      return (false);
    }
    fi_out.set_data_type(fi_pot.get_data_type());  
  }
  else 
  {
    if (!calcnorm)
    {
      fi_out.set_data_type("Vector");
    }
    else
    {
      fi_out.set_data_type("double");
    }
  }
  
  // We are doing modal mapping
  fi_out.make_constantdata();

  // we take in mapping method, but it is not used yet as there is only one method available

  if (fi_out.is_pnt_element())
  {
    pr->error("CurrentDensityMapping: The output mesh is an point cloud. Use NodalMapping to map onto a pointcloud");
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

  std::string algotype = "";
  if (multiply_with_normal)
  {
    algotype = "Normal";
    integrator = "Normal" + integrator;
  }
  else if (calcnorm)
  {
    algotype = "Norm";
  }

  SCIRun::CompileInfoHandle ci = scinew CompileInfo(
    "ALGOCurrentDensity"+algotype+"Mapping."+integrationmethod+"."+fi_pot.get_field_filename()+"."+fi_con.get_field_filename()+"."+fi_dst.get_field_filename()+".",
    "CurrentDensityMappingAlgo","CurrentDensityMapping"+algotype+"AlgoT",
    integrator + "," + fi_pot.get_field_name() +  "," + fi_con.get_field_name() + "," + fi_dst.get_field_name() + "," + fi_out.get_field_name());

  ci->add_include(TypeDescription::cc_to_h(__FILE__));
  ci->add_namespace("SCIRunAlgo");
  ci->add_namespace("SCIRun");
  
  fi_pot.fill_compile_info(ci);
  fi_con.fill_compile_info(ci);
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
  
  SCIRun::Handle<CurrentDensityMappingAlgo> algo;
  if(!(SCIRun::DynamicCompilation::compile(ci,algo,pr)))
  {
    pr->compile_error(ci->filename_);
//    SCIRun::DynamicLoader::scirun_loader().cleanup_failed_compile(ci);  
    return(false);
  }

  return(algo->CurrentDensityMapping(pr,numproc,pot,con,fdst,fout,mappingmethod,integrationmethod,integrationfilter,multiply_with_normal,calcnorm)); 
}


} // End namespace SCIRunAlgo

