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

#include <Core/Algorithms/Util/FieldInformation.h>

namespace SCIRun {

FieldInformation::FieldInformation(FieldHandle handle)
{
  std::string temp;
  // Get the name of the GenericField class
  // This should give GenericField
  field_type = handle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name();
  field_type_h = handle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_h_file_path(); 
  // Analyze the mesh type
  // This will result in
  // mesh_type, mesh_basis_type, and point_type
  const TypeDescription* mesh_td = handle->get_type_description(Field::MESH_TD_E);
  TypeDescription::td_vec* mesh_sub_td = mesh_td->get_sub_type();
  const TypeDescription* mesh_basis_td = (*mesh_sub_td)[0]; 
  TypeDescription::td_vec* mesh_basis_sub_td = mesh_basis_td->get_sub_type();
  const TypeDescription* point_td = (*mesh_basis_sub_td)[0]; 
  
  temp = mesh_td->get_name(); 
  mesh_type = temp.substr(0,temp.find("<"));
  mesh_type_h = mesh_td->get_h_file_path();
  temp = mesh_basis_td->get_name(); 
  mesh_basis_type = temp.substr(0,temp.find("<"));
  mesh_basis_type_h = mesh_basis_td->get_h_file_path();
  point_type = point_td->get_name();
  point_type_h = point_td->get_h_file_path();
  
  // Analyze the basis type
  
  const TypeDescription* basis_td = handle->get_type_description(Field::BASIS_TD_E);
  TypeDescription::td_vec* basis_sub_td = basis_td->get_sub_type();
  const TypeDescription* data_td = (*basis_sub_td)[0]; 
  
  temp = basis_td->get_name(); 
  basis_type = temp.substr(0,temp.find("<"));
  basis_type_h = basis_td->get_h_file_path();
  data_type = data_td->get_name();
  data_type_h = data_td->get_h_file_path();

  const TypeDescription* container_td = handle->get_type_description(Field::FDATA_TD_E);
  temp = container_td->get_name(); 
  container_type = temp.substr(0,temp.find("<"));
  container_type_h = container_td->get_h_file_path();
}

std::string
FieldInformation::get_field_type()
{
  return(field_type);
}

void
FieldInformation::set_field_type(std::string type)
{
  field_type = type;
  field_type_h = "";
  if (type == "GenericField") field_type_h = "Core/Datatypes/GenericField.h";
  if (type == "MultiLevelField") field_type_h = "Core/Datatypes/MultiLevelField.h";
}

std::string
FieldInformation::get_mesh_type()
{
  return(mesh_type);
}

void
FieldInformation::set_mesh_type(std::string type)
{
  mesh_type = type;
  mesh_type_h = "";
  if (type == "ScanlineMesh")
  { 
    field_type_h = "Core/Datatypes/ScanlineMesh.h";
    if (mesh_basis_type.find("Crv") == std::string::npos)
    {
      if (mesh_basis_type.find("uadraticLgn") != std::string::npos) set_mesh_basis_type("CrvQuadraticLgn");
      else if (mesh_basis_type.find("ubicHmt") != std::string::npos) set_mesh_basis_type("CrvCubicHmt");
      else if (mesh_basis_type.find("NoData") != std::string::npos) set_mesh_basis_type("NoDataBasis");
      else if (mesh_basis_type.find("Constant") != std::string::npos) set_mesh_basis_type("ConstantBasis");
      else set_mesh_basis_type("CrvLinearLgn");      
    }
    if (basis_type.find("Crv") == std::string::npos)
    {
      if (basis_type.find("uadraticLgn") != std::string::npos) set_basis_type("CrvQuadraticLgn");
      else if (basis_type.find("ubicHmt") != std::string::npos) set_basis_type("CrvCubicHmt");
      else if (basis_type.find("NoData") != std::string::npos) set_basis_type("NoDataBasis");
      else if (basis_type.find("Constant") != std::string::npos) set_basis_type("ConstantBasis");
      else set_basis_type("CrvLinearLgn");      
    }
    set_container_type("vector");    
  }
  if (type == "ImageMesh") 
  {
    field_type_h = "Core/Datatypes/ImageMesh.h";
    if (mesh_basis_type.find("Quad") == std::string::npos)
    {
      if (mesh_basis_type.find("uadraticLgn") != std::string::npos) set_mesh_basis_type("QuadBiquadraticLgn");
      else if (mesh_basis_type.find("ubicHmt") != std::string::npos) set_mesh_basis_type("QuadBicubicHmt");
      else if (mesh_basis_type.find("NoData") != std::string::npos) set_mesh_basis_type("NoDataBasis");
      else if (mesh_basis_type.find("Constant") != std::string::npos) set_mesh_basis_type("ConstantBasis");
      else set_mesh_basis_type("QuadBilinearLgn");      
    }
    if (basis_type.find("Quad") == std::string::npos)
    {
      if (basis_type.find("uadraticLgn") != std::string::npos) set_basis_type("QuadBiquadraticLgn");
      else if (basis_type.find("ubicHmt") != std::string::npos) set_basis_type("QuadBicubicHmt");
      else if (basis_type.find("NoData") != std::string::npos) set_basis_type("NoDataBasis");
      else if (basis_type.find("Constant") != std::string::npos) set_basis_type("ConstantBasis");
      else set_basis_type("QuadBilinearLgn");      
    }
    set_container_type("FData2d");    
  }
  if (type == "LatVolMesh") 
  {
    field_type_h = "Core/Datatypes/LatVolMesh.h";
    if (mesh_basis_type.find("Hex") == std::string::npos)
    {
      if (mesh_basis_type.find("uadraticLgn") != std::string::npos) set_mesh_basis_type("HexTriquadraticLgn");
      else if (mesh_basis_type.find("ubicHmt") != std::string::npos) set_mesh_basis_type("HexTricubicHmt");
      else if (mesh_basis_type.find("NoData") != std::string::npos) set_mesh_basis_type("NoDataBasis");
      else if (mesh_basis_type.find("Constant") != std::string::npos) set_mesh_basis_type("ConstantBasis");
      else set_mesh_basis_type("HexTrilinearLgn");      
    }    
    if (basis_type.find("Hex") == std::string::npos)
    {
      if (basis_type.find("uadraticLgn") != std::string::npos) set_basis_type("HexTriquadraticLgn");
      else if (basis_type.find("ubicHmt") != std::string::npos) set_basis_type("HexTricubicHmt");
      else if (basis_type.find("NoData") != std::string::npos) set_basis_type("NoDataBasis");
      else if (basis_type.find("Constant") != std::string::npos) set_basis_type("ConstantBasis");
      else set_basis_type("HexTrilinearLgn");      
    }    
    set_container_type("FData3d");    
  }
  if (type == "MaskedLatVolMesh") 
  {
    field_type_h = "Core/Datatypes/MaskedLatVolMesh.h";
    if (mesh_basis_type.find("Hex") == std::string::npos)
    {
      if (mesh_basis_type.find("uadraticLgn") != std::string::npos) set_mesh_basis_type("HexTriquadraticLgn");
      else if (mesh_basis_type.find("ubicHmt") != std::string::npos) set_mesh_basis_type("HexTricubicHmt");
      else if (mesh_basis_type.find("NoData") != std::string::npos) set_mesh_basis_type("NoDataBasis");
      else if (mesh_basis_type.find("Constant") != std::string::npos) set_mesh_basis_type("ConstantBasis");
      else set_mesh_basis_type("HexTrilinearLgn");      
    }    
    if (basis_type.find("Hex") == std::string::npos)
    {
      if (basis_type.find("uadraticLgn") != std::string::npos) set_basis_type("HexTriquadraticLgn");
      else if (basis_type.find("ubicHmt") != std::string::npos) set_basis_type("HexTricubicHmt");
      else if (basis_type.find("NoData") != std::string::npos) set_basis_type("NoDataBasis");
      else if (basis_type.find("Constant") != std::string::npos) set_basis_type("ConstantBasis");
      else set_basis_type("HexTrilinearLgn");      
    }    
    set_container_type("FData3d");    
  }
  
  if (type == "StructCurveMesh") 
  {
    field_type_h = "Core/Datatypes/StructCurveMesh.h";
    if (mesh_basis_type.find("Crv") == std::string::npos)
    {
      if (mesh_basis_type.find("uadraticLgn") != std::string::npos) set_mesh_basis_type("CrvQuadraticLgn");
      else if (mesh_basis_type.find("ubicHmt") != std::string::npos) set_mesh_basis_type("CrvCubicHmt");
      else if (mesh_basis_type.find("NoData") != std::string::npos) set_mesh_basis_type("NoDataBasis");
      else if (mesh_basis_type.find("Constant") != std::string::npos) set_mesh_basis_type("ConstantBasis");
      else set_mesh_basis_type("CrvLinearLgn");      
    }
    if (basis_type.find("Crv") == std::string::npos)
    {
      if (basis_type.find("uadraticLgn") != std::string::npos) set_basis_type("CrvQuadraticLgn");
      else if (basis_type.find("ubicHmt") != std::string::npos) set_basis_type("CrvCubicHmt");
      else if (basis_type.find("NoData") != std::string::npos) set_basis_type("NoDataBasis");
      else if (basis_type.find("Constant") != std::string::npos) set_basis_type("ConstantBasis");
      else set_basis_type("CrvLinearLgn");      
    }  
    set_container_type("vector");    
  }
  
  if (type == "StructQuadSurfMesh")
  {
    field_type_h = "Core/Datatypes/StructQuadSurfMesh.h";
    if (mesh_basis_type.find("Quad") == std::string::npos)
    {
      if (mesh_basis_type.find("uadraticLgn") != std::string::npos) set_mesh_basis_type("QuadBiquadraticLgn");
      else if (mesh_basis_type.find("ubicHmt") != std::string::npos) set_mesh_basis_type("QuadBicubicHmt");
      else if (mesh_basis_type.find("NoData") != std::string::npos) set_mesh_basis_type("NoDataBasis");
      else if (mesh_basis_type.find("Constant") != std::string::npos) set_mesh_basis_type("ConstantBasis");
      else set_mesh_basis_type("QuadBilinearLgn");      
    }
    if (basis_type.find("Quad") == std::string::npos)
    {
      if (basis_type.find("uadraticLgn") != std::string::npos) set_basis_type("QuadBiquadraticLgn");
      else if (basis_type.find("ubicHmt") != std::string::npos) set_basis_type("QuadBicubicHmt");
      else if (basis_type.find("NoData") != std::string::npos) set_basis_type("NoDataBasis");
      else if (basis_type.find("Constant") != std::string::npos) set_basis_type("ConstantBasis");
      else set_basis_type("QuadBilinearLgn");      
    }  
    set_container_type("FData2d");    
  }
  
  if (type == "StructHexVolMesh") 
  {
    field_type_h = "Core/Datatypes/StructHexVolMesh.h";
    if (mesh_basis_type.find("Hex") == std::string::npos)
    {
      if (mesh_basis_type.find("uadraticLgn") != std::string::npos) set_mesh_basis_type("HexTriquadraticLgn");
      else if (mesh_basis_type.find("ubicHmt") != std::string::npos) set_mesh_basis_type("HexTricubicHmt");
      else if (mesh_basis_type.find("NoData") != std::string::npos) set_mesh_basis_type("NoDataBasis");
      else if (mesh_basis_type.find("Constant") != std::string::npos) set_mesh_basis_type("ConstantBasis");
      else set_mesh_basis_type("HexTrilinearLgn");      
    }    
    if (basis_type.find("Hex") == std::string::npos)
    {
      if (basis_type.find("uadraticLgn") != std::string::npos) set_basis_type("HexTriquadraticLgn");
      else if (basis_type.find("ubicHmt") != std::string::npos) set_basis_type("HexTricubicHmt");
      else if (basis_type.find("NoData") != std::string::npos) set_basis_type("NoDataBasis");
      else if (basis_type.find("Constant") != std::string::npos) set_basis_type("ConstantBasis");
      else set_basis_type("HexTrilinearLgn");      
    }        
    set_container_type("FData3d");    
  }
  
  if (type == "CurveMesh") 
  {
    field_type_h = "Core/Datatypes/CurveMesh.h";
    if (mesh_basis_type.find("Crv") == std::string::npos)
    {
      if (mesh_basis_type.find("uadraticLgn") != std::string::npos) set_mesh_basis_type("CrvQuadraticLgn");
      else if (mesh_basis_type.find("ubicHmt") != std::string::npos) set_mesh_basis_type("CrvCubicHmt");
      else if (mesh_basis_type.find("NoData") != std::string::npos) set_mesh_basis_type("NoDataBasis");
      else if (mesh_basis_type.find("Constant") != std::string::npos) set_mesh_basis_type("ConstantBasis");
      else set_mesh_basis_type("CrvLinearLgn");      
    }
    if (basis_type.find("Crv") == std::string::npos)
    {
      if (basis_type.find("uadraticLgn") != std::string::npos) set_basis_type("CrvQuadraticLgn");
      else if (basis_type.find("ubicHmt") != std::string::npos) set_basis_type("CrvCubicHmt");
      else if (basis_type.find("NoData") != std::string::npos) set_basis_type("NoDataBasis");
      else if (basis_type.find("Constant") != std::string::npos) set_basis_type("ConstantBasis");
      else set_basis_type("CrvLinearLgn");      
    }
    set_container_type("vector");            
  }
  
  if (type == "TriSurfMesh") 
  {
    field_type_h = "Core/Datatypes/TriSurfMesh.h";
    if (mesh_basis_type.find("Tri") == std::string::npos)
    {
      if (mesh_basis_type.find("uadraticLgn") != std::string::npos) set_mesh_basis_type("TriQuadraticLgn");
      else if (mesh_basis_type.find("ubicHmt") != std::string::npos) set_mesh_basis_type("TriCubicHmt");
      else if (mesh_basis_type.find("NoData") != std::string::npos) set_mesh_basis_type("NoDataBasis");
      else if (mesh_basis_type.find("Constant") != std::string::npos) set_mesh_basis_type("ConstantBasis");
      else set_mesh_basis_type("TriLinearLgn");      
    }
    if (basis_type.find("Tri") == std::string::npos)
    {
      if (basis_type.find("uadraticLgn") != std::string::npos) set_basis_type("TriQuadraticLgn");
      else if (basis_type.find("ubicHmt") != std::string::npos) set_basis_type("TriCubicHmt");
      else if (basis_type.find("NoData") != std::string::npos) set_basis_type("NoDataBasis");
      else if (basis_type.find("Constant") != std::string::npos) set_basis_type("ConstantBasis");
      else set_basis_type("TriLinearLgn");      
    }       
    set_container_type("vector");            
  }
  if (type == "QuadSurfMesh")
  {
    field_type_h = "Core/Datatypes/QuadSurfMesh.h";
    if (mesh_basis_type.find("Quad") == std::string::npos)
    {
      if (mesh_basis_type.find("uadraticLgn") != std::string::npos) set_mesh_basis_type("QuadBiquadraticLgn");
      else if (mesh_basis_type.find("ubicHmt") != std::string::npos) set_mesh_basis_type("QuadBicubicHmt");
      else if (mesh_basis_type.find("NoData") != std::string::npos) set_mesh_basis_type("NoDataBasis");
      else if (mesh_basis_type.find("Constant") != std::string::npos) set_mesh_basis_type("ConstantBasis");
      else set_mesh_basis_type("QuadBilinearLgn");      
    }
    if (basis_type.find("Quad") == std::string::npos)
    {
      if (basis_type.find("uadraticLgn") != std::string::npos) set_basis_type("QuadBiquadraticLgn");
      else if (basis_type.find("ubicHmt") != std::string::npos) set_basis_type("QuadBicubicHmt");
      else if (basis_type.find("NoData") != std::string::npos) set_basis_type("NoDataBasis");
      else if (basis_type.find("Constant") != std::string::npos) set_basis_type("ConstantBasis");
      else set_basis_type("QuadBilinearLgn");      
    }
    set_container_type("vector");                
  }
  
  if (type == "TetVolMesh")
  {
    field_type_h = "Core/Datatypes/TetVolMesh.h";
    if (mesh_basis_type.find("Tet") == std::string::npos)
    {
      if (mesh_basis_type.find("uadraticLgn") != std::string::npos) set_mesh_basis_type("TetQuadraticLgn");
      else if (mesh_basis_type.find("ubicHmt") != std::string::npos) set_mesh_basis_type("TetCubicHmt");
      else if (mesh_basis_type.find("NoData") != std::string::npos) set_mesh_basis_type("NoDataBasis");
      else if (mesh_basis_type.find("Constant") != std::string::npos) set_mesh_basis_type("ConstantBasis");
      else set_mesh_basis_type("TetLinearLgn");      
    }
    if (basis_type.find("Tet") == std::string::npos)
    {
      if (basis_type.find("uadraticLgn") != std::string::npos) set_basis_type("TetQuadraticLgn");
      else if (basis_type.find("ubicHmt") != std::string::npos) set_basis_type("TetCubicHmt");
      else if (basis_type.find("NoData") != std::string::npos) set_basis_type("NoDataBasis");
      else if (basis_type.find("Constant") != std::string::npos) set_basis_type("ConstantBasis");
      else set_basis_type("TetLinearLgn");      
    }  
    set_container_type("vector");                
  }
  
  if (type == "PrismVolMesh") 
  {
    field_type_h = "Core/Datatypes/PrismVolMesh.h";
    if (mesh_basis_type.find("Prism") == std::string::npos)
    {
      if (mesh_basis_type.find("uadraticLgn") != std::string::npos) set_mesh_basis_type("PrismQuadraticLgn");
      else if (mesh_basis_type.find("ubicHmt") != std::string::npos) set_mesh_basis_type("PrismCubicHmt");
      else if (mesh_basis_type.find("NoData") != std::string::npos) set_mesh_basis_type("NoDataBasis");
      else if (mesh_basis_type.find("Constant") != std::string::npos) set_mesh_basis_type("ConstantBasis");
      else set_mesh_basis_type("PrismLinearLgn");      
    }
    if (basis_type.find("Prism") == std::string::npos)
    {
      if (basis_type.find("uadraticLgn") != std::string::npos) set_basis_type("PrismQuadraticLgn");
      else if (basis_type.find("ubicHmt") != std::string::npos) set_basis_type("PrismCubicHmt");
      else if (basis_type.find("NoData") != std::string::npos) set_basis_type("NoDataBasis");
      else if (basis_type.find("Constant") != std::string::npos) set_basis_type("ConstantBasis");
      else set_basis_type("PrismLinearLgn");      
    }      
    set_container_type("vector");            
  }
  
  if (type == "HexVolMesh") 
  {
    field_type_h = "Core/Datatypes/HexVolMesh.h";
    if (mesh_basis_type.find("Hex") == std::string::npos)
    {
      if (mesh_basis_type.find("uadraticLgn") != std::string::npos) set_mesh_basis_type("HexTriquadraticLgn");
      else if (mesh_basis_type.find("ubicHmt") != std::string::npos) set_mesh_basis_type("HexTricubicHmt");
      else if (mesh_basis_type.find("NoData") != std::string::npos) set_mesh_basis_type("NoDataBasis");
      else if (mesh_basis_type.find("Constant") != std::string::npos) set_mesh_basis_type("ConstantBasis");
      else set_mesh_basis_type("HexTrilinearLgn");      
    }    
    if (basis_type.find("Hex") == std::string::npos)
    {
      if (basis_type.find("uadraticLgn") != std::string::npos) set_basis_type("HexTriquadraticLgn");
      else if (basis_type.find("ubicHmt") != std::string::npos) set_basis_type("HexTricubicHmt");
      else if (basis_type.find("NoData") != std::string::npos) set_basis_type("NoDataBasis");
      else if (basis_type.find("Constant") != std::string::npos) set_basis_type("ConstantBasis");
      else set_basis_type("HexTrilinearLgn");      
    }        
    set_container_type("vector");            
  }
  
  if (type == "PointCloudMesh") 
  {
    field_type_h = "Core/Datatypes/PointCloudMesh.h";
    
    if (mesh_basis_type.find("NoData") != std::string::npos) set_mesh_basis_type("NoDataBasis");
    else set_mesh_basis_type("ConstantBasis");

    if (basis_type.find("NoData") != std::string::npos) set_basis_type("NoDataBasis");
    else set_basis_type("ConstantBasis");
    set_container_type("vector");            
  }
}

std::string
FieldInformation::get_mesh_basis_type()
{
  return(mesh_basis_type);
}

void
FieldInformation::set_mesh_basis_type(std::string type)
{
  mesh_basis_type = type;
  
  // currently hard coded, due to lack of proper mechanism in the core
  mesh_basis_type_h = "";
  if (type == "NoDataBasis")      mesh_basis_type_h = "Core/Basis/NoData.h";
  if (type == "ConstantBasis")    mesh_basis_type_h = "Core/Basis/Constant.h";
  if (type == "CrvLinearLgn")     mesh_basis_type_h = "Core/Basis/CrvLinearLgn.h";
  if (type == "CrvQuadraticLgn")  mesh_basis_type_h = "Core/Basis/CrvQuadraticLgn.h";
  if (type == "CrvCubicHmt")      mesh_basis_type_h = "Core/Basis/CrvCubicHmt.h";
  if (type == "HexTrilinearLgn")     mesh_basis_type_h = "Core/Basis/HexTrilinearLgn.h";
  if (type == "HexTriquadraticLgn")  mesh_basis_type_h = "Core/Basis/HexTriquadraticLgn.h";
  if (type == "HexTricubicHmt")      mesh_basis_type_h = "Core/Basis/HexTricubicHmt.h";
  if (type == "TetLinearLgn")     mesh_basis_type_h = "Core/Basis/TetLinearLgn.h";
  if (type == "TetQuadraticLgn")  mesh_basis_type_h = "Core/Basis/TetQuadraticLgn.h";
  if (type == "TetCubicHmt")      mesh_basis_type_h = "Core/Basis/TetCubicHmt.h";
  if (type == "TriLinearLgn")     mesh_basis_type_h = "Core/Basis/TriLinearLgn.h";
  if (type == "TriQuadraticLgn")  mesh_basis_type_h = "Core/Basis/TriQuadraticLgn.h";
  if (type == "TriCubicHmt")      mesh_basis_type_h = "Core/Basis/TriCubicHmt.h";
  if (type == "PrismLinearLgn")     mesh_basis_type_h = "Core/Basis/PrismLinearLgn.h";
  if (type == "PrismQuadraticLgn")  mesh_basis_type_h = "Core/Basis/PrismQuadraticLgn.h";
  if (type == "PrismCubicHmt")      mesh_basis_type_h = "Core/Basis/PrismCubicHmt.h";
  if (type == "QuadBilinearLgn")     mesh_basis_type_h = "Core/Basis/QuadBilinearLgn.h";
  if (type == "QuadBiquadraticLgn")  mesh_basis_type_h = "Core/Basis/QuadbiquadraticLgn.h";
  if (type == "QuadBicubicHmt")      mesh_basis_type_h = "Core/Basis/QuadbicubicHmt.h";
}

std::string
FieldInformation::get_point_type()
{
  return(point_type);
}

void
FieldInformation::set_point_type(std::string type)
{
  point_type = type;
  point_type_h = "";
}


std::string
FieldInformation::get_basis_type()
{
  return(basis_type);
}

void
FieldInformation::set_basis_type(std::string type)
{
  basis_type = type;
  basis_type_h = "";
  if (type == "NoDataBasis")      basis_type_h = "Core/Basis/NoData.h";
  if (type == "ConstantBasis")    basis_type_h = "Core/Basis/Constant.h";
  if (type == "CrvLinearLgn")     basis_type_h = "Core/Basis/CrvLinearLgn.h";
  if (type == "CrvQuadraticLgn")  basis_type_h = "Core/Basis/CrvQuadraticLgn.h";
  if (type == "CrvCubicHmt")      basis_type_h = "Core/Basis/CrvCubicHmt.h";
  if (type == "HexTrilinearLgn")     basis_type_h = "Core/Basis/HexTrilinearLgn.h";
  if (type == "HexTriquadraticLgn")  basis_type_h = "Core/Basis/HexTriquadraticLgn.h";
  if (type == "HexTricubicHmt")      basis_type_h = "Core/Basis/HexTricubicHmt.h";
  if (type == "TetLinearLgn")     basis_type_h = "Core/Basis/TetLinearLgn.h";
  if (type == "TetQuadraticLgn")  basis_type_h = "Core/Basis/TetQuadraticLgn.h";
  if (type == "TetCubicHmt")      basis_type_h = "Core/Basis/TetCubicHmt.h";
  if (type == "TriLinearLgn")     basis_type_h = "Core/Basis/TriLinearLgn.h";
  if (type == "TriQuadraticLgn")  basis_type_h = "Core/Basis/TriQuadraticLgn.h";
  if (type == "TriCubicHmt")      basis_type_h = "Core/Basis/TriCubicHmt.h";
  if (type == "PrismLinearLgn")     basis_type_h = "Core/Basis/PrismLinearLgn.h";
  if (type == "PrismQuadraticLgn")  basis_type_h = "Core/Basis/PrismQuadraticLgn.h";
  if (type == "PrismCubicHmt")      basis_type_h = "Core/Basis/PrismCubicHmt.h";
  if (type == "QuadBilinearLgn")     basis_type_h = "Core/Basis/QuadBilinearLgn.h";
  if (type == "QuadBiquadraticLgn")  basis_type_h = "Core/Basis/QuadbiquadraticLgn.h";
  if (type == "QuadBicubicHmt")      basis_type_h = "Core/Basis/QuadbicubicHmt.h";
}


std::string
FieldInformation::get_data_type()
{
  return(data_type);
}

void
FieldInformation::set_data_type(std::string type)
{
  data_type = type;
  data_type_h = "";
  if (type == "Vector") data_type_h = "Core/Geometry/Vector.h";
  if (type == "Tensor") data_type_h = "Core/Geometry/Tensor.h";
}

std::string
FieldInformation::get_container_type()
{
  return(container_type);
}

void
FieldInformation::set_container_type(std::string type)
{
  container_type = type;
  container_type_h = "";
  if (type == "vector") container_type_h = "vector";
  else container_type_h = "Core/Containers/FData.h";
}


std::string
FieldInformation::get_field_name()
{
  // Deal with some SCIRun design flaw
  std::string meshptr = "";
  if ((container_type.find("2d") != std::string::npos)||(container_type.find("3d") != std::string::npos)) 
    meshptr = "," + mesh_type + "<" + mesh_basis_type + "<" + point_type + "> " + "> ";
    
  std::string field_template = field_type + "<" + mesh_type + "<" + 
    mesh_basis_type + "<" + point_type + "> " + "> " + "," +
    basis_type + "<" + data_type + "> " + "," + container_type + "<" +
    data_type + meshptr + "> " + "> ";
    
  return(field_template);
}

std::string
FieldInformation::get_field_filename()
{
  return(DynamicAlgoBase::to_filename(get_field_name()));
}

void
FieldInformation::fill_compile_info(CompileInfoHandle &ci)
{
  if (field_type_h != "") ci->add_field_include(field_type_h);
  if (mesh_type_h != "") ci->add_mesh_include(mesh_type_h);
  if (mesh_basis_type_h != "") ci->add_basis_include(mesh_basis_type_h);
  if (point_type_h != "") ci->add_data_include(point_type_h);
  if (basis_type_h != "") ci->add_basis_include(basis_type_h);
  if (data_type_h != "") ci->add_data_include(data_type_h);
  if (container_type_h != "") ci->add_container_include(container_type_h);
}


bool
FieldInformation::is_isomorphic()
{
  return((mesh_basis_type == basis_type));
}

bool
FieldInformation::is_nonlinear()
{
  return((is_nonlineardata())||(is_nonlinearmesh()));
}

bool
FieldInformation::is_linear()
{
  return((is_lineardata())&&(is_linearmesh()));
}

bool
FieldInformation::is_nodata()
{
  return((basis_type == "NoDataBasis"));
}

bool
FieldInformation::is_constantdata()
{
  return((basis_type == "ConstantBasis"));
}

bool
FieldInformation::is_lineardata()
{
  return((basis_type.find("inear") != std::string::npos));
}

bool
FieldInformation::is_nonlineardata()
{
  return( (basis_type.find("uadratic") != std::string::npos)||
          (basis_type.find("ubicHmt") != std::string::npos));
}

bool
FieldInformation::is_quadraticdata()
{
  return ((basis_type.find("uadratic") != std::string::npos));
}

bool
FieldInformation::is_cubichmtdata()
{
  return ((basis_type.find("ubicHmt") != std::string::npos));
}

bool
FieldInformation::is_constantmesh()
{
  return((mesh_basis_type == "ConstantBasis"));
}

bool
FieldInformation::is_linearmesh()
{
  return((mesh_basis_type.find("inear") != std::string::npos));
}

bool
FieldInformation::is_nonlinearmesh()
{
  return( (mesh_basis_type.find("uadratic") != std::string::npos)||
          (mesh_basis_type.find("ubicHmt") != std::string::npos));
}

bool
FieldInformation::is_quadraticmesh()
{
  return ((mesh_basis_type.find("uadratic") != std::string::npos));
}

bool
FieldInformation::is_cubichmtmesh()
{
  return ((mesh_basis_type.find("ubicHmt") != std::string::npos));
}

bool
FieldInformation::is_tensor()
{
  return ((data_type == "Tensor"));
}

bool
FieldInformation::is_vector()
{
  return ((data_type == "Vector"));
}

bool
FieldInformation::is_scalar()
{
  return((!is_tensor())&&(!is_vector()));
}

bool
FieldInformation::is_double()
{
  return((data_type == "double"));
}

bool
FieldInformation::is_float()
{
  return((data_type == "float"));
}

bool
FieldInformation::is_dvt()
{
  return(is_double()||is_vector()||is_tensor());
}

bool
FieldInformation::is_structuredmesh()
{
  return((mesh_type.find("Struct")!=std::string::npos));
}

bool
FieldInformation::is_regularmesh()
{
  return((mesh_type=="ScanlineMesh")||(mesh_type=="ImageMesh")||(mesh_type=="LatVolMesh"));
}

bool
FieldInformation::is_unstructuredmesh()
{
  return((!is_regularmesh())&&(!is_structuredmesh()));
}

bool
FieldInformation::make_nodata()
{
  set_basis_type("NoDataBasis");
  return (true);
}

bool
FieldInformation::make_constantdata()
{
  set_basis_type("ConstantBasis");
  return (true);
}

bool
FieldInformation::make_lineardata()
{
  if (mesh_type == "ScanlineMesh") set_basis_type("CrvLinearLgn");
  if (mesh_type == "ImageMesh")  set_basis_type("QuadBilinearLgn");
  if (mesh_type == "LatVolMesh")  set_basis_type("HexTrilinearLgn");
  if (mesh_type == "MaskedLatVolMesh")  set_basis_type("HexTrilinearLgn");
  if (mesh_type == "StructCurveMesh") set_basis_type("CrvLinearLgn");
  if (mesh_type == "StructQuadSurfMesh") set_basis_type("QuadBilinearLgn");
  if (mesh_type == "StructHexVolMesh") set_basis_type("HexTrilinearLgn");
  if (mesh_type == "CurveMesh") set_basis_type("CrvLinearLgn");
  if (mesh_type == "TriSurfMesh") set_basis_type("TriLinearLgn");
  if (mesh_type == "QuadSurfMesh") set_basis_type("QuadBilinearLgn");
  if (mesh_type == "TetVolMesh") set_basis_type("TetLinearLgn");
  if (mesh_type == "PrismVolMesh") set_basis_type("PrismLinearLgn");
  if (mesh_type == "HexVolMesh") set_basis_type("HexTrilinearLgn");
  if (mesh_type == "PointCloudMesh") set_basis_type("ConstantBasis");
  return (true);
}

bool
FieldInformation::make_quadraticdata()
{
  if (mesh_type == "ScanlineMesh") set_basis_type("CrvQuadraticLgn");
  if (mesh_type == "ImageMesh")  set_basis_type("QuadBiquadraticLgn");
  if (mesh_type == "LatVolMesh")  set_basis_type("HexTriquadraticLgn");
  if (mesh_type == "MaskedLatVolMesh")  set_basis_type("HexTriquadraticLgn");
  if (mesh_type == "StructCurveMesh") set_basis_type("CrvQuadraticLgn");
  if (mesh_type == "StructQuadSurfMesh") set_basis_type("QuadBiquadraticLgn");
  if (mesh_type == "StructHexVolMesh") set_basis_type("HexTriquadraticLgn");
  if (mesh_type == "CurveMesh") set_basis_type("CrvQuadraticLgn");
  if (mesh_type == "TriSurfMesh") set_basis_type("TriQuadraticLgn");
  if (mesh_type == "QuadSurfMesh") set_basis_type("QuadBilinearLgn");
  if (mesh_type == "TetVolMesh") set_basis_type("TetQuadraticLgn");
  if (mesh_type == "PrismVolMesh") set_basis_type("PrismQuadraticLgn");
  if (mesh_type == "HexVolMesh") set_basis_type("HexTriquadraticLgn");
  if (mesh_type == "PointCloudMesh") set_basis_type("ConstantBasis");
  return (true);
}

bool
FieldInformation::make_cubichmtdata()
{    
  if (mesh_type == "ScanlineMesh") set_basis_type("CrvCubicHmt");
  if (mesh_type == "ImageMesh")  set_basis_type("QuadBicubicHmt");
  if (mesh_type == "LatVolMesh")  set_basis_type("HexTricubicHmt");
  if (mesh_type == "MaskedLatVolMesh")  set_basis_type("HexTriquadraticLgn");
  if (mesh_type == "StructCurveMesh") set_basis_type("CrvCubicHmt");
  if (mesh_type == "StructQuadSurfMesh") set_basis_type("QuadBicubicHmt");
  if (mesh_type == "StructHexVolMesh") set_basis_type("HexTricubicHmt");
  if (mesh_type == "CurveMesh") set_basis_type("CrvCubicHmt");
  if (mesh_type == "TriSurfMesh") set_basis_type("TriCubicHmt");
  if (mesh_type == "QuadSurfMesh") set_basis_type("QuadBicubicHmt");
  if (mesh_type == "TetVolMesh") set_basis_type("TetCubicHmt");
  if (mesh_type == "PrismVolMesh") set_basis_type("PrismCubicHmt");
  if (mesh_type == "HexVolMesh") set_basis_type("HexTricubicHmt");
  if (mesh_type == "PointCloudMesh") set_basis_type("ConstantBasis");
  return (true);
}

bool
FieldInformation::make_scalar()
{
  data_type = "double";
  return (true);
}

bool
FieldInformation::make_float()
{
  data_type = "float";
  return (true);
}

bool
FieldInformation::make_double()
{
  data_type = "double";
  return (true);
}

bool
FieldInformation::make_vector()
{
  data_type = "Vector";
  return (true);
}

bool
FieldInformation::make_tensor()
{
  data_type = "Tensor";
  return (true);
}


bool
FieldInformation::is_pointcloud()
{
  if (mesh_type == "PointCloudMesh") return (true);
  return false;
}

bool
FieldInformation::is_curve()
{
  if ((mesh_type == "CurveMesh")||(mesh_type == "ScanlineMesh")||(mesh_type == "StructCurveMesh")) return (true);
  return false;
}

bool
FieldInformation::is_surface()
{
  if ((mesh_type == "TriSurfMesh")||(mesh_type == "QuadSurfMesh")||(mesh_type == "ImageMesh")||(mesh_type == "StructQuadSurfMesh")) return (true);
  return false;
}

bool
FieldInformation::is_volume()
{
  if ((mesh_type == "TetVolMesh")||(mesh_type == "PrismVolMesh")||
      (mesh_type == "HexVolMesh")||(mesh_type == "LatVolMesh")||
      (mesh_type == "StructHexVolMesh")||(mesh_type == "MaskedLatVolMesh")) return (true);
  return false;  
}

} // end namespace

