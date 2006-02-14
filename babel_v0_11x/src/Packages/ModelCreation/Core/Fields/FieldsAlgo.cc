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

#include <Packages/ModelCreation/Core/Fields/FieldsAlgo.h>

// Matrix types
#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>

// Basis classes
#include <Core/Basis/NoData.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Basis/CrvQuadraticLgn.h>
#include <Core/Basis/HexTricubicHmt.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/HexTriquadraticLgn.h>
#include <Core/Basis/PrismCubicHmt.h>
#include <Core/Basis/PrismLinearLgn.h>
#include <Core/Basis/PrismQuadraticLgn.h>
#include <Core/Basis/QuadBicubicHmt.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Basis/QuadBiquadraticLgn.h>
#include <Core/Basis/TetCubicHmt.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/TetQuadraticLgn.h>
#include <Core/Basis/TriCubicHmt.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Basis/TriQuadraticLgn.h>

// Mesh types
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/StructQuadSurfMesh.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/ScanlineMesh.h>
#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/StructCurveMesh.h>
#include <Core/Datatypes/StructHexVolMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/PointCloudMesh.h>

#include <Packages/ModelCreation/Core/Fields/BuildMembraneTable.h>
#include <Packages/ModelCreation/Core/Fields/ClipBySelectionMask.h>
#include <Packages/ModelCreation/Core/Fields/ConvertToTetVol.h>
#include <Packages/ModelCreation/Core/Fields/ConvertToTriSurf.h>
#include <Packages/ModelCreation/Core/Fields/CompartmentBoundary.h>
#include <Packages/ModelCreation/Core/Fields/DistanceToField.h>
#include <Packages/ModelCreation/Core/Fields/FieldDataElemToNode.h>
#include <Packages/ModelCreation/Core/Fields/FieldDataNodeToElem.h>
#include <Packages/ModelCreation/Core/Fields/MappingMatrixToField.h>
#include <Packages/ModelCreation/Core/Fields/MergeFields.h>
#include <Packages/ModelCreation/Core/Fields/GetFieldData.h>
#include <Packages/ModelCreation/Core/Fields/SetFieldData.h>
#include <Packages/ModelCreation/Core/Fields/SplitFieldByElementData.h>
#include <Packages/ModelCreation/Core/Fields/SplitByConnectedRegion.h>
#include <Packages/ModelCreation/Core/Fields/TransformField.h>
#include <Packages/ModelCreation/Core/Fields/ToPointCloud.h>
#include <Packages/ModelCreation/Core/Fields/Unstructure.h>

#include <Core/Algorithms/Fields/FieldCount.h>
#include <Dataflow/Modules/Fields/FieldBoundary.h>
#include <Dataflow/Modules/Fields/ApplyMappingMatrix.h>

#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

using namespace SCIRun;

FieldsAlgo::FieldsAlgo(ProgressReporter* pr) :
  AlgoLibrary(pr)
{
}

bool FieldsAlgo::DistanceToField(FieldHandle input, FieldHandle& output, FieldHandle object)
{

  if (input.get_rep() == 0)
  {
    error("DistanceToField: No input field");
    return(false);
  }
  
  if (object.get_rep() == 0)
  {
    error("DistanceToField: No Object Field is given");
    return(false);
  }
  
  // If the object is a volume, just extract the outer boundary
  // This should speed up the calculation 
  bool isvol = false;

  if (object->mesh()->dimensionality() > 2)
  {
    error("DistanceToField: This function has only been implemented for a surface mesh, a line mesh, or a point cloud");
    return(false);
  }  

  if ((dynamic_cast<TriSurfMesh<TriLinearLgn<Point> > *>(object->mesh().get_rep())) ||
      (dynamic_cast<QuadSurfMesh<QuadBilinearLgn<Point> > *>(object->mesh().get_rep())) || 
      (dynamic_cast<ImageMesh<QuadBilinearLgn<Point> > *>(object->mesh().get_rep())) ||       
      (dynamic_cast<StructQuadSurfMesh<QuadBilinearLgn<Point> > *>(object->mesh().get_rep())) || 
      (dynamic_cast<CurveMesh<CrvLinearLgn<Point> > *>(object->mesh().get_rep())) ||  
      (dynamic_cast<StructCurveMesh<CrvLinearLgn<Point> > *>(object->mesh().get_rep())) ||   
      (dynamic_cast<ScanlineMesh<CrvLinearLgn<Point> > *>(object->mesh().get_rep())) ||       
      (dynamic_cast<PointCloudMesh<ConstantBasis<Point> >*>(object->mesh().get_rep())))   
  {

    Handle<DistanceToFieldAlgo> algo;
    
    CompileInfoHandle ci = DistanceToFieldAlgo::get_compile_info(input,object);
    
    if (!(DynamicCompilation::compile(ci, algo, false, pr_)))
    {
      error("DistanceToField: Could not dynamically compile algorithm");
      DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
      return(false);
    }
    
    if (isvol)
    {
      if(!(algo->execute_unsigned(pr_, input, output, object)))
      {
        error("DistanceToField: The dynamically compiled function return error");
        return(false);
      }    
    }
    else
    {
      if(!(algo->execute(pr_, input, output, object)))
      {
        error("DistanceToField: The dynamically compiled function return error");
        return(false);
      }
    }
    
    return(true);    
  }
  else
  {
    error("DistanceToField: Algorithm for this type of field has not yet been implemented");
    return(false);  
  }
}


bool FieldsAlgo::SignedDistanceToField(FieldHandle input, FieldHandle& output, FieldHandle object)
{

  if (input.get_rep() == 0)
  {
    error("SignedDistanceToField: No input field");
    return(false);
  }
  
  if (object.get_rep() == 0)
  {
    error("SignedDistanceToField: No Object Field is given");
    return(false);
  }
  
  // If the object is a volume, just extract the outer boundary
  // This should speed up the calculation 
  if (object->mesh()->dimensionality() == 3)
  {
    MatrixHandle dummy;
    FieldHandle  objectsurf;
    if(!(FieldBoundary(object,objectsurf,dummy)))
    {
      error("SignedDistanceToField: Getting surface mesh of object failed");
      return(false);
    }
    object = objectsurf;
  }

  if (object->mesh()->dimensionality() != 2)
  {
    error("SignedDistanceToField: This function has only been implemented for a surface mesh");
    return(false);
  }

  if ((dynamic_cast<TriSurfMesh<TriLinearLgn<Point> > *>(object->mesh().get_rep())) ||
      (dynamic_cast<QuadSurfMesh<QuadBilinearLgn<Point> > *>(object->mesh().get_rep())) ||  
      (dynamic_cast<ImageMesh<QuadBilinearLgn<Point> > *>(object->mesh().get_rep())) ||
      (dynamic_cast<StructQuadSurfMesh<QuadBilinearLgn<Point> > *>(object->mesh().get_rep())))
  {  
    Handle<DistanceToFieldAlgo> algo;
    
    CompileInfoHandle ci = DistanceToFieldAlgo::get_compile_info(input,object);
    
    if (!(DynamicCompilation::compile(ci, algo, false, pr_)))
    {
      error("SignedDistanceToField: Could not dynamically compile algorithm");
      DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
      return(false);
    }
    
    if(!(algo->execute_signed(pr_, input, output, object)))
    {
      error("SignedDistanceToField: The dynamically compiled function return error");
      return(false);
    }
    
    return(true);    
  }
  else
  {
    error("SignedDistanceToField: Algorithm for this type of field has not yet been implemented");
    return(false);  
  }
}


bool FieldsAlgo::IsInsideSurfaceField(FieldHandle input, FieldHandle& output, FieldHandle object)
{

  if (input.get_rep() == 0)
  {
    error("IsInsideSurfaceField: No input field");
    return(false);
  }
  
  if (object.get_rep() == 0)
  {
    error("IsInsideSurfaceField: No Object Field is given");
    return(false);
  }
  
  // If the object is a volume, just extract the outer boundary
  // This should speed up the calculation 
  if (object->mesh()->dimensionality() == 3)
  {
    MatrixHandle dummy;
    FieldHandle  objectsurf;
    if(!(FieldBoundary(object,objectsurf,dummy)))
    {
      error("IsInsideSurfaceField: Getting surface mesh of object failed");
      return(false);
    }
    object = objectsurf;
  }

  if (object->mesh()->dimensionality() != 2)
  {
    error("IsInsideSurfaceField: This function has only been implemented for a surface mesh");
    return(false);
  }

  if ((dynamic_cast<TriSurfMesh<TriLinearLgn<Point> > *>(object->mesh().get_rep())) ||
      (dynamic_cast<QuadSurfMesh<QuadBilinearLgn<Point> > *>(object->mesh().get_rep())) ||  
      (dynamic_cast<ImageMesh<QuadBilinearLgn<Point> > *>(object->mesh().get_rep())) ||
      (dynamic_cast<StructQuadSurfMesh<QuadBilinearLgn<Point> > *>(object->mesh().get_rep())))
  {  
    Handle<DistanceToFieldAlgo> algo;
    
    CompileInfoHandle ci = DistanceToFieldAlgo::get_compile_info(input,object);
    
    if (!(DynamicCompilation::compile(ci, algo, false, pr_)))
    {
      error("IsInsideSurfaceField: Could not dynamically compile algorithm");
      DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
      return(false);
    }
    
    if(!(algo->execute_isinside(pr_, input, output, object)))
    {
      error("IsInsideSurfaceField: The dynamically compiled function return error");
      return(false);
    }
    
    return(true);    
  }
  else
  {
    error("IsInsideSurfaceField: Algorithm for this type of field has not yet been implemented");
    return(false);  
  }
}


bool FieldsAlgo::ChangeFieldBasis(FieldHandle input,FieldHandle& output, MatrixHandle &interpolant, std::string newbasis)
{
  error("ChangeFieldBasis: algorithm not implemented");
  return(false);
}


bool FieldsAlgo::ApplyMappingMatrix(FieldHandle input, FieldHandle& output, MatrixHandle interpolant, FieldHandle datafield)
{
  if (input.get_rep() == 0)
  {
    error("ApplyMappingMatrix: No input field is given");
    return(false);  
  }

  if (datafield.get_rep() == 0)
  {
    error("ApplyMappingMatrix: No field with data to be mapped is given");
    return(false);  
  }

  if (interpolant.get_rep() == 0)
  {
    error("ApplyMappingMatrix: No interpolation matrix is given");
    return(false);  
  }


  TypeDescription::td_vec *tdv = input->get_type_description(Field::FDATA_TD_E)->get_sub_type();
  std::string accumtype = (*tdv)[0]->get_name();
  if ((accumtype.find("Vector")!=std::string::npos)&&(accumtype.find("Tensor")!=std::string::npos)) { accumtype = "double"; }
  const std::string oftn = 
    datafield->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() + "<" +
    datafield->get_type_description(Field::MESH_TD_E)->get_name() + ", " +
    datafield->get_type_description(Field::BASIS_TD_E)->get_similar_name(accumtype,0, "<", " >, ") +
    datafield->get_type_description(Field::FDATA_TD_E)->get_similar_name(accumtype,0, "<", " >") + " >";

  CompileInfoHandle ci =
    ApplyMappingMatrixAlgo::get_compile_info(input->get_type_description(),
            input->order_type_description(),datafield->get_type_description(),
            oftn,datafield->order_type_description(),
            input->get_type_description(Field::FDATA_TD_E),accumtype);

  Handle<ApplyMappingMatrixAlgo> algo;      
  if (!DynamicCompilation::compile(ci, algo,pr_))
  {
    error("ApplyMappingmatrix: Could not compile dynamic function");
    DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
    return(false);
  }
         
  output = algo->execute(datafield, input->mesh(), interpolant);
  
  if (output.get_rep() == 0)
  {
    error("ApplyMappingmatrix: Could not dynamically compile algorithm");
    return(false);
  }
 
  return(true);
}

bool FieldsAlgo::ClipFieldBySelectionMask(FieldHandle input, FieldHandle& output, MatrixHandle selmask,MatrixHandle &interpolant)
{
  if (input.get_rep() == 0)
  {
    error("ClipFieldBySelectionMask: No input field is given");
    return(false);  
  }

  if (!input->mesh()->is_editable()) 
  {
    FieldHandle temp;
    if(!Unstructure(input,temp))
    {
      error("ClipFieldBySelectionMask: Could not edit the mesh");
      return(false);
    }
    input = temp;
  }

  Handle<ClipBySelectionMaskAlgo> algo;
  const TypeDescription *ftd = input->get_type_description();
  CompileInfoHandle ci = ClipBySelectionMaskAlgo::get_compile_info(ftd);
  
  if (!DynamicCompilation::compile(ci, algo, false, pr_))
  {
    error("ClipFieldBySelectionMask: Could not compile dynamic function");
    // DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
    return(false);
  }
  
  SelectionMask mask(selmask);
  if (!mask.isvalid())
  {
    error("ClipFieldBySelectionMask: SelectionMask is not valid");
    return(false);
  }
  
  if(!algo->execute(pr_, input, output, selmask, interpolant))
  {
    error("ClipFieldBySelectionMask: The dynamically compiled function return error");
    return(false);
  }
  
  return(true);
}



bool FieldsAlgo::FieldBoundary(FieldHandle input, FieldHandle& output,MatrixHandle &interpolant)
{
  if (input.get_rep() == 0)
  {
    error("FieldBoundary: No input field was given");  
    return(false);
  }
  
  Handle<FieldBoundaryAlgo> algo;
  
  MeshHandle mesh = input->mesh();

  const TypeDescription *mtd = mesh->get_type_description();
  CompileInfoHandle ci = FieldBoundaryAlgo::get_compile_info(mtd);
  if (!(DynamicCompilation::compile(ci, algo,false,pr_)))
  {
    error("FieldBoundary: Could not dynamically compile algorithm");
    return(false);
  }
  
  algo->execute(pr_, mesh, output, interpolant, input->basis_order());

  return(true);
}


bool FieldsAlgo::SetFieldData(FieldHandle input, FieldHandle& output,MatrixHandle data, bool keepscalartype)
{
  int numnodes;  
  int numelems;  

  if (input.get_rep() == 0)
  {
    error("SetFieldData: No input field given");
    return(false);
  }

  if (!(GetFieldInfo(input,numnodes,numelems)))
  {
    error("SetFieldData: Could not query mesh sizes");
    return(false);
  }
  
  if (input.get_rep() == 0)
  {
    error("SetFieldData: No input field was given");
    return(false); 
  }
  
  if (data.get_rep() == 0)
  {
    error("SetFieldData: No input data was given");
    return(false); 
  }
  
   
  SCIRun::CompileInfoHandle ci = SetFieldDataAlgo::get_compile_info(input,data,numnodes,numelems,keepscalartype);
  
  Handle<SetFieldDataAlgo> algo;
  if (!(DynamicCompilation::compile(ci, algo, false, pr_))) 
  {
    error("SetFieldData: Could not dynamically compile algorithm");
    return(false);    
  }
  
  algo->setfielddata(pr_, input, output, data);

  if (output.get_rep() == 0)
  {
    error("SetFieldData: Dynamic algorithm failed");
    return(false);      
  }

  // For whoever still uses the property manager
  // Copy the properties.
  output->copy_properties(input.get_rep());

  return(true);
}

bool FieldsAlgo::GetFieldInfo(FieldHandle input, int& numnodes, int& numelems)
{
  if (input.get_rep() == 0)
  {
    error("GetFieldInfo: No input field given");
    return(false);
  }
  
  const TypeDescription *meshtd = input->mesh()->get_type_description();
  CompileInfoHandle ci = FieldCountAlgorithm::get_compile_info(meshtd);
  Handle<FieldCountAlgorithm> algo;
  if (!(DynamicCompilation::compile(ci, algo, false, pr_)))
  {
    error("GetFieldInfo: Could not dynamically compile algorithm");
    return(false);
  }
  
  algo->execute(input->mesh(),numnodes,numelems);
  return(true);
}


bool FieldsAlgo::GetFieldData(FieldHandle input, MatrixHandle& data)
{
  if (input.get_rep() == 0)
  {
    error("GetFieldData: No input field given");
    return(false);
  }
  
  // Compute output matrix.
  if (input->basis_order() == -1)
  {
    remark("GetFieldData: Input field contains no data, no output was matrix created.");
  }
  else
  {
    CompileInfoHandle ci = GetFieldDataAlgo::get_compile_info(input);
    
    Handle<GetFieldDataAlgo> algo;
    if (!DynamicCompilation::compile(ci, algo, false, pr_))
    {
      error("GetFieldData: Could not dynamically compile algorithm");
      return(false);
    }

    algo->getfielddata(pr_, input, data);
    
    if (data.get_rep() == 0)
    {
      error("GetFieldData: Dynamic algorithm failed");
      return(false);      
    }
  }
  return(true);
}

bool FieldsAlgo::FieldDataNodeToElem(FieldHandle input, FieldHandle& output, std::string method)
{
  if (input.get_rep() == 0)
  {
    error("FieldDataNodeToElem: No input field given");
    return(false);
  }
  
  if (input->basis_order() == 0)
  {
    warning("FieldDataNodeToElem: The data is already located at the elements");
    return(true);
  }
  
   if (input->basis_order() < 0)
  {
    error("FieldDataNodeToElem: Input field has no data on the nodes");
    return(false);
  }
   
  CompileInfoHandle ci = FieldDataNodeToElemAlgo::get_compile_info(input);
  
  Handle<FieldDataNodeToElemAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, false, pr_))
  {
    error("FieldDataNodeToElem: Could not dynamically compile algorithm");
    return(false);
  }

  if(!(algo->execute(pr_,input,output,method)))
  {
    error("FieldDataNodeToElem: Dynamic algorithm failed");
    return(false);
  }

  if (output.get_rep() == 0)
  {
    error("FieldDataNodeToElem: Dynamic algorithm failed");
    return(false);      
  }

  return(true);
}

bool FieldsAlgo::FieldDataElemToNode(FieldHandle input, FieldHandle& output, std::string method)
{
  if (input.get_rep() == 0)
  {
    error("FieldDataElemToNode: No input field given");
    return(false);
  }
  
  
  if (input->basis_order() > 0)
  {
    warning("FieldDataNodeToElem: The data is already located at the nodes");
    return(true);
  }
  
   if (input->basis_order() < 0)
  {
    error("FieldDataNodeToElem: Input field has no data on the nodes");
    return(false);
  }
   
  CompileInfoHandle ci = FieldDataElemToNodeAlgo::get_compile_info(input);
  
  Handle<FieldDataElemToNodeAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, false, pr_))
  {
    error("FieldDataElemToNode: Could not dynamically compile algorithm");
    return(false);
  }

  if(!(algo->execute(pr_,input,output,method)))
  {
    error("FieldDataElemToNode: Dynamic algorithm failed");
    return(false);
  }

  if (output.get_rep() == 0)
  {
    error("FieldDataElemToNode: Dynamic algorithm failed");
    return(false);      
  }

  return(true);
}


bool FieldsAlgo::IsClosedSurface(FieldHandle input)
{
  return(true);
}

bool FieldsAlgo::IsClockWiseSurface(FieldHandle input)
{
  return(true);
}

bool FieldsAlgo::IsCounterClockWiseSurface(FieldHandle input)
{
  return(!(IsClockWiseSurface(input)));
}


// NEWLY CREATED FUNCTIONS:


bool FieldsAlgo::CompartmentBoundary(FieldHandle input,FieldHandle& output, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly)
{
  CompartmentBoundaryAlgo algo;
  return(algo.CompartmentBoundary(pr_,input,output,minrange,maxrange,userange,addouterboundary,innerboundaryonly));
}

bool FieldsAlgo::ConvertToTetVol(FieldHandle input, FieldHandle& output)
{
  ConvertToTetVolAlgo algo;
  return(algo.ConvertToTetVol(pr_,input,output));
}

bool FieldsAlgo::ConvertToTriSurf(FieldHandle input, FieldHandle& output)
{
  ConvertToTriSurfAlgo algo;
  return(algo.ConvertToTriSurf(pr_,input,output));
}

bool FieldsAlgo::MappingMatrixToField(FieldHandle input, FieldHandle& output, MatrixHandle mappingmatrix)
{
  MappingMatrixToFieldAlgo algo;
  return(algo.MappingMatrixToField(pr_,input,output,mappingmatrix));
}

bool FieldsAlgo::MakeEditable(FieldHandle input,FieldHandle& output)
{
  output = input;
  if (!input->mesh()->is_editable()) 
  {
    if(!Unstructure(input,output))
    {
      error("MakeEditable: Could not unstructure the mesh");
      return(false);
    }
  }
  return (true);
}

bool FieldsAlgo::MergeFields(std::vector<FieldHandle> inputs, FieldHandle& output, double tolerance, bool mergefields, bool mergeelements)
{
  for (size_t p = 0; p < inputs.size(); p++) if (!MakeEditable(inputs[0],inputs[0])) return (false);
  MergeFieldsAlgo algo;
  return(algo.MergeFields(pr_,inputs,output,tolerance,mergefields,mergeelements));
}


bool FieldsAlgo::MergeNodes(FieldHandle input, FieldHandle& output, double tolerance, bool mergeelements)
{
  if (MakeEditable(input,input)) return (false);
  
  std::vector<FieldHandle> inputs(1);
  inputs[0] = input;
  
  MergeFieldsAlgo algo;
  return(algo.MergeFields(pr_,inputs,output,tolerance,true,mergeelements));
}


bool FieldsAlgo::SplitFieldByElementData(FieldHandle input, FieldHandle& output)
{
  FieldHandle input_editable;
  if (!MakeEditable(input,input_editable)) return (false);
  SplitFieldByElementDataAlgo algo;
  return(algo.SplitFieldByElementData(pr_,input_editable,output));
}


bool FieldsAlgo::SplitFieldByConnectedRegion(FieldHandle input, std::vector<FieldHandle>& output)
{
  FieldHandle input_editable;
  if (!MakeEditable(input,input_editable)) return (false);
  SplitByConnectedRegionAlgo algo;
  return(algo.SplitByConnectedRegion(pr_,input_editable,output));
}


bool FieldsAlgo::ToPointCloud(FieldHandle input,FieldHandle& output)
{
  ToPointCloudAlgo algo;
  return(algo.ToPointCloud(pr_,input,output));
}


bool FieldsAlgo::TransformField(FieldHandle input,FieldHandle& output,Transform& transform,bool rotatedata)
{
  TransformFieldAlgo algo;
  return(algo.TransformField(pr_,input,output,transform,rotatedata));
}


bool FieldsAlgo::Unstructure(FieldHandle input,FieldHandle& output)
{
  UnstructureAlgo algo;
  return(algo.Unstructure(pr_,input,output));
}


bool FieldsAlgo::BundleToFieldArray(BundleHandle input, std::vector<FieldHandle>& output)
{
  output.resize(input->numFields());
  for (int p=0; p < input->numFields(); p++) output[p] = input->getField(input->getFieldName(p));
  return (true);
}


bool FieldsAlgo::FieldArrayToBundle(std::vector<FieldHandle>& input, BundleHandle output)
{
  output = scinew Bundle();
  if (output.get_rep() == 0)
  {
    error("FieldArrayToBundle: Could not allocate new bundle");
    return (false);
  }
  
  for (size_t p=0; p < input.size(); p++)
  {
    std::ostringstream oss;
    oss << "field" << p; 
    output->setField(oss.str(),input[p]);
  }
  return (true);
}


bool FieldsAlgo::BuildMembraneTable(FieldHandle elementtype, FieldHandle membranemodel, MatrixHandle& membranetable)
{
  BuildMembraneTableAlgo algo;
  return(algo.BuildMembraneTable(pr_,elementtype,membranemodel,membranetable));
}


bool FieldsAlgo::MatrixToField(MatrixHandle input, FieldHandle& output,std::string datalocation)
{
  MatrixHandle mat = dynamic_cast<Matrix *>(input->dense());
  if (mat.get_rep() == 0)
  {
    error("MatrixToField: Could not convert matrix into dense matrix");
    return (false);    
  } 

  int m = mat->ncols();
  int n = mat->nrows();
  double* dataptr = mat->get_data_pointer();
  int k = 0;
  
  if (datalocation == "Node")
  {
    ImageMesh<QuadBilinearLgn<Point> >::handle_type mesh_handle = scinew ImageMesh<QuadBilinearLgn<Point> >(m,n,Point(static_cast<double>(m),0.0,0.0),Point(0.0,static_cast<double>(n),0.0));
    GenericField<ImageMesh<QuadBilinearLgn<Point> >,QuadBilinearLgn<double>, FData2d<double, ImageMesh<QuadBilinearLgn<Point> > > >* field = scinew GenericField<ImageMesh<QuadBilinearLgn<Point> >,QuadBilinearLgn<double>, FData2d<double, ImageMesh<QuadBilinearLgn<Point> > > >(mesh_handle);
    output = dynamic_cast<Field *>(field);
    ImageMesh<QuadBilinearLgn<Point> >::Node::iterator it, it_end;
    mesh_handle->begin(it);
    mesh_handle->end(it_end);
    while (it != it_end)
    {
      field->set_value(dataptr[k++],*it);
      ++it;
    }
  }
  else if (datalocation == "Element")
  {
    ImageMesh<QuadBilinearLgn<Point> >::handle_type mesh_handle = scinew ImageMesh<QuadBilinearLgn<Point> >(m+1,n+1,Point(static_cast<double>(m+1),0.0,0.0),Point(0.0,static_cast<double>(n+1),0.0));
    GenericField<ImageMesh<QuadBilinearLgn<Point> >,ConstantBasis<double>, FData2d<double, ImageMesh<QuadBilinearLgn<Point> > > >* field = scinew GenericField<ImageMesh<QuadBilinearLgn<Point> >,ConstantBasis<double>, FData2d<double, ImageMesh<QuadBilinearLgn<Point> > > >(mesh_handle);
    output = dynamic_cast<Field *>(field);
    ImageMesh<QuadBilinearLgn<Point> >::Elem::iterator it, it_end;
    mesh_handle->begin(it);
    mesh_handle->end(it_end);
    while (it != it_end)
    {
      field->set_value(dataptr[k++],*it);
      ++it;
    }  
  }
  else
  {
    error("MatrixToField: Data location information is not recognized");
    return (false);      
  }
  
  return (true);
}





} // ModelCreation
