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

#include <Packages/ModelCreation/Core/Fields/FieldsMath.h>

// Matrix types
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

#include <Packages/ModelCreation/Core/Fields/ClipBySelectionMask.h>
#include <Packages/ModelCreation/Core/Fields/DistanceToField.h>
#include <Packages/ModelCreation/Core/Fields/FieldDataElemToNode.h>
#include <Packages/ModelCreation/Core/Fields/FieldDataNodeToElem.h>
#include <Packages/ModelCreation/Core/Fields/SplitFieldByElementData.h>
#include <Packages/ModelCreation/Core/Fields/MappingMatrixToField.h>
#include <Packages/ModelCreation/Core/Fields/MergeFields.h>
#include <Packages/ModelCreation/Core/Fields/GetFieldData.h>
#include <Packages/ModelCreation/Core/Fields/SetFieldData.h>

#include <Core/Algorithms/Fields/FieldCount.h>
#include <Dataflow/Modules/Fields/Unstructure.h>
#include <Dataflow/Modules/Fields/FieldBoundary.h>
#include <Dataflow/Modules/Fields/ApplyMappingMatrix.h>
#include <Dataflow/Modules/Fields/ApplyMappingMatrix.h>


namespace ModelCreation {

using namespace SCIRun;

FieldsMath::FieldsMath(Module* module) :
  module_(module)
{
  pr_ = dynamic_cast<ProgressReporter *>(module_);
}

FieldsMath::FieldsMath(ProgressReporter* pr) :
  module_(0), pr_(pr)
{
}

FieldsMath::~FieldsMath() 
{
}

bool FieldsMath::DistanceToField(FieldHandle input, FieldHandle& output, FieldHandle object)
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

/*
  if (object->mesh()->dimensionality() == 3)
  {
    MatrixHandle dummy;
    FieldHandle  objectsurf;
    if(!(FieldBoundary(object,objectsurf,dummy)))
    {
      error("DistanceToField: Getting surface mesh of object failed");
      return(false);
    }
    object = objectsurf;
    isvol = true;
  }
*/

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


bool FieldsMath::SignedDistanceToField(FieldHandle input, FieldHandle& output, FieldHandle object)
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


bool FieldsMath::IsInsideSurfaceField(FieldHandle input, FieldHandle& output, FieldHandle object)
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


bool FieldsMath::ChangeFieldBasis(FieldHandle input,FieldHandle& output, MatrixHandle &interpolant, std::string newbasis)
{
  error("ChangeFieldBasis: algorithm not implemented");
  return(false);
}


bool FieldsMath::ApplyMappingMatrix(FieldHandle input, FieldHandle& output, MatrixHandle interpolant, FieldHandle datafield)
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

bool FieldsMath::ClipFieldBySelectionMask(FieldHandle input, FieldHandle& output, MatrixHandle selmask,MatrixHandle &interpolant)
{

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
    DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
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


bool FieldsMath::Unstructure(FieldHandle input,FieldHandle& output)
{

  if (input.get_rep() == 0)
  {
    error("Unstructure: No input field was given");
    return(false);
  }

  std::string dstname = "";
  std::string srcname = input->mesh()->get_type_description()->get_name();
  SCIRun::TypeDescription::td_vec* tdv = input->mesh()->get_type_description()->get_sub_type();
  (*tdv)[0]->get_name();
  std::string basisname = (*tdv)[0]->get_name();
  
  if ((srcname.find("LatVolMesh") != std::string::npos) || (srcname.find("StructHexVolMesh") != std::string::npos))
  {
    dstname = "HexVolMesh<" + basisname + "<Point> >";
  }
  else if ((srcname.find("ImageMesh") != std::string::npos ) || (srcname.find("StructQuadSurfMesh") != std::string::npos ))
  {
    dstname = "QuadSurfMesh<" + basisname + "<Point> >";
  }  
  else if ((srcname.find("ScanlineMesh")  != std::string::npos) || (srcname.find("StructCurveMesh")  != std::string::npos))
  {
    dstname = "CurveMesh<" + basisname + "<Point> >";
  }

  if (dstname == "")
  {
    error("Unstructure: Algorithm does not know how to unstructure a field of type " + srcname);
    return(false);
  }
  else
  {
    const TypeDescription *ftd = input->get_type_description();
    TypeDescription::td_vec *tdvdata = input->get_type_description(Field::FDATA_TD_E)->get_sub_type();
    std::string dataname = (*tdvdata)[0]->get_name();
    
    SCIRun::CompileInfoHandle ci = SCIRun::UnstructureAlgo::get_compile_info(ftd, 
        dstname,input->get_type_description(Field::BASIS_TD_E)->get_name(),dataname);
        
    Handle<SCIRun::UnstructureAlgo> algo;
    if (!(DynamicCompilation::compile(ci, algo, false, pr_))) 
    {
      error("Unstructure: Could not dynamically compile algorithm");
      return(false);
    }
    
    output = algo->execute(pr_, input);

    if (output.get_rep())
    {
      output->copy_properties(input.get_rep());
    }
  }

  return(true);
}



bool FieldsMath::FieldBoundary(FieldHandle input, FieldHandle& output,MatrixHandle &interpolant)
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


bool FieldsMath::SetFieldData(FieldHandle input, FieldHandle& output,MatrixHandle data, bool keepscalartype)
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

bool FieldsMath::GetFieldInfo(FieldHandle input, int& numnodes, int& numelems)
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


bool FieldsMath::GetFieldData(FieldHandle input, MatrixHandle& data)
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
    if (!DynamicCompilation::compile(ci, algo, true, pr_))
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

bool FieldsMath::FieldDataNodeToElem(FieldHandle input, FieldHandle& output, std::string method)
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

bool FieldsMath::FieldDataElemToNode(FieldHandle input, FieldHandle& output, std::string method)
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


bool FieldsMath::IsClosedSurface(FieldHandle input)
{
  return(true);
}

bool FieldsMath::IsClockWiseSurface(FieldHandle input)
{
  return(true);
}

bool FieldsMath::IsCounterClockWiseSurface(FieldHandle input)
{
  return(!(IsClockWiseSurface(input)));
}


bool FieldsMath::SplitFieldByElementData(FieldHandle input, FieldHandle& output)
{
  if (input.get_rep() == 0)
  {
    error("SplitFieldByElementData: No input field");
    return(false);
  }
  
  if (input->basis_order() != 0)
  {
    error("SplitFieldByElementData: This function only works for data located at the elements");
    return(false);
  }
  
  if (!input->mesh()->is_editable()) 
  {
    FieldHandle temp;
    if(!Unstructure(input,temp))
    {
      error("SplitFieldByElementData: Could not make the mesh editable");
      return(false);
    }
    input = temp;
  }

  Handle<SplitFieldByElementDataAlgo> algo;
  const TypeDescription *ftd = input->get_type_description();
    
  CompileInfoHandle ci = SplitFieldByElementDataAlgo::get_compile_info(ftd);
    
  if (!(DynamicCompilation::compile(ci, algo, false, pr_)))
  {
    error("SplitFieldByElementData: Could not dynamically compile algorithm");
//    DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
    return(false);
  }
    
  if(!(algo->execute(pr_, input, output)))
  {
    error("SplitFieldByElementData: The dynamically compiled function return error");
    return(false);
  }    
  
  return(true);
}

bool FieldsMath::MappingMatrixToField(FieldHandle input, FieldHandle& output, MatrixHandle mappingmatrix)
{
  if (input.get_rep() == 0)
  {
    error("MappingMatrixToField: No input field");
    return(false);
  }

  if (mappingmatrix.get_rep() == 0)
  {
    error("MappingMatrixToField: No input mapping matrix");
    return(false);
  }    
    
  Handle<MappingMatrixToFieldAlgo> algo;
  CompileInfoHandle ci = MappingMatrixToFieldAlgo::get_compile_info(input);
    
  if (!(DynamicCompilation::compile(ci, algo, false, pr_)))
  {
    error("MappingMatrixToField: Could not dynamically compile algorithm");
    DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
    return(false);
  }
    
  if(!(algo->execute(pr_, input, output, mappingmatrix)))
  {
    error("MappingMatrixToField: The dynamically compiled function return error");
    return(false);
  }    
  
  return(true);
}


bool FieldsMath::MergeFields(std::vector<FieldHandle> inputs, FieldHandle& output, double tolerance, bool mergefields, bool forcepointcloud)
{
  if (inputs.size() == 0)
  {
    error("MappingMatrixToField: No input field");
    return(false);
  } 

  if (inputs[0].get_rep() == 0)
  {
    error("MappingMatrixToField: No input field");
    return(false);
  } 

  Handle<MergeFieldsAlgo> algo;
  CompileInfoHandle ci = MergeFieldsAlgo::get_compile_info(inputs[0]);
    
  if (!(DynamicCompilation::compile(ci, algo, false, pr_)))
  {
    error("MergeFields: Could not dynamically compile algorithm");
//    DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
    return(false);
  }
    
  if(!(algo->mergefields(pr_, inputs, output, tolerance, mergefields,forcepointcloud)))
  {
    error("MergeFields: The dynamically compiled function return error");
    return(false);
  }    
  
  return(true);

}



} // ModelCreation
