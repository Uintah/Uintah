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

  if (object->mesh()->dimensionality() > 2)
  {
    error("DistanceToField: This function has only been implemented for a surface mesh");
    return(false);
  }  

  if ((dynamic_cast<TriSurfMesh*>(object->mesh().get_rep())) ||
      (dynamic_cast<QuadSurfMesh*>(object->mesh().get_rep())) || 
      (dynamic_cast<ImageMesh*>(object->mesh().get_rep())) ||       
      (dynamic_cast<StructQuadSurfMesh*>(object->mesh().get_rep())) || 
      (dynamic_cast<CurveMesh*>(object->mesh().get_rep())) ||  
      (dynamic_cast<StructCurveMesh*>(object->mesh().get_rep())) ||   
      (dynamic_cast<ScanlineMesh*>(object->mesh().get_rep())) ||       
      (dynamic_cast<PointCloudMesh*>(object->mesh().get_rep())))   
  {

    Handle<DistanceToFieldAlgo> algo;
    const TypeDescription *ftd = input->get_type_description();
    const TypeDescription *oftd = object->get_type_description();
    
    CompileInfoHandle ci = DistanceToFieldAlgo::get_compile_info(ftd,oftd);
    
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

  if ((dynamic_cast<TriSurfMesh*>(object->mesh().get_rep())) ||
      (dynamic_cast<QuadSurfMesh*>(object->mesh().get_rep())) ||  
      (dynamic_cast<ImageMesh*>(object->mesh().get_rep())) ||
      (dynamic_cast<StructQuadSurfMesh*>(object->mesh().get_rep())) || 
      (dynamic_cast<CurveMesh*>(object->mesh().get_rep())))   
  {  
    Handle<DistanceToFieldAlgo> algo;
    const TypeDescription *ftd = input->get_type_description();
    const TypeDescription *oftd = object->get_type_description();
    
    CompileInfoHandle ci = DistanceToFieldAlgo::get_compile_info(ftd,oftd);
    
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

  if ((dynamic_cast<TriSurfMesh*>(object->mesh().get_rep())) ||
      (dynamic_cast<QuadSurfMesh*>(object->mesh().get_rep())) ||  
      (dynamic_cast<ImageMesh*>(object->mesh().get_rep())) ||
      (dynamic_cast<StructQuadSurfMesh*>(object->mesh().get_rep())) || 
      (dynamic_cast<CurveMesh*>(object->mesh().get_rep())))   
  {  
    Handle<DistanceToFieldAlgo> algo;
    const TypeDescription *ftd = input->get_type_description();
    const TypeDescription *oftd = object->get_type_description();
    
    CompileInfoHandle ci = DistanceToFieldAlgo::get_compile_info(ftd,oftd);
    
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


bool FieldsMath::ChangeFieldBasis(FieldHandle input,FieldHandle& output, MatrixHandle &interpolant, int new_basis_order)
{

  if (input.get_rep() == 0)
  {
    error("ChangeFieldBasis: no input field is given");
    return(false);
  }
  
  int basis_order = input->basis_order();
  
  if (basis_order == new_basis_order)
  {
    return(true);
  }

  // Create a field identical to the input, except for the edits.
  const TypeDescription *fsrctd = input->get_type_description();
  CompileInfoHandle ci = ChangeFieldBasisAlgoCreate::get_compile_info(fsrctd);
  Handle<ChangeFieldBasisAlgoCreate> algo;
  if (!DynamicCompilation::compile(ci, algo, pr_))
  {
    error("ChangeFieldBasis: Could not dynamically compile algorithm");
    DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
    return(false);
  }
  
  output = algo->execute(pr_, input, new_basis_order, interpolant);
  if (output.get_rep() == 0)
  {
    error("ChangeFieldBasis: Dynamically compiled algorithm failed");
    return(false);
  }

  return(true);
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

  std::string actype = datafield->get_type_description(1)->get_name();

  // Integer data cannot be interpolated at this moment
  if (datafield->query_scalar_interface(pr_) != NULL) { actype = "double"; }

  const TypeDescription *iftd = datafield->get_type_description();
  const TypeDescription *iltd = datafield->order_type_description();
  const TypeDescription *oftd = input->get_type_description();
  const TypeDescription *oltd = input->order_type_description();

  CompileInfoHandle ci =
      ApplyMappingMatrixAlgo::get_compile_info(iftd, iltd,oftd, oltd,actype, false);
      
  Handle<ApplyMappingMatrixAlgo> algo;      
  if (!DynamicCompilation::compile(ci, algo,pr_))
  {
    error("ApplyMappingmatrix: Could not compile dynamic function");
    DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
    return(false);
  }
         
  output = algo->execute(datafield, input->mesh(), interpolant,input->basis_order());
  
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
  
  if (srcname == get_type_description((LatVolMesh *)0)->get_name() ||
	srcname == get_type_description((StructHexVolMesh *)0)->get_name())
  {
    dstname = "HexVolField";
  }
  else if (srcname == get_type_description((ImageMesh *)0)->get_name() ||
           srcname == get_type_description((StructQuadSurfMesh *)0)->get_name())
  {
    dstname = "QuadSurfField";
  }  
  else if (srcname == get_type_description((ScanlineMesh *)0)->get_name() ||
           srcname == get_type_description((StructCurveMesh *)0)->get_name())
  {
    dstname = "CurveField";
  }

  if (dstname == "")
  {
    error("Unstructure: Algorithm does not know how to unstructure a field of type " + srcname);
    return(false);
  }
  else
  {
    const TypeDescription *ftd = input->get_type_description();
    CompileInfoHandle ci = SCIRun::UnstructureAlgo::get_compile_info(ftd, dstname);
    Handle<UnstructureAlgo> algo;
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


bool FieldsMath::SetFieldData(FieldHandle input, FieldHandle& output,MatrixHandle data)
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
  
  int svt_flag = 0;
  int matrix_svt_flag = svt_flag;

  if (input->query_scalar_interface(pr_).get_rep())
  {
    svt_flag = 0;
  }
  else if (input->query_vector_interface(pr_).get_rep())
  {
    svt_flag = 1;
  }
  else if (input->query_tensor_interface(pr_).get_rep())
  {
    svt_flag = 2;
  }
  
  if (data->nrows() == 6 || data->ncols() == 6)
  {
    matrix_svt_flag = 3;
  }
  else if (data->nrows() == 9 || data->ncols() == 9)
  {
    matrix_svt_flag = 2;
  }
  else if (data->nrows() == 3 || data->ncols() == 3)
  {
    matrix_svt_flag = 1;
  }
  else if (data->nrows() == 1 || data->ncols() == 1)
  {
    matrix_svt_flag = 0;
  }
  else
  {
    error("SelFieldData: Input matrix row/column size mismatch.");
    error("SelFieldData: Input matrix does not appear to fit in the field.");
    return(false);
  }

  int datasize = 0;
  if (input->basis_order() == 0)
  {
    datasize = numelems;
  }
  if (input->basis_order() == 1)
  {
    datasize = numnodes;
  }


  if (matrix_svt_flag == 3 && datasize == 6)
  {
    if (data->nrows() == 3 || data->ncols() == 3)
    {
      matrix_svt_flag = 1;
    }
    else if (data->nrows() == 1 || data->ncols() == 1)
    {
      matrix_svt_flag = 0;
    }
  }  
  if (matrix_svt_flag == 2 && datasize == 9)
  {
    if (data->nrows() == 3 || data->ncols() == 3)
    {
      matrix_svt_flag = 1;
    }
    else if (data->nrows() == 1 || data->ncols() == 1)
    {
      matrix_svt_flag = 0;
    }
  }
  
  if (matrix_svt_flag == 1 && datasize == 3)
  {
    if (data->nrows() == 1 || data->ncols() == 1)
    {
      matrix_svt_flag = 0;
    }
  }
  
  if ((data->nrows() == 9 || data->nrows() == 6) &&
      (data->ncols() == 9 || data->ncols() == 6))
  {
    std::ostringstream oss;
    oss << "SelFieldData: Input matrix is " << data->nrows() + "x" << data->ncols();
    oss << ".  Using rows or columns as tensors is ambiguous.";
    remark(oss.str());
  }
  else if (data->nrows() == 3 && data->ncols() == 3)
  {
    remark("SetFieldData: Input matrix is 3x3.  Using rows/columns for vectors is ambiguous.");
  }
  
  CompileInfoHandle ci = ManageFieldDataAlgoMesh::get_compile_info(input->get_type_description(),matrix_svt_flag,-1);
  
  Handle<ManageFieldDataAlgoMesh> algo;
  if (!(DynamicCompilation::compile(ci, algo, false, pr_))) 
  {
    error("SetFieldData: Could not dynamically compile algorithm");
    return(false);    
  }
  
  output = algo->execute(pr_, input->mesh(), data);

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
  
  int svt_flag = 0;

  if (input->query_scalar_interface(pr_).get_rep())
  {
    svt_flag = 0;
  }
  else if (input->query_vector_interface(pr_).get_rep())
  {
    svt_flag = 1;
  }
  else if (input->query_tensor_interface(pr_).get_rep())
  {
    svt_flag = 2;
  }

  // Compute output matrix.
  if (input->basis_order() == -1)
  {
    remark("GetFieldData: Input field contains no data, no output was matrix created.");
  }
  else
  {
    CompileInfoHandle ci = ManageFieldDataAlgoField::get_compile_info(input->get_type_description(), svt_flag);
    
    Handle<ManageFieldDataAlgoField> algo;
    if (!DynamicCompilation::compile(ci, algo, true, pr_))
    {
      error("GetFieldData: Could not dynamically compile algorithm");
      return(false);
    }

    int datasize;
    data = algo->execute(input, datasize);
    
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
  
  if (method=="Interpolate")
  {
    if(!((dynamic_cast<SCIRun::LatVolMesh *>(input->mesh().get_rep())) ||
         (dynamic_cast<SCIRun::ImageMesh *>(input->mesh().get_rep())) ||
         (dynamic_cast<SCIRun::CurveMesh *>(input->mesh().get_rep())) ||
         (dynamic_cast<SCIRun::ScanlineMesh *>(input->mesh().get_rep())) ||
         (dynamic_cast<SCIRun::TriSurfMesh *>(input->mesh().get_rep())) ||
         (dynamic_cast<SCIRun::TetVolMesh *>(input->mesh().get_rep()))))
    {
      error("FieldDataNodeToElem: Interpolation for this type of field has not yet been implemented");
      return(false);
    }
  }
  
  if (input->basis_order() < 1)
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
  
  if (input->basis_order() != 0)
  {
    error("FieldDataElemToNode: Input field has no data at the elements");
    return(false);
  }

  if (method=="Interpolate")
  {
    if(!((dynamic_cast<SCIRun::LatVolMesh *>(input->mesh().get_rep())) ||
         (dynamic_cast<SCIRun::ImageMesh *>(input->mesh().get_rep())) ||
         (dynamic_cast<SCIRun::CurveMesh *>(input->mesh().get_rep())) ||
         (dynamic_cast<SCIRun::ScanlineMesh *>(input->mesh().get_rep())) ||
         (dynamic_cast<SCIRun::TriSurfMesh *>(input->mesh().get_rep())) ||
         (dynamic_cast<SCIRun::TetVolMesh *>(input->mesh().get_rep()))))
    {
      error("FieldDataNodeToElem: Interpolation for this type of field has not yet been implemented");
      return(false);
    }
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


} // ModelCreation
