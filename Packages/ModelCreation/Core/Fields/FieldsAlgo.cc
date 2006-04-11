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
#include <Core/Datatypes/NrrdData.h>
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

#include <Packages/ModelCreation/Core/Fields/ApplyMappingMatrix.h>
#include <Packages/ModelCreation/Core/Fields/BuildMembraneTable.h>
#include <Packages/ModelCreation/Core/Fields/ClearAndChangeFieldBasis.h>
#include <Packages/ModelCreation/Core/Fields/ClipBySelectionMask.h>
#include <Packages/ModelCreation/Core/Fields/ConvertToTetVol.h>
#include <Packages/ModelCreation/Core/Fields/ConvertToTriSurf.h>
#include <Packages/ModelCreation/Core/Fields/CompartmentBoundary.h>
#include <Packages/ModelCreation/Core/Fields/DistanceToField.h>
#include <Packages/ModelCreation/Core/Fields/DistanceField.h>
#include <Packages/ModelCreation/Core/Fields/FieldDataElemToNode.h>
#include <Packages/ModelCreation/Core/Fields/FieldDataNodeToElem.h>
#include <Packages/ModelCreation/Core/Fields/FieldBoundary.h>
#include <Packages/ModelCreation/Core/Fields/IsInsideField.h>
#include <Packages/ModelCreation/Core/Fields/LinkFieldBoundary.h>
#include <Packages/ModelCreation/Core/Fields/LinkToCompGridByDomain.h>
#include <Packages/ModelCreation/Core/Fields/MappingMatrixToField.h>
#include <Packages/ModelCreation/Core/Fields/MergeFields.h>
#include <Packages/ModelCreation/Core/Fields/NrrdToField.h>
#include <Packages/ModelCreation/Core/Fields/GetFieldData.h>
#include <Packages/ModelCreation/Core/Fields/GetFieldInfo.h>
#include <Packages/ModelCreation/Core/Fields/SetFieldData.h>
#include <Packages/ModelCreation/Core/Fields/ScaleField.h>
#include <Packages/ModelCreation/Core/Fields/SplitFieldByDomain.h>
#include <Packages/ModelCreation/Core/Fields/SplitByConnectedRegion.h>
#include <Packages/ModelCreation/Core/Fields/TransformField.h>
#include <Packages/ModelCreation/Core/Fields/ToPointCloud.h>
#include <Packages/ModelCreation/Core/Fields/Unstructure.h>
#include <Packages/ModelCreation/Core/Fields/TriSurfPhaseFilter.h>


#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

using namespace SCIRun;

FieldsAlgo::FieldsAlgo(ProgressReporter* pr) :
  AlgoLibrary(pr)
{
}


bool FieldsAlgo::ChangeFieldBasis(FieldHandle input,FieldHandle& output, MatrixHandle &interpolant, std::string newbasis)
{
  error("ChangeFieldBasis: algorithm not implemented");
  return(false);
}


bool FieldsAlgo::ApplyMappingMatrix(FieldHandle fsrc,  FieldHandle fdst, FieldHandle& output, MatrixHandle mapping)
{
  ApplyMappingMatrixAlgo algo;
  return(algo.ApplyMappingMatrix(pr_,fsrc,fdst,output,mapping));
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


bool FieldsAlgo::FieldBoundary(FieldHandle input, FieldHandle& output,MatrixHandle& mapping)
{
    FieldBoundaryAlgo algo;
  return(algo.FieldBoundary(pr_,input,output,mapping));
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
  GetFieldInfoAlgo algo;
  return(algo.GetFieldInfo(pr_,input,numnodes,numelems));
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


bool FieldsAlgo::CompartmentBoundary(FieldHandle input,FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly)
{
  CompartmentBoundaryAlgo algo;
  return(algo.CompartmentBoundary(pr_,input,output,DomainLink,minrange,maxrange,userange,addouterboundary,innerboundaryonly));
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

bool FieldsAlgo::IsInsideField(FieldHandle input, FieldHandle& output, FieldHandle objectfield)
{
  IsInsideFieldAlgo algo;
  return(algo.IsInsideField(pr_,input,output,objectfield));
}


bool FieldsAlgo::LinkFieldBoundary(FieldHandle input, MatrixHandle& NodeLink, MatrixHandle& ElemLink, double tol, bool linkx, bool linky, bool linkz)
{
  LinkFieldBoundaryAlgo algo;
  return(algo.LinkFieldBoundary(pr_,input,NodeLink,ElemLink,tol,linkx,linky,linkz));
}

bool FieldsAlgo::LinkToCompGrid(MatrixHandle NodeLink, MatrixHandle& GeomToComp, MatrixHandle& CompToGeom)
{
  if (NodeLink.get_rep() == 0)
  {
    error("LinkToCompGrid: No matrix on input");
    return (false);
  }

  if (!(NodeLink->is_sparse()))
  {
    error("LinkToCompGrid: NodeLink Matrix is not sparse");
    return (false);  
  }

  if (NodeLink->nrows() != NodeLink->ncols())
  {
    error("LinkToCompGrid: NodeLink Matrix needs to be square");
    return (false);      
  }
  
  SparseRowMatrix* spr = dynamic_cast<SparseRowMatrix*>(NodeLink.get_rep());
  int m = spr->ncols();
  int *rows = spr->rows;
  int *cols = spr->columns;
  double *vals = spr->a;
  
  int *rr = scinew int[m+1];
  int *cc = scinew int[m];
  double *vv = scinew double[m];  
  if ((rr == 0)||(cc == 0)||(vv == 0))
  {
    if (rr) delete[] rr;
    if (cc) delete[] cc;
    if (vv) delete[] vv;
    
    error("LinkToCompGrid: Could not allocate memory for sparse matrix");
    return (false);        
  }
  
  for (int r=0; r<m; r++) rr[r] = r;

  for (int r=0; r<m; r++)
  {
    for (int c=rows[r]; c<rows[r+1]; c++)
    {
      if (cols[c] > r) 
      {
        rr[cols[c]] = r;
      }
    }
  }

  for (int r=0; r< m; r++)
  {
    int p = r;
    while (rr[p] != p) p = rr[p];
    rr[r] = p;      
  }

  int k=0;
  for (int r=0; r<m; r++)
  {
    if (rr[r] == r) 
    {
      rr[r] = k++;
    }
    else
    {
      rr[r] = rr[rr[r]];
    }
  }

  for (int r = 0; r < m; r++)
  {
    cc[r] = rr[r];
    rr[r] = r;
    vv[r] = 1.0;
  }
  rr[m] = m; // An extra entry goes on the end of rr.

  spr = scinew SparseRowMatrix(m, k, rr, cc, m, vv);

  if (spr == 0)
  {
    error("LinkToCompGrid: Could build geometry to computational mesh mapping matrix");
    return (false);
  }

  CompToGeom = spr;
  GeomToComp = spr->transpose();

  if ((GeomToComp.get_rep() == 0)||(CompToGeom.get_rep() == 0))
  {
    error("LinkToCompGrid: Could build geometry to computational mesh mapping matrix");
    return (false);
  }
  
  return (true);
}


bool FieldsAlgo::LinkToCompGridByDomain(FieldHandle Geometry, MatrixHandle NodeLink, MatrixHandle& GeomToComp, MatrixHandle& CompToGeom)
{
  LinkToCompGridByDomainAlgo algo;
  return (algo.LinkToCompGridByDomain(pr_,Geometry,NodeLink,GeomToComp,CompToGeom));
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


bool FieldsAlgo::MergeFields(std::vector<FieldHandle> inputs, FieldHandle& output, double tolerance, bool mergefields, bool mergeelements, bool matchvalue)
{
  for (size_t p = 0; p < inputs.size(); p++) if (!MakeEditable(inputs[0],inputs[0])) return (false);
  MergeFieldsAlgo algo;
  return(algo.MergeFields(pr_,inputs,output,tolerance,mergefields,mergeelements,matchvalue));
}


bool FieldsAlgo::MergeNodes(FieldHandle input, FieldHandle& output, double tolerance, bool mergeelements, bool matchvalue)
{
  if (MakeEditable(input,input)) return (false);
  
  std::vector<FieldHandle> inputs(1);
  inputs[0] = input;
  
  MergeFieldsAlgo algo;
  return(algo.MergeFields(pr_,inputs,output,tolerance,true,mergeelements,matchvalue));
}


bool FieldsAlgo::SplitFieldByDomain(FieldHandle input, FieldHandle& output)
{
  FieldHandle input_editable;
  if (!MakeEditable(input,input_editable)) return (false);
  SplitFieldByDomainAlgo algo;
  return(algo.SplitFieldByDomain(pr_,input_editable,output));
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


bool FieldsAlgo::ClearAndChangeFieldBasis(FieldHandle input,FieldHandle& output, std::string newbasis)
{
  ClearAndChangeFieldBasisAlgo algo;
  return(algo.ClearAndChangeFieldBasis(pr_,input,output,newbasis));
}

bool FieldsAlgo::ScaleField(FieldHandle input, FieldHandle& output, double scaledata, double scalemesh)
{
  ScaleFieldAlgo algo;
  return(algo.ScaleField(pr_,input,output,scaledata,scalemesh));
}

bool FieldsAlgo::BundleToFieldArray(BundleHandle input, std::vector<FieldHandle>& output)
{
  output.resize(input->numFields());
  for (int p=0; p < input->numFields(); p++) output[p] = input->getField(input->getFieldName(p));
  return (true);
}


bool FieldsAlgo::FieldArrayToBundle(std::vector<FieldHandle> input, BundleHandle& output)
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



bool FieldsAlgo::NrrdToField(NrrdDataHandle input, FieldHandle& output,std::string datalocation)
{
  if (input.get_rep() == 0)
  {
    error("NrrdToField: No input Nrrd");
    return (false);    
  } 

  Nrrd *nrrd = input->nrrd_;

  if (nrrd == 0)
  {
    error("NrrdToField: NrrdData does not contain Nrrd");
    return (false);      
  }

  NrrdToFieldAlgo algo;

  switch (nrrd->type)
  {
    case nrrdTypeChar : 
      return(algo.NrrdToField<char>(pr_,input,output,datalocation));
    case nrrdTypeUChar : 
      return(algo.NrrdToField<unsigned char>(pr_,input,output,datalocation));
    case nrrdTypeShort : 
      return(algo.NrrdToField<short>(pr_,input,output,datalocation));
    case nrrdTypeUShort : 
      return(algo.NrrdToField<unsigned short>(pr_,input,output,datalocation));              
    case nrrdTypeInt : 
      return(algo.NrrdToField<int>(pr_,input,output,datalocation));
    case nrrdTypeUInt : 
      return(algo.NrrdToField<unsigned int>(pr_,input,output,datalocation));
    case nrrdTypeFloat : 
      return(algo.NrrdToField<float>(pr_,input,output,datalocation));
    case nrrdTypeDouble : 
      return(algo.NrrdToField<double>(pr_,input,output,datalocation));
    default: 
      error("NrrdToField: This datatype is not supported");
      return (false);
  }
}


bool FieldsAlgo::DistanceField(FieldHandle input, FieldHandle& output, FieldHandle object)
{
  if (object->mesh()->dimensionality() == 3)
  {
    FieldHandle dobject;
    MatrixHandle dummy;
    FieldBoundary(object,dobject,dummy);
    if (dobject.get_rep() == 0)
    {
      error("DistanceField: Could not compute field boundary");
      return (false);
    }
    
    DistanceFieldCellAlgo algo;
    return(algo.DistanceField(pr_,input,output,object,dobject));
  }
  else if (object->mesh()->dimensionality() == 2)
  {
    // Some how find_closest_face has not been implemented for other fields
    // THe following will call Unstructure internally
    if(!(MakeEditable(object,object))) return (false);
    DistanceFieldFaceAlgo algo;
    return(algo.DistanceField(pr_,input,output,object));
  }
  else if (object->mesh()->dimensionality() == 1)
  {
    DistanceFieldEdgeAlgo algo;
    return(algo.DistanceField(pr_,input,output,object));  
  }
  else if (object->mesh()->dimensionality() == 0)
  {
    DistanceFieldNodeAlgo algo;
    return(algo.DistanceField(pr_,input,output,object));  
  }
  
  return (false);
}

bool FieldsAlgo::SignedDistanceField(FieldHandle input, FieldHandle& output, FieldHandle object)
{
  if (object->mesh()->dimensionality() == 2)
  {
    if(!(MakeEditable(object,object))) return (false);
    DistanceFieldFaceAlgo algo;
    return(algo.DistanceField(pr_,input,output,object));  
  }
  else
  {
    error("SignedDistanceField: This function is only available for surface meshes");
    return (false);
  }
}

bool FieldsAlgo::TriSurfPhaseFilter(FieldHandle input, FieldHandle& output, FieldHandle& phaseline, FieldHandle& phasepoint)
{
  TriSurfPhaseFilterAlgo algo;
  return(algo.TriSurfPhaseFilter(pr_,input,output,phaseline,phasepoint));  
}


} // ModelCreation
