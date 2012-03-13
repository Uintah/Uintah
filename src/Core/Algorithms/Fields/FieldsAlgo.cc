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

#include <Core/Algorithms/Fields/FieldsAlgo.h>
#include <Core/Algorithms/Fields/ApplyMappingMatrix.h>
#include <Core/Algorithms/Fields/ClearAndChangeFieldBasis.h>
#include <Core/Algorithms/Fields/ClipBySelectionMask.h>
#include <Core/Algorithms/Fields/ConvertToTetVol.h>
#include <Core/Algorithms/Fields/ConvertToTriSurf.h>
#include <Core/Algorithms/Fields/DomainBoundary.h>
#include <Core/Algorithms/Fields/DistanceField.h>
#include <Core/Algorithms/Fields/FieldDataElemToNode.h>
#include <Core/Algorithms/Fields/FieldDataNodeToElem.h>
#include <Core/Algorithms/Fields/FieldBoundary.h>
#include <Core/Algorithms/Fields/GatherFields.h>
#include <Core/Algorithms/Fields/GetFieldData.h>
#include <Core/Algorithms/Fields/GetFieldDataMinMax.h>
#include <Core/Algorithms/Fields/GetFieldInfo.h>
#include <Core/Algorithms/Fields/IsInsideField.h>
#include <Core/Algorithms/Fields/LinkFieldBoundary.h>
#include <Core/Algorithms/Fields/LinkToCompGrid.h>
#include <Core/Algorithms/Fields/LinkToCompGridByDomain.h>
#include <Core/Algorithms/Fields/MappingMatrixToField.h>
#include <Core/Algorithms/Fields/MergeFields.h>
#include <Core/Algorithms/Fields/ScaleField.h>
#include <Core/Algorithms/Fields/SetFieldData.h>
#include <Core/Algorithms/Fields/SplitFieldByDomain.h>
#include <Core/Algorithms/Fields/SplitByConnectedRegion.h>
#include <Core/Algorithms/Fields/TransformField.h>
#include <Core/Algorithms/Fields/ToPointCloud.h>
#include <Core/Algorithms/Fields/Unstructure.h>

#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>

#include <Core/Basis/Constant.h>
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/TriLinearLgn.h>

#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <sgi_stl_warnings_on.h>

namespace SCIRunAlgo {

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

  ClipBySelectionMaskAlgo algo;
  return(algo.ClipBySelectionMask(pr_,input,output,selmask,interpolant));
}


bool FieldsAlgo::FieldBoundary(FieldHandle input, FieldHandle& output,MatrixHandle& mapping)
{
  FieldBoundaryAlgo algo;
  return(algo.FieldBoundary(pr_,input,output,mapping));
}


bool FieldsAlgo::GetFieldInfo(FieldHandle input, int& numnodes, int& numelems)
{
  GetFieldInfoAlgo algo;
  return(algo.GetFieldInfo(pr_,input,numnodes,numelems));
}


bool FieldsAlgo::GetFieldData(FieldHandle input, MatrixHandle& data)
{
  GetFieldDataAlgo algo;
  return(algo.GetFieldData(pr_,input,data));
}


bool FieldsAlgo::GetFieldDataMinMax(FieldHandle input, double& min, double& max)
{
  GetFieldDataMinMaxAlgo algo;
  return(algo.GetFieldDataMinMax(pr_,input,min,max));
}


bool FieldsAlgo::SetFieldData(FieldHandle input, FieldHandle& output, MatrixHandle data, bool keepscalartype)
{
  SetFieldDataAlgo algo;
  return(algo.SetFieldData(pr_,input,output,data,keepscalartype));
}


bool FieldsAlgo::FieldDataNodeToElem(FieldHandle input, FieldHandle& output, std::string method)
{
  FieldDataNodeToElemAlgo algo;
  return(algo.FieldDataNodeToElem(pr_,input,output,method));
}


bool FieldsAlgo::FieldDataElemToNode(FieldHandle input, FieldHandle& output, std::string method)
{
  FieldDataElemToNodeAlgo algo;
  return(algo.FieldDataElemToNode(pr_,input,output,method));
}


bool FieldsAlgo::DomainBoundary(FieldHandle input,FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly, bool disconnect)
{
  if (disconnect)
  {
    DomainBoundary2Algo algo;
    return(algo.DomainBoundary(pr_,input,output,DomainLink,minrange,maxrange,userange,addouterboundary,innerboundaryonly));  
  }
  else
  {
    DomainBoundaryAlgo algo;
    return(algo.DomainBoundary(pr_,input,output,DomainLink,minrange,maxrange,userange,addouterboundary,innerboundaryonly));
  }
}


bool FieldsAlgo::IndexedDomainBoundary(FieldHandle input,FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly, bool disconnect)
{
  if (disconnect)
  {
    DomainBoundary4Algo algo;
    return(algo.DomainBoundary(pr_,input,output,DomainLink,minrange,maxrange,userange,addouterboundary,innerboundaryonly));  
  }
  else
  {
    DomainBoundary3Algo algo;
    return(algo.DomainBoundary(pr_,input,output,DomainLink,minrange,maxrange,userange,addouterboundary,innerboundaryonly));
  }
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
  LinkToCompGridAlgo algo;
  return (!(algo.LinkToCompGrid(pr_, NodeLink, GeomToComp, CompToGeom)));
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


bool FieldsAlgo::GatherFields(std::list<FieldHandle> inputs, FieldHandle& output)
{
  std::list<FieldHandle>::iterator it, it_end;
  it = inputs.begin();
  it_end = inputs.end();
  while (it != it_end)
  {
    if (!MakeEditable(*it,*it)) return (false);
    ++it;
  }
  GatherFieldsAlgo algo;
  return(algo.GatherFields(pr_,inputs,output));
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
    // The following will call Unstructure internally
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
    SignedDistanceFieldFaceAlgo algo;
    return(algo.DistanceField(pr_,input,output,object));  
  }
  else
  {
    error("SignedDistanceField: This function is only available for surface meshes");
    return (false);
  }
}

} // end namespace SCIRunAlgo

using namespace SCIRun;

template<class T>
void
cast_to_mesh_here( void * in_mesh, T *& out_mesh )
{
  out_mesh = dynamic_cast<T*>( (T*)in_mesh );
}

template void cast_to_mesh_here< GenericField<PointCloudMesh<ConstantBasis<Point> >, ConstantBasis<double>, std::vector<double, std::allocator<double> > > >( void *, GenericField<PointCloudMesh<ConstantBasis<Point> >, ConstantBasis<double>, std::vector<double, std::allocator<double> > > *& );

template void cast_to_mesh_here< GenericField<TetVolMesh<TetLinearLgn<Point> >, TetLinearLgn<double>, std::vector<double, std::allocator<double> > > >( void *, GenericField<TetVolMesh<TetLinearLgn<Point> >, TetLinearLgn<double>, std::vector<double, std::allocator<double> > > *& );
template void cast_to_mesh_here< GenericField<TriSurfMesh<TriLinearLgn<Point> >, TriLinearLgn<double>, std::vector<double, std::allocator<double> > > >( void *, GenericField<TriSurfMesh<TriLinearLgn<Point> >, TriLinearLgn<double>, std::vector<double, std::allocator<double> > > *& );
template void cast_to_mesh_here< TriSurfMesh<TriLinearLgn<Point> > >( void *, TriSurfMesh<TriLinearLgn<Point> > *& );

template void cast_to_mesh_here<GenericField<LatVolMesh<HexTrilinearLgn<Point> >, ConstantBasis<float>, FData3d<float, LatVolMesh<HexTrilinearLgn<Point> > > > >(void*, GenericField<LatVolMesh<HexTrilinearLgn<Point> >, ConstantBasis<float>, FData3d<float, LatVolMesh<HexTrilinearLgn<Point> > > >*&);
template void cast_to_mesh_here<GenericField<LatVolMesh<HexTrilinearLgn<Point> >, ConstantBasis<double>, FData3d<double, LatVolMesh<HexTrilinearLgn<Point> > > > >(void*, GenericField<LatVolMesh<HexTrilinearLgn<Point> >, ConstantBasis<double>, FData3d<double, LatVolMesh<HexTrilinearLgn<Point> > > >*&);
template void cast_to_mesh_here<GenericField<LatVolMesh<HexTrilinearLgn<Point> >, ConstantBasis<int>, FData3d<int, LatVolMesh<HexTrilinearLgn<Point> > > > >(void*, GenericField<LatVolMesh<HexTrilinearLgn<Point> >, ConstantBasis<int>, FData3d<int, LatVolMesh<HexTrilinearLgn<Point> > > >*&);


template void cast_to_mesh_here<GenericField<ImageMesh<QuadBilinearLgn<Point> >, ConstantBasis<float>, FData2d<float, ImageMesh<QuadBilinearLgn<Point> > > > >(void*, GenericField<ImageMesh<QuadBilinearLgn<Point> >, ConstantBasis<float>, FData2d<float, ImageMesh<QuadBilinearLgn<Point> > > >*&);
template void cast_to_mesh_here<GenericField<CurveMesh<CrvLinearLgn<Point> >, CrvLinearLgn<double>, std::vector<double, std::allocator<double> > > >(void*, GenericField<CurveMesh<CrvLinearLgn<Point> >, CrvLinearLgn<double>, std::vector<double, std::allocator<double> > >*&);
