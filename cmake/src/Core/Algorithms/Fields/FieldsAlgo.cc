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
#include <Core/Algorithms/Fields/GetBoundingBox.h>
#include <Core/Algorithms/Fields/ClearAndChangeFieldBasis.h>
#include <Core/Algorithms/Fields/ClipBySelectionMask.h>
#include <Core/Algorithms/Fields/ConvertToTetVol.h>
#include <Core/Algorithms/Fields/ConvertToTriSurf.h>
#include <Core/Algorithms/Fields/CurrentDensityMapping.h>
#include <Core/Algorithms/Fields/DomainBoundary.h>
#include <Core/Algorithms/Fields/DistanceField.h>
#include <Core/Algorithms/Fields/FieldDataElemToNode.h>
#include <Core/Algorithms/Fields/FieldDataNodeToElem.h>
#include <Core/Algorithms/Fields/FieldBoundary.h>
#include <Core/Algorithms/Fields/FindClosestNodeByValue.h>
#include <Core/Algorithms/Fields/FindClosestNode.h>
#include <Core/Algorithms/Fields/GatherFields.h>
#include <Core/Algorithms/Fields/GetFieldData.h>
#include <Core/Algorithms/Fields/GetFieldDataMinMax.h>
#include <Core/Algorithms/Fields/GetFieldMeasure.h>
#include <Core/Algorithms/Fields/GetFieldInfo.h>
#include <Core/Algorithms/Fields/IsInsideField.h>
#include <Core/Algorithms/Fields/IndicesToData.h>
#include <Core/Algorithms/Fields/LinkFieldBoundary.h>
#include <Core/Algorithms/Fields/LinkToCompGrid.h>
#include <Core/Algorithms/Fields/LinkToCompGridByDomain.h>
#include <Core/Algorithms/Fields/MappingMatrixToField.h>
#include <Core/Algorithms/Fields/Mapping.h>
#include <Core/Algorithms/Fields/MergeFields.h>
#include <Core/Algorithms/Fields/MergeMeshes.h>
#include <Core/Algorithms/Fields/RemoveUnusedNodes.h>
#include <Core/Algorithms/Fields/ScaleField.h>
#include <Core/Algorithms/Fields/SetFieldData.h>
#include <Core/Algorithms/Fields/SplitFieldByDomain.h>
#include <Core/Algorithms/Fields/SplitByConnectedRegion.h>
#include <Core/Algorithms/Fields/TransformField.h>
#include <Core/Algorithms/Fields/ToPointCloud.h>
#include <Core/Algorithms/Fields/Unstructure.h>

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

bool FieldsAlgo::GetBoundingBox(FieldHandle input,  FieldHandle& output)
{
  GetBoundingBoxAlgo algo;
  return(algo.GetBoundingBox(pr_,input,output));
}


bool FieldsAlgo::ClipFieldBySelectionMask(FieldHandle input, FieldHandle& output, MatrixHandle selmask,MatrixHandle &interpolant)
{
  ClipBySelectionMaskAlgo algo;
  return(algo.ClipBySelectionMask(pr_,input,output,selmask,interpolant));
}

bool FieldsAlgo::ClipFieldByField(FieldHandle input, FieldHandle& output, FieldHandle objfield, MatrixHandle &interpolant)
{
  MatrixHandle mask;
  FieldHandle  temp;
  if(!(IsInsideField(input,temp,objfield,"char","constant"))) return (false);
  if(!(GetFieldData(temp,mask))) return (false);
  return(ClipFieldBySelectionMask(input,output,mask,interpolant));
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

bool FieldsAlgo::GetFieldMeasure(FieldHandle input, std::string method, double& measure)
{
  GetFieldMeasureAlgo algo;
  return(algo.GetFieldMeasure(pr_,input,method,measure));
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

bool FieldsAlgo::FindClosestNodeByValue(FieldHandle input, std::vector<unsigned int>& output, FieldHandle& points, double value)
{
  FindClosestNodeByValueAlgo algo;
  return(algo.FindClosestNodeByValue(pr_,input,output,points,value));
}

bool FieldsAlgo::FindClosestNode(FieldHandle input, std::vector<unsigned int>& output, FieldHandle& points)
{
  FindClosestNodeAlgo algo;
  return(algo.FindClosestNode(pr_,input,output,points));
}


bool FieldsAlgo::DomainBoundary(FieldHandle input,FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly, bool noinnerboundary, bool disconnect)
{
  if (disconnect)
  {
    DomainBoundary2Algo algo;
    return(algo.DomainBoundary(pr_,input,output,DomainLink,minrange,maxrange,userange,addouterboundary,innerboundaryonly,noinnerboundary));  
  }
  else
  {
    DomainBoundaryAlgo algo;
    return(algo.DomainBoundary(pr_,input,output,DomainLink,minrange,maxrange,userange,addouterboundary,innerboundaryonly,noinnerboundary));
  }
}


bool FieldsAlgo::IndexedDomainBoundary(FieldHandle input,FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly, bool noinnerboundary, bool disconnect)
{
  if (disconnect)
  {
    DomainBoundary4Algo algo;
    return(algo.DomainBoundary(pr_,input,output,DomainLink,minrange,maxrange,userange,addouterboundary,innerboundaryonly,noinnerboundary));  
  }
  else
  {
    DomainBoundary3Algo algo;
    return(algo.DomainBoundary(pr_,input,output,DomainLink,minrange,maxrange,userange,addouterboundary,innerboundaryonly,noinnerboundary));
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

bool FieldsAlgo::IsInsideField(FieldHandle input, FieldHandle& output, FieldHandle objectfield, std::string output_type, std::string basis_type,bool partial_inside,double outval, double inval)
{
  IsInsideFieldAlgo algo;
  output = 0;
  return(algo.IsInsideField(pr_,input,output,objectfield,inval,outval,output_type,basis_type,partial_inside));
}

bool FieldsAlgo::IsInsideFields(FieldHandle input, FieldHandle& output, std::vector<FieldHandle> objectfields, std::string output_type,std::string basis_type,bool partial_inside,double outval)
{
  IsInsideFieldAlgo algo;
  output = 0;
  for (unsigned int p = 1; p <= objectfields.size(); p++)
  {
    if (!(algo.IsInsideField(pr_, input, output, objectfields[p-1],
                             static_cast<double>(p), outval, output_type,
                             basis_type, partial_inside)))
    {
      output = 0;
      return (false);
    }
  }
  return (true);
}


bool FieldsAlgo::IndicesToData(FieldHandle input, FieldHandle& output, MatrixHandle data)
{
  IndicesToDataAlgo algo;
  return(algo.IndicesToData(pr_,input,output,data));
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


bool FieldsAlgo::CurrentDensityMapping(int numproc, FieldHandle pot, FieldHandle con, FieldHandle dst, FieldHandle& output, std::string mappingmethod,
                       std::string integrationmethod, std::string integrationfilter, bool multiply_with_normal, bool calcnorm)
{
  CurrentDensityMappingAlgo algo;
  return (algo.CurrentDensityMapping(pr_,numproc,pot,con,dst,output,mappingmethod,integrationmethod,integrationfilter,multiply_with_normal,calcnorm));
}

bool FieldsAlgo::CurrentDensityMapping(FieldHandle pot, FieldHandle con, FieldHandle dst,  FieldHandle& output, std::string mappingmethod,
                       std::string integrationmethod, std::string integrationfilter, bool multiply_with_normal, bool calcnorm)
{
  CurrentDensityMappingAlgo algo;
  return (algo.CurrentDensityMapping(pr_,0,pot,con,dst,output,mappingmethod,integrationmethod,integrationfilter,multiply_with_normal,calcnorm));
}


bool FieldsAlgo::ModalMapping(int numproc, FieldHandle src, FieldHandle dst, FieldHandle& output, std::string mappingmethod,
                       std::string integrationmethod, std::string integrationfilter, double def_value)
{
  ModalMappingAlgo algo;
  return (algo.ModalMapping(pr_,numproc,src,dst,output,mappingmethod,integrationmethod,integrationfilter,def_value));
}

bool FieldsAlgo::ModalMapping(FieldHandle src, FieldHandle dst, FieldHandle& output, std::string mappingmethod,
                       std::string integrationmethod, std::string integrationfilter, double def_value)
{
  ModalMappingAlgo algo;
  return (algo.ModalMapping(pr_,0,src,dst,output,mappingmethod,integrationmethod,integrationfilter,def_value));
}


bool FieldsAlgo::GradientModalMapping(int numproc, FieldHandle src, FieldHandle dst, FieldHandle& output, std::string mappingmethod,
                       std::string integrationmethod, std::string integrationfilter, bool calcnorm)
{
  GradientModalMappingAlgo algo;
  return (algo.GradientModalMapping(pr_,numproc,src,dst,output,mappingmethod,integrationmethod,integrationfilter,calcnorm));
}

bool FieldsAlgo::GradientModalMapping(FieldHandle src, FieldHandle dst, FieldHandle& output, std::string mappingmethod,
                       std::string integrationmethod, std::string integrationfilter, bool calcnorm)
{
  GradientModalMappingAlgo algo;
  return (algo.GradientModalMapping(pr_,0,src,dst,output,mappingmethod,integrationmethod,integrationfilter,calcnorm));
}

bool FieldsAlgo::NodalMapping(int numproc, FieldHandle src, FieldHandle dst, FieldHandle& output, std::string mappingmethod, double def_value)
{
  NodalMappingAlgo algo;
  return (algo.NodalMapping(pr_,numproc,src,dst,output,mappingmethod,def_value));
}

bool FieldsAlgo::NodalMapping(FieldHandle src, FieldHandle dst, FieldHandle& output, std::string mappingmethod, double def_value)
{
  NodalMappingAlgo algo;
  return (algo.NodalMapping(pr_,0,src,dst,output,mappingmethod,def_value));
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
  for (size_t p = 0; p < inputs.size(); p++) if (!MakeEditable(inputs[p],inputs[p])) return (false);
  MergeFieldsAlgo algo;
  return(algo.MergeFields(pr_,inputs,output,tolerance,mergefields,mergeelements,matchvalue));
}

bool FieldsAlgo::MergeMeshes(std::vector<FieldHandle> inputs, FieldHandle& output, double tolerance, bool mergefields, bool mergeelements)
{
  for (size_t p = 0; p < inputs.size(); p++) if (!MakeEditable(inputs[p],inputs[p])) return (false);
  MergeMeshesAlgo algo;
  return(algo.MergeMeshes(pr_,inputs,output,tolerance,mergefields,mergeelements));
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

bool FieldsAlgo::ScaleField(FieldHandle input, FieldHandle& output, double scaledata, double scalemesh, bool scale_from_center)
{
  ScaleFieldAlgo algo;
  return(algo.ScaleField(pr_,input,output,scaledata,scalemesh,scale_from_center));
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


bool FieldsAlgo::RemoveUnusedNodes(FieldHandle input, FieldHandle& output)
{
  RemoveUnusedNodesAlgo algo;
  return(algo.RemoveUnusedNodes(pr_,input,output));
}

bool FieldsAlgo::CleanMesh(FieldHandle input, FieldHandle& output, bool removeunusednodes, bool removeunusedelems, bool reorientelems, bool mergenodes, bool mergeelems)
{
  return (false);
}

} // end namespace SCIRunAlgo

