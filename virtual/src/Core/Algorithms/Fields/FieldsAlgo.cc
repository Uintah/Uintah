/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
#include <Core/Algorithms/Fields/ConvertMeshToTetVol.h>
#include <Core/Algorithms/Fields/ConvertMeshToTriSurf.h>
#include <Core/Algorithms/Fields/MapCurrentDensityOntoField.h>
#include <Core/Algorithms/Fields/GetDomainBoundary.h>
#include <Core/Algorithms/Fields/DistanceField.h>
#include <Core/Algorithms/Fields/MapFieldDataFromElemToNode.h>
#include <Core/Algorithms/Fields/MapFieldDataFromNodeToElem.h>
#include <Core/Algorithms/Fields/GetFieldBoundary.h>
#include <Core/Algorithms/Fields/FindClosestNodeByValue.h>
#include <Core/Algorithms/Fields/FindClosestNode.h>
#include <Core/Algorithms/Fields/GatherFields.h>
#include <Core/Algorithms/Fields/GetFieldData.h>
#include <Core/Algorithms/Fields/GetFieldDataMinMax.h>
#include <Core/Algorithms/Fields/GetFieldMeasure.h>
#include <Core/Algorithms/Fields/GetFieldInfo.h>
#include <Core/Algorithms/Fields/CalculateIsInsideField.h>
#include <Core/Algorithms/Fields/ConvertIndicesToFieldData.h>
#include <Core/Algorithms/Fields/CalculateLinkBetweenOppositeFieldBoundaries.h>
#include <Core/Algorithms/Fields/CreateLinkBetweenMeshAndCompGrid.h>
#include <Core/Algorithms/Fields/CreateLinkBetweenMeshAndCompGridByDomain.h>
#include <Core/Algorithms/Fields/ConvertMappingMatrixToFieldData.h>
#include <Core/Algorithms/Fields/Mapping.h>
#include <Core/Algorithms/Fields/MergeFields.h>
#include <Core/Algorithms/Fields/MergeMeshes.h>
#include <Core/Algorithms/Fields/RemoveUnusedNodes.h>
#include <Core/Algorithms/Fields/ScaleFieldMeshAndData.h>
#include <Core/Algorithms/Fields/SetFieldData.h>
#include <Core/Algorithms/Fields/SplitAndMergeFieldByDomain.h>
#include <Core/Algorithms/Fields/SplitByConnectedRegion.h>
#include <Core/Algorithms/Fields/TransformMeshWithTransform.h>
#include <Core/Algorithms/Fields/ToPointCloud.h>
#include <Core/Algorithms/Fields/ConvertToUnstructuredMesh.h>

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

bool 
FieldsAlgo::ClipFieldByMesh(FieldHandle input, FieldHandle& output, 
			    FieldHandle objfield, MatrixHandle &interpolant)
{
  MatrixHandle mask;
  FieldHandle  temp;
  if(!(CalculateIsInsideField(input, temp, objfield, "char", "constant"))) {
    return (false);
  }
  if(!(GetFieldData(temp,mask))) return (false);
  return(ClipFieldBySelectionMask(input, output, mask, interpolant));
}


bool FieldsAlgo::GetFieldBoundary(FieldHandle input, FieldHandle& output,MatrixHandle& mapping)
{
  GetFieldBoundaryAlgo algo;
  return(algo.GetFieldBoundary(pr_,input,output,mapping));
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


bool 
FieldsAlgo::MapFieldDataFromNodeToElem(FieldHandle input, FieldHandle& output, 
				       std::string method)
{
  MapFieldDataFromNodeToElemAlgo algo;
  return(algo.MapFieldDataFromNodeToElem(pr_,input,output,method));
}


bool FieldsAlgo::MapFieldDataFromElemToNode(FieldHandle input, FieldHandle& output, std::string method)
{
  MapFieldDataFromElemToNodeAlgo algo;
  return(algo.MapFieldDataFromElemToNode(pr_,input,output,method));
}

bool FieldsAlgo::FindClosestNodeByValue(FieldHandle input, 
					std::vector<unsigned int>& output, 
					FieldHandle& points, double value)
{
  FindClosestNodeByValueAlgo algo;
  return(algo.FindClosestNodeByValue(pr_,input,output,points,value));
}

bool FieldsAlgo::FindClosestNode(FieldHandle input, 
				 std::vector<unsigned int>& output, 
				 FieldHandle& points)
{
  FindClosestNodeAlgo algo;
  return(algo.FindClosestNode(pr_,input,output,points));
}


bool 
FieldsAlgo::GetDomainBoundary(FieldHandle input,FieldHandle& output, 
			      MatrixHandle DomainLink, double minrange, 
			      double maxrange, bool userange, 
			      bool addouterboundary, bool innerboundaryonly, 
			      bool noinnerboundary, bool disconnect)
{
  if (disconnect)
  {
    GetDomainBoundary2Algo algo;
    return(algo.GetDomainBoundary(pr_, input, output, DomainLink, minrange,
				  maxrange, userange, addouterboundary, 
				  innerboundaryonly, noinnerboundary));  
  }
  else
  {
    GetDomainBoundaryAlgo algo;
    return(algo.GetDomainBoundary(pr_, input, output, DomainLink, minrange,
				  maxrange, userange, addouterboundary,
				  innerboundaryonly, noinnerboundary));
  }
}


bool 
FieldsAlgo::ConvertMeshToTetVol(FieldHandle input, FieldHandle& output)
{
  ConvertMeshToTetVolAlgo algo;
  return(algo.ConvertMeshToTetVol(pr_, input, output));
}


bool 
FieldsAlgo::ConvertMeshToTriSurf(FieldHandle input, FieldHandle& output)
{
  ConvertMeshToTriSurfAlgo algo;
  return(algo.ConvertMeshToTriSurf(pr_, input, output));
}

bool 
FieldsAlgo::CalculateIsInsideField(FieldHandle input, FieldHandle& output, 
				   FieldHandle objectfield, 
				   std::string output_type, 
				   std::string basis_type,
				   bool partial_inside,
				   double outval, double inval)
{
  CalculateIsInsideFieldAlgo algo;
  output = 0;
  return(algo.CalculateIsInsideField(pr_, input, output, objectfield,
				     inval, outval, output_type, 
				     basis_type, partial_inside));
}

bool 
FieldsAlgo::CalculateInsideWhichField(FieldHandle input, 
				      FieldHandle& output, 
				      std::vector<FieldHandle> objectfields, 
				      std::string output_type,
				      std::string basis_type,
				      bool partial_inside,
				      double outval)
{
  CalculateIsInsideFieldAlgo algo;
  output = 0;
  for (unsigned int p = 1; p <= objectfields.size(); p++)
  {
    if (!(algo.CalculateIsInsideField(pr_, input, output, objectfields[p-1],
				      static_cast<double>(p), outval, 
				      output_type,
				      basis_type, partial_inside)))
    {
      output = 0;
      return (false);
    }
  }
  return (true);
}


bool 
FieldsAlgo::ConvertIndicesToFieldData(FieldHandle input, FieldHandle& output, 
				      MatrixHandle data)
{
  ConvertIndicesToFieldDataAlgo algo;
  return(algo.ConvertIndicesToFieldData(pr_, input, output, data));
}


bool 
FieldsAlgo::CalculateLinkBetweenOppositeFieldBoundaries(FieldHandle input, 
							MatrixHandle& NodeLink,
							MatrixHandle& ElemLink,
							double tol, 
							bool linkx, 
							bool linky, 
							bool linkz)
{
  CalculateLinkBetweenOppositeFieldBoundariesAlgo algo;
  return(algo.CalculateLinkBetweenOppositeFieldBoundaries(pr_, input, NodeLink,
							  ElemLink, tol, linkx,
							  linky, linkz));
}

bool 
FieldsAlgo::CreateLinkBetweenMeshAndCompGrid(MatrixHandle NodeLink, 
					     MatrixHandle& GeomToComp, 
					     MatrixHandle& CompToGeom)
{
  CreateLinkBetweenMeshAndCompGridAlgo algo;
  return (!(algo.CreateLinkBetweenMeshAndCompGrid(pr_, NodeLink, GeomToComp, 
						  CompToGeom)));
}

bool 
FieldsAlgo::CreateLinkBetweenMeshAndCompGridByDomain(FieldHandle Geometry, 
						     MatrixHandle NodeLink, 
						     MatrixHandle& GeomToComp, 
						     MatrixHandle& CompToGeom)
{
  CreateLinkBetweenMeshAndCompGridByDomainAlgo algo;
  return (algo.CreateLinkBetweenMeshAndCompGridByDomain(pr_, Geometry, 
							NodeLink, GeomToComp,
							CompToGeom));
}


bool 
FieldsAlgo::MapCurrentDensityOntoField(int numproc, FieldHandle pot, 
				       FieldHandle con, FieldHandle dst, 
				       FieldHandle& output, 
				       std::string mappingmethod,
				       std::string integrationmethod, 
				       std::string integrationfilter, 
				       bool multiply_with_normal, 
				       bool calcnorm)
{
  MapCurrentDensityOntoFieldAlgo algo;
  return (algo.MapCurrentDensityOntoField(pr_, numproc, pot, con, dst, output,
					  mappingmethod, integrationmethod,
					  integrationfilter, 
					  multiply_with_normal, calcnorm));
}

bool 
FieldsAlgo::MapCurrentDensityOntoField(FieldHandle pot, FieldHandle con, 
				       FieldHandle dst,  FieldHandle& output, 
				       std::string mappingmethod,
				       std::string integrationmethod, 
				       std::string integrationfilter, 
				       bool multiply_with_normal, 
				       bool calcnorm)
{
  MapCurrentDensityOntoFieldAlgo algo;
  return (algo.MapCurrentDensityOntoField(pr_, 0, pot, con, dst, output, 
					  mappingmethod, integrationmethod,
					  integrationfilter, 
					  multiply_with_normal,
					  calcnorm));
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


bool 
FieldsAlgo::MapFieldDataGradientOntoField(int numproc, FieldHandle src, 
					  FieldHandle dst, FieldHandle& output,
					  std::string mappingmethod,
					  std::string integrationmethod, 
					  std::string integrationfilter, 
					  bool calcnorm)
{
  MapFieldDataGradientOntoFieldAlgo algo;
  return (algo.MapFieldDataGradientOntoField(pr_, numproc, src, dst, output, 
					     mappingmethod, integrationmethod,
					     integrationfilter, calcnorm));
}

bool 
FieldsAlgo::MapFieldDataGradientOntoField(FieldHandle src, FieldHandle dst, 
					  FieldHandle& output, 
					  std::string mappingmethod,
					  std::string integrationmethod, 
					  std::string integrationfilter, 
					  bool calcnorm)
{
  MapFieldDataGradientOntoFieldAlgo algo;
  return (algo.MapFieldDataGradientOntoField(pr_, 0, src, dst, output,
					     mappingmethod, integrationmethod,
					     integrationfilter, calcnorm));
}

bool 
FieldsAlgo::MapFieldDataOntoFieldNodes(int numproc, FieldHandle src, 
				       FieldHandle dst, FieldHandle& output, 
				       std::string mappingmethod, 
				       double def_value)
{
  MapFieldDataOntoFieldNodesAlgo algo;
  return (algo.MapFieldDataOntoFieldNodes(pr_, numproc, src, dst, output,
					  mappingmethod, def_value));
}

bool 
FieldsAlgo::MapFieldDataOntoFieldNodes(FieldHandle src, FieldHandle dst, 
				       FieldHandle& output, 
				       std::string mappingmethod, 
				       double def_value)
{
  MapFieldDataOntoFieldNodesAlgo algo;
  return (algo.MapFieldDataOntoFieldNodes(pr_, 0, src, dst, output,
					  mappingmethod, def_value));
}

bool 
FieldsAlgo::ConvertMappingMatrixToFieldData(FieldHandle input, 
					    FieldHandle& output, 
					    MatrixHandle mappingmatrix)
{
  ConvertMappingMatrixToFieldDataAlgo algo;
  return(algo.ConvertMappingMatrixToFieldData(pr_, input, output, 
					      mappingmatrix));
}

bool FieldsAlgo::MakeEditable(FieldHandle input,FieldHandle& output)
{
  output = input;
  if (!input->mesh()->is_editable()) 
  {
    if(!ConvertToUnstructuredMesh(input,output))
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


bool FieldsAlgo::SplitAndMergeFieldByDomain(FieldHandle input, FieldHandle& output)
{
  FieldHandle input_editable;
  if (!MakeEditable(input,input_editable)) return (false);
  SplitAndMergeFieldByDomainAlgo algo;
  return(algo.SplitAndMergeFieldByDomain(pr_,input_editable,output));
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
  TransformMeshWithTransformAlgo algo;
  return(algo.TransformMeshWithTransform(pr_,
					 input,output,transform,rotatedata));
}


bool FieldsAlgo::ConvertToUnstructuredMesh(FieldHandle input,FieldHandle& output)
{
  ConvertToUnstructuredMeshAlgo algo;
  return(algo.ConvertToUnstructuredMesh(pr_,input,output));
}


bool FieldsAlgo::ClearAndChangeFieldBasis(FieldHandle input,FieldHandle& output, std::string newbasis)
{
  ClearAndChangeFieldBasisAlgo algo;
  return(algo.ClearAndChangeFieldBasis(pr_,input,output,newbasis));
}

bool 
FieldsAlgo::ScaleFieldMeshAndData(FieldHandle input, FieldHandle& output, 
				  double scaledata, double scalemesh, 
				  bool scale_from_center)
{
  ScaleFieldMeshAndDataAlgo algo;
  return(algo.ScaleFieldMeshAndData(pr_, input, output, scaledata, scalemesh,
				    scale_from_center));
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
    GetFieldBoundary(object,dobject,dummy);
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
    // The following will call ConvertToUnstructuredMesh internally
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

