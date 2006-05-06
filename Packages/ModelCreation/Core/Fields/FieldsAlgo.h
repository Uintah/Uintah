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


#ifndef MODELCREATION_CORE_FIELDS_FIELDALGO_H
#define MODELCREATION_CORE_FIELDS_FIELDALGO_H 1

#include <Core/Algorithms/Util/AlgoLibrary.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

using namespace SCIRun;

class FieldsAlgo : public AlgoLibrary {

  public:
    FieldsAlgo(ProgressReporter* pr); // normal case
  
    // Funtions borrow from Core of SCIRun
    bool ApplyMappingMatrix(FieldHandle fsrc,  FieldHandle fdst, FieldHandle& output, MatrixHandle mapping);
    bool ChangeFieldBasis(FieldHandle input,FieldHandle& output, MatrixHandle &interpolant, std::string newbasis);
    
    // ManageFieldData split into two parts
    // Need to upgrade code for these when we are done with HO integration
    bool SetFieldData(FieldHandle input, FieldHandle& output,MatrixHandle data, bool keepscalartype = false);
    bool GetFieldData(FieldHandle input, MatrixHandle& data);
	
    // Due to some oddity in the FieldDesign information like this cannot be queried directly
    bool GetFieldInfo(FieldHandle input, int& numnodes, int& numelems);
    
    // ClipFieldBySelectionMask:
    // Clip using a selectionmask
    bool ClipFieldBySelectionMask(FieldHandle input, FieldHandle& output, MatrixHandle SelectionMask,MatrixHandle &interpolant);
    
    // Change where the data is located
    bool FieldDataNodeToElem(FieldHandle input, FieldHandle& output, std::string method);
    bool FieldDataElemToNode(FieldHandle input, FieldHandle& output, std::string method);
  
    // BundleToFieldArray
    // Created an vector of fields out of the bundle type
    bool BundleToFieldArray(BundleHandle input, std::vector<FieldHandle>& output);

    // ClearAndChangeFieldBasis
    // Similar to ChangeBasis but do not do the interpolation stuff
    bool ClearAndChangeFieldBasis(FieldHandle input,FieldHandle& output, std::string newbasis);

    // DomainBoundary
    // Extract the boundaries between compartments in a volume or surface field
    // The data needs to be on the elements. This function only extracts internal
    // boundaries, use field boundary to extract the outer surfaces.
    bool DomainBoundary(FieldHandle input, FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly);

    // ConvertToTetVol:
    // This function converts an hexvol or latvol into a tetvol. The functionality
    // is similar to HexToTet, but does more checks and is more robust and works
    // on unconnected data.
    bool ConvertToTetVol(FieldHandle input, FieldHandle& output);

    // ConvertToTriSurf:
    // This function converts an quadsurf or image into a trisurf. The functionality
    // is similar to QuadToTri, but does more checks and is more robust and works
    // on unconnected data.
    bool ConvertToTriSurf(FieldHandle input, FieldHandle& output);

    // Compute distance fields
    bool DistanceField(FieldHandle input, FieldHandle& output, FieldHandle object);
    bool SignedDistanceField(FieldHandle input, FieldHandle& output, FieldHandle object);

    // FieldArrayToBundle
    // Created an vector of fields out of the bundle type
    bool FieldArrayToBundle(std::vector<FieldHandle> input, BundleHandle& output);
    
    // FieldBoundary:
    // This function extracts the outer boundaries of a field
    bool FieldBoundary(FieldHandle input, FieldHandle& output, MatrixHandle &interpolant);

    // IsInsiedField:
    // This is an implementation of locate
    bool IsInsideField(FieldHandle input, FieldHandle& output, FieldHandle object);

    // LinkFieldBoundary:
    // Compute the node-to-node link and the edge-elementy-to-edge-element matrix
    bool LinkFieldBoundary(FieldHandle input, MatrixHandle& NodeLink, MatrixHandle& ElemLink, double tol, bool linkx = true, bool linky = true, bool linkz = true);
    
    // LinkToCompGrid:
    // Compute the mapping to merge nodes over the outer boundary of the mesh
    bool LinkToCompGrid(MatrixHandle NodeLink,MatrixHandle& GeomToComp, MatrixHandle& CompToGeom);

    // LinkToCompGrid:
    // Compute the mapping to merge nodes over the outer boundary of the mesh for elements of the same domain type    
    bool LinkToCompGridByDomain(FieldHandle input, MatrixHandle NodeLink, MatrixHandle& GeomToComp, MatrixHandle& CompToGeom);

    // MappingMatrixToField:
    // This function will assign to each node the value of the original node.
    // Hence by selecting areas in this field one can obtain all nodes located
    // inside the original field
    bool MappingMatrixToField(FieldHandle input, FieldHandle& output, MatrixHandle mappingmatrix);

    // MakeEditable: Make a mesh editable. This function calls unstructure if
    // needed.
    bool MakeEditable(FieldHandle input, FieldHandle& output);

    // MergeFields: Merge a set of fields of the same type together into one
    // new output field. If mergenodes is true, nodes will be merge if the
    // distance between them is smaller than tolerance  
    bool MergeFields(std::vector<FieldHandle> inputs, FieldHandle& output, double tolerance, bool mergenodes = true, bool mergeelements = true, bool matchvalue = true);

    // MergeNodes: Merge the nodes in a field together if the distance between
    // them is smaller than tolerance.
    bool MergeNodes(FieldHandle input, FieldHandle& output, double tolerance, bool mergeelements = true, bool matchvalue = true);
 
    // ScaleField:
    // Scales FieldData and MeshData, used to change units properly
    bool ScaleField(FieldHandle input, FieldHandle& output, double scaledata, double scalemesh);

    // SplitFieldByDomain:
    // Use the element data to segment the input field into volumes/areas with a
    // constant value. This means node splitting at the edges between the
    // different areas/volumes.
    bool SplitFieldByDomain(FieldHandle input, FieldHandle& output);    

    // SplitFieldByConnectedRegion:
    // Use the connectivity data to split the field so each unconnected region is its own
    // field.
    bool SplitFieldByConnectedRegion(FieldHandle input, std::vector<FieldHandle>& output);    

    // ToPointCloud: Remove all element information from a mesh and only extract
    // the actual points in the mesh.
    bool ToPointCloud(FieldHandle input, FieldHandle& output);
    
    // TransformField: Transform a field and rotate vectors and tensors accordingly
    bool TransformField(FieldHandle input, FieldHandle& output, Transform& transform, bool rotatedata = true);
    
    // Unstructure: Unstructure a mesh from a regular or structured mesh into
    // an unstructured mesh. This is often needed to make a mesh editable
    bool Unstructure(FieldHandle input,FieldHandle& output);
    
    // TriSurfPhaseFilter: Reconstruct phase shifts 
    bool TriSurfPhaseFilter(FieldHandle input, FieldHandle& output, FieldHandle& phaseline, FieldHandle& phasepoint);    
    
    // TracePoints: Trace how points over time
    bool TracePoints(ProgressReporter *pr, FieldHandle pointcloud, FieldHandle old_curvefield, FieldHandle& curvefield, double val, double tol);
};


} // ModelCreation

#endif
