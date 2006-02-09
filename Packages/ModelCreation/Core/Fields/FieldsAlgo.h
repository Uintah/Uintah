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

#include <Packages/ModelCreation/Core/Util/AlgoLibrary.h>

#include <Core/Bundle/Bundle.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Dataflow/Network/Module.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sstream>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

using namespace SCIRun;

class FieldsAlgo : public AlgoLibrary {

  public:
    FieldsAlgo(ProgressReporter* pr); // normal case

    // Funtions borrow from Core of SCIRun
    bool ApplyMappingMatrix(FieldHandle input, FieldHandle& output, MatrixHandle interpolant, FieldHandle datafield);
    bool ChangeFieldBasis(FieldHandle input,FieldHandle& output, MatrixHandle &interpolant, std::string newbasis);
    
    // ManageFieldData split into two parts
    // Need to upgrade code for these when we are done with HO integration
    bool SetFieldData(FieldHandle input, FieldHandle& output,MatrixHandle data, bool keepscalartype);
    bool GetFieldData(FieldHandle input, MatrixHandle& data);
	
    // Due to some oddity in the FieldDesign information like this cannot be queried directly
    bool GetFieldInfo(FieldHandle input, int& numnodes, int& numelems);
    
    bool ClipFieldBySelectionMask(FieldHandle input, FieldHandle& output, MatrixHandle SelectionMask,MatrixHandle &interpolant);
    bool DistanceToField(FieldHandle input, FieldHandle& output, FieldHandle object);
    bool SignedDistanceToField(FieldHandle input, FieldHandle& output, FieldHandle object);
    
    bool IsInsideSurfaceField(FieldHandle input, FieldHandle& output, FieldHandle object);
    bool IsInsideVolumeField(FieldHandle input, FieldHandle& output, FieldHandle object);

    // Change where the data is located
    bool FieldDataNodeToElem(FieldHandle input, FieldHandle& output, std::string method);
    bool FieldDataElemToNode(FieldHandle input, FieldHandle& output, std::string method);

    bool FilterFieldElements(FieldHandle input, FieldHandle& output, bool removezerosize = true, bool removedegenerate = true, bool removedouble = true);

    // Check properties of surface field
    bool IsClosedSurface(FieldHandle input);
    bool IsClockWiseSurface(FieldHandle input);
    bool IsCounterClockWiseSurface(FieldHandle input);

    // BuildMembraneTable
    // Find the surfaces in membranemodel and fir them to the ones found in
    // elementtype. This will produce a table that can be used to see which surfaces
    // in the elementtype mesh can be used to model the membrane model.
    // This is a support function for the CardioWave Interface
    bool BuildMembraneTable(FieldHandle elementtype, FieldHandle membranemodel, MatrixHandle& table);
  
    // BundleToFieldArray
    // Created an vector of fields out of the bundle type
    bool BundleToFieldArray(BundleHandle input, std::vector<FieldHandle>& output);

    // CompartmentBoundary
    // Extract the boundaries between compartments in a volume or surface field
    // The data needs to be on the elements. This function only extracts internal
    // boundaries, use field boundary to extract the outer surfaces.
    bool CompartmentBoundary(FieldHandle input, FieldHandle& output, double minrange, double maxrange, bool userange, bool addouterboundary);

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

    // FieldArrayToBundle
    // Created an vector of fields out of the bundle type
    bool FieldArrayToBundle(std::vector<FieldHandle>& input, BundleHandle output);
    
    // FieldBoundary:
    // This function extracts the outer boundaries of a field
    bool FieldBoundary(FieldHandle input, FieldHandle& output, MatrixHandle &interpolant);
    
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
    bool MergeFields(std::vector<FieldHandle> inputs, FieldHandle& output, double tolerance, bool mergenodes = true, bool mergeelements = true);

    // MergeNodes: Merge the nodes in a field together if the distance between
    // them is smaller than tolerance.
    bool MergeNodes(FieldHandle input, FieldHandle& output, double tolerance, bool mergeelements = true);

    // SplitFieldByElementData:
    // Use the element data to segment the input field into volumes/areas with a
    // constant value. This means node splitting at the edges between the
    // different areas/volumes.
    bool SplitFieldByElementData(FieldHandle input, FieldHandle& output);    

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
    
};


} // ModelCreation

#endif
