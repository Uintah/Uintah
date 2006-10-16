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


#ifndef CORE_ALGORITHMS_FIELDS_FIELDALGO_H
#define CORE_ALGORITHMS_FIELDS_FIELDALGO_H 1

#include <Core/Algorithms/Util/AlgoLibrary.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <list>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Core/Algorithms/Fields/share.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class SCISHARE FieldsAlgo : public AlgoLibrary {

  public:
    FieldsAlgo(ProgressReporter* pr); // normal case
  
    // ApplyMappingMatrix:
    // Copy the  data from one field to the other using a mapping matrix
    bool ApplyMappingMatrix(FieldHandle fsrc,  FieldHandle fdst, FieldHandle& output, MatrixHandle mapping);

    // ApplyMappingMatrix:
    // Get the bounding box of a data set
    bool GetBoundingBox(FieldHandle input,  FieldHandle& output);

  
    // BundleToFieldArray:
    // Created an vector of fields out of the bundle type
    bool BundleToFieldArray(BundleHandle input, std::vector<FieldHandle>& output);
        
    // ChangeFieldBasis:
    // Change the type of data basis of the field:
    // Currently supported types are Constant, Linear and Quadratic
    bool ChangeFieldBasis(FieldHandle input,FieldHandle& output, MatrixHandle &interpolant, std::string newbasis);

    // ClearAndChangeFieldBasis:
    // Similar to ChangeBasis but do not do the interpolation stuff and just clear the
    // field. This is handy in creating a new field.
    bool ClearAndChangeFieldBasis(FieldHandle input,FieldHandle& output, std::string newbasis);

    // ClipFieldBySelectionMask:
    // Clip using a selectionmask to clip a field
    bool ClipFieldBySelectionMask(FieldHandle input, FieldHandle& output, MatrixHandle SelectionMask,MatrixHandle &interpolant);

    // ClipFieldByField:
    // Clip using a mesh.
    bool ClipFieldByField(FieldHandle input, FieldHandle& output, FieldHandle objfield, MatrixHandle &interpolant);

    // ConvertToTetVol:
    // This function converts an hexvol or latvol into a tetvol. The functionality
    // is similar to HexToTet, but does more checks and is more robust and works
    // on unconnected data. The function will fail if topology will not allow for
    // a properly connected tetrahedral field using a scheme that adds 5 tetrahedral
    // for each hexahedral element.
    bool ConvertToTetVol(FieldHandle input, FieldHandle& output);

    // ConvertToTriSurf:
    // This function converts an quadsurf or image into a trisurf. The functionality
    // is similar to QuadToTri, but does more checks and is more robust and works
    // on unconnected data.
    bool ConvertToTriSurf(FieldHandle input, FieldHandle& output);
    
    // DomainBoundary:
    // Extract the boundaries between compartments in a volume or surface field
    // The data needs to be on the elements. This function only extracts internal
    // boundaries, use field boundary to extract the outer surfaces.
    // minrange, maxrange: scanning range for boundaries only in elements with
    //    values in this range.
    // userange: switch on ranging
    // addouterboundary: Add the boundary at the outside of the field the faces/edges
    //    between the edge of field and the empty space (compare FieldBoundary)
    //    if ranging is on only the elements in range that have an outer face/edge are included
    // innerboundaryonly: Discard any of the boundaries that are not between elements with
    //    values that are in range.
    // DomainLink: addition to make opposite boundaries link together. In this case elements
    //    at opposite boundaries can still have a linking boundary and hence they may show
    //    up as internal boundaries 
    bool DomainBoundary(FieldHandle input, FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly, bool noinnerboundary, bool disconnect = true);

    // IndexedBoundary:
    // Like the version above, but it adds a vector to each element, indicating the original face/curve index, and the
    // two indices of the elements on both sides of the face/curve. These indices are put in a vector.
    bool IndexedDomainBoundary(FieldHandle input, FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly, bool noinnerboundary, bool disconnect = true);
    
    // FieldDataNodeToElem and FieldDataElemToNode:
    // Change where the data is located the algorithms now employ summing, median, maximum and minimum
    // operators to move data. Currently interpolation is NOT YET using basis interpolation
    // Methods are: Interpolate, Average, Max, Min, Median, Sum
    bool FieldDataNodeToElem(FieldHandle input, FieldHandle& output, std::string method);
    bool FieldDataElemToNode(FieldHandle input, FieldHandle& output, std::string method);
    
    // FindClosestNodeByValue:
    // Find the closest node with a certain value. This is needed for the multi domain stuff where multiple nodes
    // might exist in the same position
    bool FindClosestNodeByValue(FieldHandle input, std::vector<unsigned int>& output, FieldHandle& points, double value);
    bool FindClosestNode(FieldHandle input, std::vector<unsigned int>& output, FieldHandle& points);
    
    // ManageFieldData was split into two parts:
    // GetFieldData: Extract the data from the field into a matrix
    // SetFieldData: Put the data contained in a matrix into the field
    // Still need to upgrade these for non-linear elements
    bool GetFieldData(FieldHandle input, MatrixHandle& data);
    bool SetFieldData(FieldHandle input, FieldHandle& output,MatrixHandle data, bool keepscalartype = false);
	
    // GetFieldInfo:
    // Due to some oddity in the field design information like number of nodes and elements cannot be queried directly
    // Hence we have implemented it here directly through a algorithm
    bool GetFieldInfo(FieldHandle input, int& numnodes, int& numelems);

    // GetFieldDataMinMax:
    bool GetFieldDataMinMax(FieldHandle input, double& min, double& max);

    // GetFieldMeasure:
    bool GetFieldMeasure(FieldHandle input, string method, double& measuer);

    // DistanceToField and SignedDistanceToField:
    // Compute distance field. These functions take a destination field (input)
    // and an object field to which the distance needs to be computed.
    bool DistanceField(FieldHandle input, FieldHandle& output, FieldHandle object);
    bool SignedDistanceField(FieldHandle input, FieldHandle& output, FieldHandle object);

    // FieldArrayToBundle:
    // Created an vector of fields out of the bundle type
    bool FieldArrayToBundle(std::vector<FieldHandle> input, BundleHandle& output);
   
    // FieldBoundary:
    // This function extracts the outer boundaries of a field. This function is similar
    // to the one included in the Core of SCIRun, but works as well for Curves and
    // employs more modern compact code.
    bool FieldBoundary(FieldHandle input, FieldHandle& output, MatrixHandle &interpolant);
 
    // GatherFields: 
    // Gather a set of fields of the same type together into one
    // new output field. This function will not do any merging.
    // NOTE: in SCIRun the module GatherFields was misnamed as it actually
    // does merge fields.
    bool GatherFields(std::list<FieldHandle> inputs, FieldHandle& output);

    // IndicesToData:
    // Transform indexed data into to data. Using a matrix as a lookup table.
    bool IndicesToData(FieldHandle input, FieldHandle& output, MatrixHandle data);

    // IsInsiedField:
    // This is an implementation of locate whether an element is contained within an
    // object.
    bool IsInsideField(FieldHandle input, FieldHandle& output, FieldHandle object, std::string output_type = "double", std::string basis_type = "same as input",bool partial_inside = false, double outval = 0.0, double inval = 1.0);
    bool IsInsideFields(FieldHandle input, FieldHandle& output, std::vector<FieldHandle> objectfields, std::string output_type = "double", std::string basis_type = "same as input",bool partial_inside = false, double outval = 0.0);
    
    // LinkFieldBoundary:
    // Compute the node-to-node link and the edge-element-to-edge-element matrix.
    // This function assumes that the field uses Cartesian coordinates and that it is 
    // a piece of a jigsaw puzzle. And that the field can serve as a tile in the
    // x,y, and z direction. The function will automatically detect the size of the
    // tile (border do not need to be flat planes but just need to mirror each other).
    // tol: tolerance for detection of connecting nodes
    // linkx, linky, linkz: switch on borders on which we want to have linkage information
    // NodeLink: Sparse matrix which denotes which nodes are connected to which nodes
    //            Note: one node can be connected to multiple nodes (e.g. corner of a cube)
    // ElemLink: Sparse matrix which describes how the faces/edges in the mesh are connected
    //            to the opposite edges/faces. Note: this is a pure one to one mapping
    bool LinkFieldBoundary(FieldHandle input, MatrixHandle& NodeLink, MatrixHandle& ElemLink, double tol, bool linkx = true, bool linky = true, bool linkz = true);
    
    // LinkToCompGrid:
    // Compute the mapping to merge nodes over the outer boundary of the mesh into a computational grid.
    // GeomToComp and CompToGeom: These are the mapping matrices that map geometrical space into
    //  computational space and vice versa.
    bool LinkToCompGrid(MatrixHandle NodeLink,MatrixHandle& GeomToComp, MatrixHandle& CompToGeom);

    // LinkToCompGrid:
    // Compute the mapping to merge nodes over the outer boundary of the mesh for elements of the same domain type.
    // We assume that there is only a link if the elemets are of the same type. The types of elements
    // need to be defined in the input field. Hence this function will only work for data on the
    // elements.
    // GeomToComp and CompToGeom: These are the mapping matrices that map geometrical space into
    //  computational space and vice versa.       
    bool LinkToCompGridByDomain(FieldHandle input, MatrixHandle NodeLink, MatrixHandle& GeomToComp, MatrixHandle& CompToGeom);

    // MappingMatrixToField:
    // This function will assign to each node the value of the original node.
    // Hence by selecting areas in this field one can obtain all nodes located
    // inside the original field. The result is that we will have the node numbers
    // of the original field onto the new output field.
    bool MappingMatrixToField(FieldHandle input, FieldHandle& output, MatrixHandle mappingmatrix);

    // MakeEditable: 
    // Make a mesh editable. This function calls unstructure if needed.
    bool MakeEditable(FieldHandle input, FieldHandle& output);

    // MergeFields: 
    // Merge a set of fields of the same type together into one
    // new output field. If mergenodes is true, nodes will be merge if the
    // distance between them is smaller than tolerance  
    bool MergeFields(std::vector<FieldHandle> inputs, FieldHandle& output, double tolerance, bool mergenodes = true, bool mergeelements = true, bool matchvalue = true);
    bool MergeMeshes(std::vector<FieldHandle> inputs, FieldHandle& output, double tolerance, bool mergenodes = true, bool mergeelements = true);

    // MergeNodes: 
    // Merge the nodes in a field together if the distance between
    // them is smaller than tolerance. This is an alternative front end to
    // MergeFields.
    bool MergeNodes(FieldHandle input, FieldHandle& output, double tolerance, bool mergeelements = true, bool matchvalue = true);
 
 
    // ModalMapping:
    // Map data from a source field (any type) onto the elements of the destination field.
    // This mapping involves integrating over the elements to get a fair representation of
    // the original field
    // MappingMethod:
    //  How do we select data
    //  ClosestNodalData = Find the closest data containing element or node
    //  ClosestInterpolatedData = Find the closest interpolated data point. Inside the volume it picks the value
    //                            the interpolation model predicts and outside it makes a shortest projection
    //  InterpolatedData = Uses interpolated data using the interpolation model whereever possible and assumes no
    //                     value outside the source field
    // IntegrationMethod:
    //  Gaussian1 = Use 1st order Gaussian weights and nodes for integration
    //  Gaussian2 = Use 2nd order Gaussian weights and nodes for integration
    //  Gaussian3 = Use 3rd order Gaussian weights and nodes for integration
    //  Regular1  = Use 1 evenly space node in each dimension 
    //  Regular2  = Use 2 evenly space nodes in each dimension 
    //  Regular3  = Use 3 evenly space nodes in each dimension 
    //  Regular4  = Use 4 evenly space nodes in each dimension 
    // Integration Filter:
    //  Average =  take average value over integration nodes but disregard weights
    //  Integrate = sum values over integration nodes using gaussian weights
    //  Minimum = find minimum value using integration nodes
    //  Maximum = find maximum value using integration nodes
    //  Median  = find median value using integration nodes
    //  MostCommon = find most common value among integration nodes    
    bool ModalMapping(int numproc, FieldHandle src, FieldHandle dst, FieldHandle& output, std::string mappingmethod,
                       std::string integrationmethod, std::string integrationfilter, double def_value = 0.0);
    bool ModalMapping(FieldHandle src, FieldHandle dst, FieldHandle& output, std::string mappingmethod,
                       std::string integrationmethod, std::string integrationfilter, double def_value = 0.0);


    bool GradientModalMapping(int numproc, FieldHandle src, FieldHandle dst, FieldHandle& output, std::string mappingmethod,
                       std::string integrationmethod, std::string integrationfilter, bool calcnorm = false);
    bool GradientModalMapping(FieldHandle src, FieldHandle dst, FieldHandle& output, std::string mappingmethod,
                       std::string integrationmethod, std::string integrationfilter,bool calcnorm = false);

    // CurrentDensityMapping:
    // Map data from a potential field and a conductivity field into a current density field. The underlying geometry
    // of all the different fields can be different. This method is intended to map volumetric potential and conductivity
    // data onto a surface. The method has an option to multiply with the surface normal and hence one can sample and
    // integrate the potential/conductivity data into the  current that passes through an element. Summing this vector
    // will generate the total amount of current through a model.
    //
    // MappingMethod:
    //  How do we select data
    //  InterpolatedData = Uses interpolated data using the interpolation model whereever possible and assumes no
    //                     value outside the source field
    // IntegrationMethod:
    //  Gaussian1 = Use 1st order Gaussian weights and nodes for integration
    //  Gaussian2 = Use 2nd order Gaussian weights and nodes for integration
    //  Gaussian3 = Use 3rd order Gaussian weights and nodes for integration
    //  Regular1  = Use 1 evenly space node in each dimension 
    //  Regular2  = Use 2 evenly space nodes in each dimension 
    //  Regular3  = Use 3 evenly space nodes in each dimension 
    //  Regular4  = Use 4 evenly space nodes in each dimension 
    // Integration Filter:
    //  Average =  take average value over integration nodes but disregard weights
    //  Integrate = sum values over integration nodes using gaussian weights
    //  Minimum = find minimum value using integration nodes
    //  Maximum = find maximum value using integration nodes
    //  Median  = find median value using integration nodes
    //  MostCommon = find most common value among integration nodes 
    //
    // Most likely one wants to use the integration option to compute fluxes through a mesh
    // This method can be used with any sampling of output mesh, although the best results
    // are probably obtained when using a mesh that is a slice of the volumetric mesh.
    bool CurrentDensityMapping(int numproc, FieldHandle pot, FieldHandle con, FieldHandle dst, FieldHandle& output, std::string mappingmethod,
                       std::string integrationmethod, std::string integrationfilter, bool multiply_with_normal, bool calcnorm);
    bool CurrentDensityMapping(FieldHandle pot, FieldHandle con, FieldHandle dst, FieldHandle& output, std::string mappingmethod,
                       std::string integrationmethod, std::string integrationfilter, bool multiply_with_normal, bool calcnorm);
    

    // NodalMapping:
    // Map data from a source field (any type) onto the nodes of the destination field.
    // MappingMethod:
    //  How do we select data
    //  ClosestNodalData = Find the closest data containing element or node
    //  ClosestInterpolatedData = Find the closest interpolated data point. Inside the volume it picks the value
    //                            the interpolation model predicts and outside it makes a shortest projection
    //  InterpolatedData = Uses interpolated data using the interpolation model whereever possible and assumes no
    //                     value outside the source field

    bool NodalMapping(int numproc, FieldHandle src, FieldHandle dst, FieldHandle& output, std::string mappingmethod, double def_value = 0.0);
    bool NodalMapping( FieldHandle src, FieldHandle dst, FieldHandle& output, std::string mappingmethod, double def_value = 0.0);
 
 
    // RemoveUnusedNodes:
    // Remove any nodes that are not connected to an element
    bool RemoveUnusedNodes(FieldHandle input, FieldHandle& output);
 
    // ScaleField:
    // Scales FieldData and MeshData, used to change units properly both in
    // geometry and in data space. 
    bool ScaleField(FieldHandle input, FieldHandle& output, double scaledata, double scalemesh, bool scale_from_center = true);

    // SplitFieldByDomain:
    // Use the element data to segment the input field into volumes/areas with a
    // constant value. This means node splitting at the edges between the
    // different areas/volumes. The result will be a mesh with unconnected
    // regions of the same value type.
    bool SplitFieldByDomain(FieldHandle input, FieldHandle& output);    

    // SplitFieldByConnectedRegion:
    // Use the connectivity data to split the field so each unconnected region is its own
    // field. The main goal is to use DomainBoundary first to get the surfaces that
    // the compartments in the field. Then using this function one can extract
    // the different surfaces that are not connected.
    bool SplitFieldByConnectedRegion(FieldHandle input, std::vector<FieldHandle>& output);    

    // ToPointCloud: 
    // Remove all element information from a mesh and only extract
    // the actual points in the mesh.
    bool ToPointCloud(FieldHandle input, FieldHandle& output);
    
    // TransformField: 
    // Transform a field and rotate vectors and tensors accordingly.
    // The version in SCIRun's core does not allow for rotating the data
    // when transforming the data. 
    bool TransformField(FieldHandle input, FieldHandle& output, Transform& transform, bool rotatedata = true);
    
    // Unstructure: 
    // Unstructure a mesh from a regular or structured mesh into
    // an unstructured mesh. This is often needed to make a mesh editable
    bool Unstructure(FieldHandle input,FieldHandle& output);

    // CleanMesh:
    // Cleanup a mesh:
    //  removeunusednodes   -> remove nodes from mesh that are not used
    //  removeunusedelems   -> remove elements with zero volume/area/length
    //  reorientelems       -> reorient elements (maximize internal volume/area/length)
    //  mergenodes          -> merge identical nodes
    //  mergeelems          -> merge identical elements
    bool CleanMesh(FieldHandle input, FieldHandle& output, bool removeunusednodes, bool removeunusedelems, bool reorientelems, bool mergenodes, bool mergeelems);    
};


} // end namespace 

#endif
