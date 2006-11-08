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


#ifndef CORE_ALGORITHMS_FIELDS_TOPOINTCLOUD_H
#define CORE_ALGORITHMS_FIELDS_TOPOINTCLOUD_H 1

// The following include file will include all tools needed for doing 
// dynamic compilation and will include all the standard dataflow types
#include <Core/Algorithms/Util/DynamicAlgo.h>

// Additionally we include sci_hash_map here as it is needed by the algorithm
#include <sci_hash_map.h>

namespace SCIRunAlgo {

using namespace SCIRun;

// The idea of a dynamically compiling class is the following:
//  A general pointer to the Field is given to the algorithm, this is a pointer
//  to the base class of the Field which is the same for all possible Field 
//  types. To discover what type it is a virtual function call into the Field is
//  made revealing its true identity. This type information is used to compile
//  a templated algorithm that uses the specific field type as a template 
//  argument. When the algorithm is compiled it will execute the algorithm.
//
// In this file two classes are defined:
//  A base class ToPointCloudAlgo: This class contains the general access point for
//  the dynamic compiled algorithm. It takes handles (pointers) to the Field
//  base class. It will examine all the input arguments and determine which
//  type the input Fields actually are. When this is known it will tell the
//  dynamic compiler which algorithm needs to be compiled to carry out the 
//  operation with high efficiency.
//
//  A template class ToPointCloudAlgoT: This class is derived from the base class
//  and has a similar call to the algorithm. This templated class contains the 
//  actual algorithm. When the dynamic compiler is invoked it will take in a
//  handle to the base algorithm, internally it will overload this with the
//  handle to the templated algorithm. Since the templated class will overload
//  the access function calling this one from the base class will get us to the
//  proper algorithm.  


// ToPointCloudAlgo:
//
// This class is the general access point to the dynamic algorithms in the class.
// All dynamic algorithms in the class are defined as virtual functions and have
// the following form:
//  virtual bool MyFunction(ProgressReporter* pr, FieldHandle input1, 
//                                FieldHandle input2, ... FieldHandle& output) 
//
// If the algorithm fails it returns false, if it succeeded it return true.
//
// In order to forward the error message all dynamic algorithms take in a 
// pointer to the current ProgressReporter. The ProgressReporter reports 
// everything from the progress the algorithm made, to errors, remarks, warnings
// and is the general access point to forward messages to the user.
//
// To call this specific algorithm from a module use the following:
//
// ToPointCloudAlgo algo;
// if(!(algo->ToPointCloud(this,input,output)))
// {
//   // algorothm failed
// }
//
// Note that the module class has been derived from the ProgressReporter and 
// hence the pointer to the module can be used to initialise the ProgressReporter
//

class ToPointCloudAlgo : public DynamicAlgoBase
{
public:
  virtual bool ToPointCloud(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
  
  template <class T>
  bool ToPointCloudV(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
};


// ToPointCloudAlgoT:
//
// This class is the actual implementation of the algorithm. Note that the 
// name of the algorithm ends with a 'T' to denote that the algorithm is 
// templated, this is not strictly needed but improves readability if the code.
// This templated class can overloads the algorithm access point with the 
// proper algorithm.   

template <class FSRC, class FDST>
class ToPointCloudAlgoT : public ToPointCloudAlgo
{
public:
  virtual bool ToPointCloud(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
};


// Since the algorithm is templated, we need to define its implemenation in the
// header file.

template <class FSRC, class FDST>
bool ToPointCloudAlgoT<FSRC, FDST>::ToPointCloud(ProgressReporter *pr, FieldHandle input, FieldHandle& output)
{
  // Safety check: test whether the input is of the propertype
  // Handle: the function 'get_rep()' gets the pointer contained in the handle. 
  // The safety check is done by using a dynamic_cast to the type we want to
  // have. The dynamic_cast is a C++ facility that checks the type of the object
  // dynamically. If the object is of a different type it results in a null
  // pointer and otherwise it returns the pointer to the top level object in the
  // class hierarchy.

  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  { // The object is of a different type
  
    // Error reporting:
    // we forward the specific message to the ProgressReporter and return a
    // false to indicate that an error has occured.
    pr->error("ConvertMeshToPointCloud: Could not obtain input field");
    return (false);
  }

  // To get a pointer to the specific mesh we use the function:
  // get_typed_mesh(). In order to get the proper pointer definition we look
  // into the field class where the type mesh_handle_type is defined.
  // As the type consists of a templatename and the actual type defined inside
  // the templated class we use 'typename' to indicate to the compiler that
  // FSRC::mesh_handle_type still needs to be parsed by specifying the actual
  // template class.
  
  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("ConvertMeshToPointCloud: No mesh associated with input field");
    return (false);
  }

  // Create a new mesh. Again the mesh_type is defined in the template so we
  // use typename again.
  
  // Memory management strategy:
  // Handles are smart pointers that automatically deallocate memory when the
  // handle object is deallocated (unless the handle has been copied). Using
  // handles to refer to new objects will make sure that, if we encounter an
  // error in the code and we exit with a return (false), all memory will be
  // freed. The handle was not copied in that case and thus the object it is
  // pointing to will be automatically destroyed.
  
  typename FDST::mesh_handle_type omesh = scinew typename FDST::mesh_type();
  if (omesh == 0)
  {
    pr->error("ConvertMeshToPointCloud: Could not create output mesh");
    return (false);
  }

  // Actual algorithm starts here:
  
  // Synchronize makes sure that all function calls to retrieve the elements and
  // nodes are working. Some mesh types need this.
  imesh->synchronize(Mesh::NODES_E);
  if (imesh->dimensionality() == 1) imesh->synchronize(Mesh::EDGES_E);
  if (imesh->dimensionality() == 2) imesh->synchronize(Mesh::FACES_E);
  if (imesh->dimensionality() == 3) imesh->synchronize(Mesh::CELLS_E);

  // Define iterators over the nodes
  typename FSRC::mesh_type::Node::iterator bn, en;
  typename FSRC::mesh_type::Node::size_type numnodes;

  imesh->begin(bn); // get begin iterator
  imesh->end(en);   // get end iterator
  imesh->size(numnodes); // get the number of nodes in the mesh
  
  // It is always good to preallocate memory if the number of nodes in the output
  // mesh is known
  omesh->node_reserve(numnodes);
  
  // Iterate over all nodes and copy each node
  while (bn != en) 
  {
    Point point;
    imesh->get_center(point, *bn);
    omesh->add_point(point);
    ++bn;
  }

  // We have the new msh now make a new field
  FDST *ofield = scinew FDST(omesh);
  output = dynamic_cast<FDST*>(ofield);
  if (ofield == 0)
  {
    pr->error("ConvertMeshToPointCloud: Could not create output field");
    return (false);  
  }
  
  // Make sure Fdata matches the size of the number of nodes
  ofield->resize_fdata();

  // Is this a linear input field, if so we can copy data from node to node
  if (ifield->basis_order() == 1) 
  {
    typename FSRC::fdata_type::iterator bid  = ifield->fdata().begin();
    typename FSRC::fdata_type::iterator eid = ifield->fdata().end();
    typename FDST::fdata_type::iterator bod = ofield->fdata().begin();

    while (bid != eid) 
    {
      *bod = *bid;
      ++bod; ++bid;
    }
  }

  // In the other cases we cannot preserve data and the output field should 
  // already be of the class NoDataBasis.

  // Make sure we copy all the properties.
  // Some old modules make still use of the property manager and hence to have
  // them work properly the properties from the input field will be assigned to
  // the output field. 
  // As there is no centeral list of properties used, some of the properties 
  // may not be valid anymore in the output field. Hence one is advised not to
  // use the property manager as their validity in the FieldObject cannot be
  // garanteed. 
	output->copy_properties(input.get_rep());
  
  // Success:
  return (true);
}




template <class T>
bool ToPointCloudAlgo::ToPointCloudV(ProgressReporter *pr, FieldHandle input, FieldHandle& output)
{
  // Handle: the function 'get_rep()' gets the pointer contained in the handle. 
  // The safety check is done by using a dynamic_cast to the type we want to
  // have. The dynamic_cast is a C++ facility that checks the type of the object
  // dynamically. If the object is of a different type it results in a null
  // pointer and otherwise it returns the pointer to the top level object in the
  // class hierarchy.

  // Get pointers to the meshes and fields. These should be a little faster than
  // using the handle dereference operation, as it does not do a safety check
  // whether the pointer exists. As long as we have local copies of the handles
  // The objects cannot be deleted and we can safely use the pointers.
  Field *ifield = input.get_rep();
  if (ifield == 0)
  { // The object is of a different type
  
    // Error reporting:
    // we forward the specific message to the ProgressReporter and return a
    // false to indicate that an error has occured.
    pr->error("ConvertMeshToPointCloud: Could not obtain input field");
    return (false);
  }

  // Get a pointer to the mesh as well
  Mesh *imesh = ifield->mesh().get_rep();
  if (imesh == 0)
  {
    pr->error("ConvertMeshToPointCloud: No mesh associated with input field");
    return (false);
  }

  // The output mesh was already create based on the field information in the
  // main algorithm

  Field *ofield = output.get_rep();
  if (ofield == 0)
  { // The object is of a different type
  
    // Error reporting:
    // we forward the specific message to the ProgressReporter and return a
    // false to indicate that an error has occured.
    pr->error("ConvertMeshToPointCloud: Could not obtain output field");
    return (false);
  }

  // Get a pointer to the mesh as well
  Mesh *omesh = ofield->mesh().get_rep();
  if (omesh == 0)
  {
    pr->error("ConvertMeshToPointCloud: No mesh associated with input field");
    return (false);
  }

  // Actual algorithm starts here:
  
  // Synchronize makes sure that all function calls to retrieve the elements and
  // nodes are working. Some mesh types need this.
  imesh->synchronize(Mesh::NODES_E);
  if (imesh->dimensionality() == 1) imesh->synchronize(Mesh::EDGES_E);
  if (imesh->dimensionality() == 2) imesh->synchronize(Mesh::FACES_E);
  if (imesh->dimensionality() == 3) imesh->synchronize(Mesh::CELLS_E);

  // Define iterators over the nodes
  Mesh::VNode::iterator bn, en;
  Mesh::VNode::size_type numnodes;

  imesh->begin(bn); // get begin iterator
  imesh->end(en);   // get end iterator
  imesh->size(numnodes); // get the number of nodes in the mesh
  
  // It is always good to preallocate memory if the number of nodes in the output
  // mesh is known
  omesh->node_reserve(numnodes);
  
  // Iterate over all nodes and copy each node
  while (bn != en) 
  {
    Point point;
    imesh->get_center(point, *bn);
    omesh->add_node(point);
    ++bn;
  }

  // Make sure Fdata matches the size of the number of nodes
  ofield->resize_fdata();

  // Is this a linear input field, if so we can copy data from node to node
  if (ifield->basis_order() == 1) 
  {
    // The Field class has both a iterator as well as a direct interface access 
    // using indices. As the latter is faster we use the direct access 

    Mesh::size_type sz = ifield->num_values();
    T val;
    for (Mesh::index_type r=0; r<sz; r++)
    {
      ifield->get_value(val,r);
      ofield->set_value(val,r);
    }
  }

  // In the other cases we cannot preserve data and the output field should 
  // already be of the class NoDataBasis.

	output->copy_properties(input.get_rep());
  
  // Success:
  return (true);
}




} // end namespace SCIRunAlgo

#endif 

