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



#ifndef Datatypes_Field_h
#define Datatypes_Field_h

#include <Core/Datatypes/PropertyManager.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Util/DynamicCompilation.h>

#include <Core/Datatypes/share.h>

namespace SCIRun {
 
typedef LockingHandle<ScalarFieldInterface> ScalarFieldInterfaceHandle;
typedef LockingHandle<VectorFieldInterface> VectorFieldInterfaceHandle;
typedef LockingHandle<TensorFieldInterface> TensorFieldInterfaceHandle;

class SCISHARE Field: public PropertyManager
{
public:
  enum  td_info_e {
    FULL_TD_E,
    FIELD_NAME_ONLY_E,
    MESH_TD_E,
    BASIS_TD_E,
    FDATA_TD_E
  };

  Field();

  virtual ~Field();
  virtual Field *clone() const = 0;
  
  virtual int basis_order() const = 0;
  virtual const TypeDescription *order_type_description() const = 0;

  //! Required virtual functions
  virtual MeshHandle mesh() const = 0;
  virtual void mesh_detach() = 0;
  virtual const TypeDescription* 
  get_type_description(td_info_e td = FULL_TD_E) const = 0; 
  

  //! Required interfaces
  virtual ScalarFieldInterfaceHandle query_scalar_interface(
						      ProgressReporter* = 0);
  virtual VectorFieldInterfaceHandle query_vector_interface(
						      ProgressReporter* = 0);
  virtual TensorFieldInterfaceHandle query_tensor_interface(
						      ProgressReporter* = 0);

  //! Persistent I/O.
  static  PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  //! All instantiable classes need to define this.
  virtual bool is_scalar() const = 0;

  virtual unsigned int data_size() const = 0;
};

typedef LockingHandle<Field> FieldHandle;


template <class FIELD>
static FieldHandle
append_fields(vector<FIELD *> fields)
{
  typename FIELD::mesh_type *omesh = scinew typename FIELD::mesh_type();

  unsigned int offset = 0;
  unsigned int i;
  for (i=0; i < fields.size(); i++)
  {
    typename FIELD::mesh_handle_type imesh = fields[i]->get_typed_mesh();
    typename FIELD::mesh_type::Node::iterator nitr, nitr_end;
    imesh->begin(nitr);
    imesh->end(nitr_end);
    while (nitr != nitr_end)
    {
      Point p;
      imesh->get_center(p, *nitr);
      omesh->add_point(p);
      ++nitr;
    }

    typename FIELD::mesh_type::Elem::iterator eitr, eitr_end;
    imesh->begin(eitr);
    imesh->end(eitr_end);
    while (eitr != eitr_end)
    {
      typename FIELD::mesh_type::Node::array_type nodes;
      imesh->get_nodes(nodes, *eitr);
      unsigned int j;
      for (j = 0; j < nodes.size(); j++)
      {
	nodes[j] = ((unsigned int)nodes[j]) + offset;
      }
      omesh->add_elem(nodes);
      ++eitr;
    }
    
    typename FIELD::mesh_type::Node::size_type size;
    imesh->size(size);
    offset += (unsigned int)size;
  }

  FIELD *ofield = scinew FIELD(omesh);
  offset = 0;
  for (i=0; i < fields.size(); i++)
  {
    typename FIELD::mesh_handle_type imesh = fields[i]->get_typed_mesh();
    typename FIELD::mesh_type::Node::iterator nitr, nitr_end;
    imesh->begin(nitr);
    imesh->end(nitr_end);
    while (nitr != nitr_end)
    {
      double val;
      fields[i]->value(val, *nitr);
      typename FIELD::mesh_type::Node::index_type
	new_index(((unsigned int)(*nitr)) + offset);
      ofield->set_value(val, new_index);
      ++nitr;
    }

    typename FIELD::mesh_type::Node::size_type size;
    imesh->size(size);
    offset += (unsigned int)size;
  }

  return ofield;
}

} // end namespace SCIRun

#endif // Datatypes_Field_h
















