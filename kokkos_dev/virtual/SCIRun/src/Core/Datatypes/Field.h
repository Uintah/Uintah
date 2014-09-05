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



#ifndef CORE_DATATYPES_FIELD_H
#define CORE_DATATYPES_FIELD_H 1

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

class FDataResult
{
public:
  std::vector<double> scalar;
  std::vector<Vector> vector;
  std::vector<Tensor> tensor;
  std::vector<Mesh::index_type> index;
};

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

  virtual Mesh::size_type data_size() const = 0;
  inline Mesh::size_type num_values() const { return (data_size()); }
  
  // VIRTUAL INTERFACE FOR FIELDS
  // value and set_value are available for each datatype supported so far
  // by SCIRun.
  // value& is not supported virtually as most of the values run through
  // a casting operation
  
  virtual bool has_virtual_interface();
  
  virtual void resize_fdata();
  virtual void resize_fdata(size_t size);
  
  virtual void get_value(int &val, Mesh::index_type i) const;
  virtual void get_value(double &val, Mesh::index_type i) const;
  virtual void get_value(Vector &val, Mesh::index_type i) const;
  virtual void get_value(Tensor &val, Mesh::index_type i) const;

  template<class T>  inline void get_value(T& val, Mesh::VNode::index_type idx) const
  { get_value(val,static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void get_value(T& val, Mesh::VEdge::index_type idx) const
  { get_value(val,static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void get_value(T& val, Mesh::VFace::index_type idx) const
  { get_value(val,static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void get_value(T& val, Mesh::VCell::index_type idx) const
  { get_value(val,static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void get_value(T& val, Mesh::VElem::index_type idx) const
  { get_value(val,static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void get_value(T& val, Mesh::VDElem::index_type idx) const
  { get_value(val,static_cast<SCIRun::Mesh::index_type>(idx)); }

  virtual void set_value(const int &val, Mesh::index_type i);
  virtual void set_value(const double &val, Mesh::index_type i);
  virtual void set_value(const Vector &val, Mesh::index_type i);
  virtual void set_value(const Tensor &val, Mesh::index_type i);

  template<class T>  inline void set_value(const T& val, Mesh::VNode::index_type idx)
  { set_value(val,static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void set_value(const T& val, Mesh::VEdge::index_type idx)
  { set_value(val,static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void set_value(const T& val, Mesh::VFace::index_type idx)
  { set_value(val,static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void set_value(const T& val, Mesh::VCell::index_type idx)
  { set_value(val,static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void set_value(const T& val, Mesh::VElem::index_type idx)
  { set_value(val,static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void set_value(const T& val, Mesh::VDElem::index_type idx)
  { set_value(val,static_cast<SCIRun::Mesh::index_type>(idx)); }

  virtual void interpolate(double &val, const vector<double> &coords, Mesh::index_type elem_idx) const;
  virtual void interpolate(Vector &val, const vector<double> &coords, Mesh::index_type elem_idx) const;
  virtual void interpolate(Tensor &val, const vector<double> &coords, Mesh::index_type elem_idx) const;

  template<class T>  inline void interpolate(T& val, const vector<double> &coords, Mesh::VNode::index_type idx) const
  { interpolate(val, coords, static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void interpolate(T& val, const vector<double> &coords, Mesh::VEdge::index_type idx) const
  { interpolate(val, coords, static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void interpolate(T& val, const vector<double> &coords, Mesh::VFace::index_type idx) const
  { interpolate(val, coords, static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void interpolate(T& val, const vector<double> &coords, Mesh::VCell::index_type idx) const
  { interpolate(val, coords, static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void interpolate(T& val, const vector<double> &coords, Mesh::VElem::index_type idx) const
  { interpolate(val, coords, static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void interpolate(T& val, const vector<double> &coords, Mesh::VDElem::index_type idx) const
  { interpolate(val, coords, static_cast<SCIRun::Mesh::index_type>(idx)); }

  virtual void gradient(vector<double> &val, const vector<double> &coords, Mesh::index_type elem_idx) const;
  virtual void gradient(vector<Vector> &val, const vector<double> &coords, Mesh::index_type elem_idx) const;
  virtual void gradient(vector<Tensor> &val, const vector<double> &coords, Mesh::index_type elem_idx) const;

  template<class T>  inline void gradient(vector<T>& val, const vector<double> &coords, Mesh::VNode::index_type idx) const
  { gradient(val, coords, static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void gradient(vector<T>& val, const vector<double> &coords, Mesh::VEdge::index_type idx) const
  { gradient(val, coords, static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void gradient(vector<T>& val, const vector<double> &coords, Mesh::VFace::index_type idx) const
  { gradient(val, coords, static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void gradient(vector<T>& val, const vector<double> &coords, Mesh::VCell::index_type idx) const
  { gradient(val, coords, static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void gradient(vector<T>& val, const vector<double> &coords, Mesh::VElem::index_type idx) const
  { gradient(val, coords, static_cast<SCIRun::Mesh::index_type>(idx)); }
  template<class T>  inline void gradient(vector<T>& val, const vector<double> &coords, Mesh::VDElem::index_type idx) const
  { gradient(val, coords, static_cast<SCIRun::Mesh::index_type>(idx)); }

  virtual bool fdata_operation(const std::string& op, FDataResult& result) const;
  
  inline bool min(double& val, Mesh::index_type& idx) const
  { 
    FDataResult result; 
    if(fdata_operation("min",result)) 
    { val = result.scalar[0]; idx = result.index[0]; return (true); }
    return (false);
  }

  inline bool min(double& val) const
  { 
    FDataResult result; 
    if(fdata_operation("min",result)) 
    { val = result.scalar[0]; return (true); }
    return (false);
  }

  inline bool max(double& val, Mesh::index_type& idx) const
  { 
    FDataResult result; 
    if(fdata_operation("max",result)) 
    { val = result.scalar[0]; idx = result.index[0]; return (true); }
    return (false);
  }

  inline bool max(double& val) const
  { 
    FDataResult result; 
    if(fdata_operation("max",result)) 
    { val = result.scalar[0]; return (true); }
    return (false);
  }

  inline bool minmax(double& min, double& max, Mesh::index_type& minidx, Mesh::index_type& maxidx) const
  { 
    FDataResult result; 
    if(fdata_operation("minmax",result)) 
    { 
      min = result.scalar[0]; minidx = result.index[0]; 
      max = result.scalar[1]; maxidx = result.index[1];     
      return (true); 
    }
    return (false);
  }

  inline bool minmax(double& min, double& max) const
  { 
    FDataResult result; 
    if(fdata_operation("minmax",result)) 
    { min = result.scalar[0]; max = result.scalar[1]; return (true); }
    return (false);
  }

};

typedef LockingHandle<Field> FieldHandle;

class SCISHARE FieldTypeID {
  public:
    // Constructor
    FieldTypeID(const string& type, 
                FieldHandle (*field_maker)(),
                FieldHandle (*field_maker_mesh)(MeshHandle));
    
    string type;
    FieldHandle (*field_maker)();
    FieldHandle (*field_maker_mesh)(MeshHandle);
};


FieldHandle Create_Field(string type);
FieldHandle Create_Field(string type,MeshHandle mesh);


} // end namespace SCIRun

#endif // Datatypes_Field_h
















