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



#ifndef CORE_DATATYPES_GENERICFIELD_H
#define CORE_DATATYPES_GENERICFIELD_H 1

#include <Core/Basis/Locate.h>
#include <Core/Containers/StackVector.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/builtin.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Datatypes/MeshTypes.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/FDataOperations.h>

#include <Core/Datatypes/share.h>

namespace SCIRun {

template <class Mesh, class Basis, class FData>
class GenericField: public Field 
{
public:
  //! Typedefs to support the Field concept.
  typedef GenericField<Mesh, Basis, FData>                 field_type;
  typedef typename FData::value_type                       value_type;
  typedef Mesh                                             mesh_type;
  typedef LockingHandle<mesh_type>                         mesh_handle_type;
  typedef Basis                                            basis_type;
  typedef FData                                            fdata_type;
  typedef LockingHandle<GenericField<Mesh, Basis, FData> > handle_type;

  //! only Pio should use this constructor
  GenericField();
  //! Use this constructor to actually have a field with a mesh
  GenericField(mesh_handle_type mesh);

  virtual ~GenericField();

  //! Clone the field data, but not the mesh.
  //! Use mesh_detach() first to clone the complete field
  virtual GenericField<Mesh, Basis, FData> *clone() const;

  //! Obtain a Handle to the Mesh
  virtual MeshHandle mesh() const;
  //! Clone the the mesh
  virtual void mesh_detach();

  virtual bool is_scalar() const;
  
  //! Get the size of the data stored in the data container
  virtual unsigned int data_size() const;

  //! OBSOLETE
  virtual const TypeDescription *order_type_description() const;

  //! Get the order of the field data
  //! -1 = no data
  //! 0 = constant data per element
  //! 1 = linear data per element
  //! >1 = non linear data per element
  virtual int basis_order() const { return basis_.polynomial_order(); }

  //! DIRECT ACCESS TO CONTAINER
  //! NOTE: We may change containe types in the future
  //! DIRECT ACCESS REQUIRES DYNAMIC COMPILATION
  fdata_type& fdata();
  const fdata_type& fdata() const;
  
  virtual void resize_fdata();
  virtual void resize_fdata(size_t size);

  //! Get the classes on which this function relies:
  //! Get the basis describing interpolation within an element
  Basis& get_basis()  { return basis_; }
  
  //! Get the mesh describing how the elements fit together
  const mesh_handle_type &get_typed_mesh() const;

  //! Persistent I/O.
  virtual void io(Piostream &stream);
  
  //! Tag the constructor of this class and put it in the Pio DataBase
  static  PersistentTypeID type_id;
  
  //! Tag the constructor of this class and put it in the Field DataBase
  static  FieldTypeID field_id;
  
  //! Function to retrieve the name of this field class
  static  const string type_name(int n = -1);
 
  //! A different way of tagging a class. Currently two systems are used next
  //! to each other: type_name and get_type_description. Neither is perfect
  virtual 
  const TypeDescription* get_type_description(td_info_e td = FULL_TD_E) const;

  // -- mutability --
  virtual void freeze();
  virtual void thaw();

  //! Static functions to instantiate the field from Pio or using Create_Field()
  static Persistent *maker();
  static FieldHandle field_maker();  
  static FieldHandle field_maker_mesh(MeshHandle mesh);
  
  //! Does the interface class have a complete virtual interface?
  //! Not all classes have and those still rely on dynamic compilation
  virtual bool has_virtual_interface();

  //! FIELD FUNCTIONS THAT RELY ON VIRTUAL OVERLOADING OF THE FIELD CLASS (NO DYNAMIC COMPILATION NEEDED)

  //! Get data from data location (Element or Node)
  virtual void get_value(int &val, SCIRun::Mesh::index_type i) const;
  virtual void get_value(double &val, SCIRun::Mesh::index_type i) const;
  virtual void get_value(Vector &val, SCIRun::Mesh::index_type i) const;
  virtual void get_value(Tensor &val, SCIRun::Mesh::index_type i) const;

  //! Set data at data location (Element or Node)
  virtual void set_value(const int &val, SCIRun::Mesh::index_type i);
  virtual void set_value(const double &val, SCIRun::Mesh::index_type i);
  virtual void set_value(const Vector &val, SCIRun::Mesh::index_type i);
  virtual void set_value(const Tensor &val, SCIRun::Mesh::index_type i);

  //! Compute value at arbitrary location
  //! Using the field variable basis interpolate a gradient within the 
  //! element, indicated at the paramentric coordinates coords.
  virtual void interpolate(double &val, const vector<double> &coords, SCIRun::Mesh::index_type elem_idx) const;
  virtual void interpolate(Vector &val, const vector<double> &coords, SCIRun::Mesh::index_type elem_idx) const;
  virtual void interpolate(Tensor &val, const vector<double> &coords, SCIRun::Mesh::index_type elem_idx) const;

  //! Compute gradient at arbitrary location
  //! Using the field variable basis interpolate a gradient within the 
  //! element, indicated at the paramentric coordinates coords.
  virtual void gradient(vector<double> &val, const vector<double> &coords, SCIRun::Mesh::index_type elem_idx) const;
  virtual void gradient(vector<Vector> &val, const vector<double> &coords, SCIRun::Mesh::index_type elem_idx) const;
  virtual void gradient(vector<Tensor> &val, const vector<double> &coords, SCIRun::Mesh::index_type elem_idx) const;

  virtual bool fdata_operation(const std::string& op, FDataResult& result) const;

  //! FIELD FUNCTIONS THAT RELY ON DYNAMIC COMPILATION (NOT VIRTUAL)
 
  //! Required interface to support Field Concept.
  bool value(value_type &val, typename mesh_type::Node::index_type i) const;
  bool value(value_type &val, typename mesh_type::Edge::index_type i) const;
  bool value(value_type &val, typename mesh_type::Face::index_type i) const;
  bool value(value_type &val, typename mesh_type::Cell::index_type i) const;

  //! Required interface to support Field Concept.
  void set_value(const value_type &val, typename mesh_type::Node::index_type i);
  void set_value(const value_type &val, typename mesh_type::Edge::index_type i);
  void set_value(const value_type &val, typename mesh_type::Face::index_type i);
  void set_value(const value_type &val, typename mesh_type::Cell::index_type i);

  //! No safety check for the following calls, be sure you know where data is.
  const value_type &value(typename mesh_type::Node::index_type i) const;
  const value_type &value(typename mesh_type::Edge::index_type i) const;
  const value_type &value(typename mesh_type::Face::index_type i) const;
  const value_type &value(typename mesh_type::Cell::index_type i) const;

  //! Using the field variable basis interpolate a value within the 
  //! element, indicated at the paramentric coordinates coords.
  void interpolate(value_type &val, const vector<double> &coords, 
		   typename mesh_type::Elem::index_type ei) const
  {

    ElemData<field_type> fcd(*this, ei);
    val = basis_.interpolate(coords, fcd);
  }

  //! Using the field variable basis interpolate a gradient within the 
  //! element, indicated at the paramentric coordinates coords.
  void  gradient(vector<value_type>& grad, const vector<double>& coords,
          typename mesh_type::Elem::index_type ci ) const;

  //! OBSOLETE: Function is improper
  //! NOTE: Do not use this function. Use gradient instead.
  void cell_gradient(typename mesh_type::Elem::index_type ci,
		     DenseMatrix *&grad) const;
 
private:
  friend class ElemData;

  template <class FLD>
  class ElemData
  {
  public:
    typedef typename FData::value_type value_type;

    ElemData(const FLD& fld, 
	     typename FLD::mesh_type::Elem::index_type idx) :
      fld_(fld),
      index_(idx)
    {
      fld_.mesh_->get_nodes(nodes_, idx);
      if (fld_.basis_order_ > 1) {
        fld_.mesh_->get_edges(edges_, idx);
      }
    }
    
    // basis may encode extra values based on cell index.
    unsigned elem_index() const { return index_; }
    
    inline 
    typename FData::value_type elem() const {
      return fld_.fdata_[index_];
    }

    inline 
    unsigned node0_index() const {
      return nodes_[0]; 
    }
    inline 
    unsigned node1_index() const {
      return nodes_[1];
    }
    inline 
    unsigned node2_index() const {
      return nodes_[2];
    }
    inline 
    unsigned node3_index() const {
      return nodes_[3];
    }
    inline 
    unsigned node4_index() const {
      return nodes_[4];
    }
    inline 
    unsigned node5_index() const {
      return nodes_[5];
    }
    inline 
    unsigned node6_index() const {
      return nodes_[6];
    }
    inline 
    unsigned node7_index() const {
      return nodes_[7];
    }

    inline 
    unsigned edge0_index() const {
      return edges_[0]; 
    }
    inline 
    unsigned edge1_index() const {
      return edges_[1];
    }
    inline 
    unsigned edge2_index() const {
      return edges_[2];
    }
    inline 
    unsigned edge3_index() const {
      return edges_[3];
    }
    inline 
    unsigned edge4_index() const {
      return edges_[4];
    }
    inline 
    unsigned edge5_index() const {
      return edges_[5];
    }
    inline 
    unsigned edge6_index() const {
      return edges_[6];
    }
    inline 
    unsigned edge7_index() const {
      return edges_[7];
    }
    inline 
    unsigned edge8_index() const {
      return edges_[8];
    }
    inline 
    unsigned edge9_index() const {
      return edges_[9];
    }
    inline 
    unsigned edge10_index() const {
      return edges_[10];
    }
    inline 
    unsigned edge11_index() const {
      return edges_[11];
    }

    inline 
    typename FData::value_type node0() const {
      return fld_.fdata_[nodes_[0]];
    }
    inline 
    typename FData::value_type node1() const {
      return fld_.fdata_[nodes_[1]];
    }
    inline 
    typename FData::value_type node2() const {
      return fld_.fdata_[nodes_[2]];
    }
    inline 
    typename FData::value_type node3() const {
      return fld_.fdata_[nodes_[3]];
    }
    inline 
    typename FData::value_type node4() const {
      return fld_.fdata_[nodes_[4]];
    }
    inline 
    typename FData::value_type node5() const {
      return fld_.fdata_[nodes_[5]];
    }
    inline 
    typename FData::value_type node6() const {
      return fld_.fdata_[nodes_[6]];
    }
    inline 
    typename FData::value_type node7() const {
      return fld_.fdata_[nodes_[7]];
    }


  private:
    const FLD                                   &fld_; //the field 
    typename FLD::mesh_type::Node::array_type    nodes_;
    typename FLD::mesh_type::Edge::array_type    edges_;
    typename FLD::mesh_type::Elem::index_type    index_;
  };


protected:

  //! A (generic) mesh.
  mesh_handle_type             mesh_;
  //! Data container.
  fdata_type                   fdata_;
  Basis                        basis_;
  
  int basis_order_;
  int mesh_dimensionality_;
}; 



template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::freeze()
{
  mesh_->freeze();
  // Call base class freeze..
  PropertyManager::freeze();
}

template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::thaw()
{
  // Call base class thaw..
  PropertyManager::thaw();
}

template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::resize_fdata()
{
  if (basis_order_ == 0 && mesh_dimensionality_ == 3)
  {
    typename mesh_type::Cell::size_type ssize;
    mesh_->synchronize(Mesh::CELLS_E);
    mesh_->size(ssize);
    fdata().resize(ssize);
  } 
  else if (basis_order_ == 0 && mesh_dimensionality_ == 2)
  {
    typename mesh_type::Face::size_type ssize;
    mesh_->synchronize(Mesh::FACES_E);
    mesh_->size(ssize);
    fdata().resize(ssize);
  } 
  else if (basis_order_ == 0 && mesh_dimensionality_ == 1)
  {
    typename mesh_type::Edge::size_type ssize;
    mesh_->synchronize(Mesh::EDGES_E);
    mesh_->size(ssize);
    fdata().resize(ssize);
  } 
  else if (basis_order_ == -1)
  {
    // do nothing (really, we want to resize to zero)
  }
  else
  {
    typename mesh_type::Node::size_type ssize;
    mesh_->synchronize(Mesh::NODES_E);
    mesh_->size(ssize);
    fdata().resize(ssize);
  }
}

template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::resize_fdata(size_t size)
{
  fdata().resize(size);
}



// PIO
const int GENERICFIELD_VERSION = 3;


template <class Mesh, class Basis, class FData>
Persistent *
GenericField<Mesh, Basis, FData>::maker()
{
  return scinew GenericField<Mesh, Basis, FData>;
}

template <class Mesh, class Basis, class FData>
FieldHandle
GenericField<Mesh, Basis, FData>::field_maker()
{
  return FieldHandle(scinew GenericField<Mesh, Basis, FData>());
}


template <class Mesh, class Basis, class FData>
FieldHandle
GenericField<Mesh, Basis, FData>::field_maker_mesh(MeshHandle mesh)
{
  mesh_handle_type mesh_handle = dynamic_cast<mesh_type *>(mesh.get_rep());
  if (mesh_handle.get_rep()) return FieldHandle(scinew GenericField<Mesh, Basis, FData>(mesh_handle));
  else return FieldHandle(0);
}



template <class Mesh, class Basis, class FData>
PersistentTypeID 
GenericField<Mesh, Basis, FData>::type_id(type_name(-1), "Field", maker);

template <class Mesh, class Basis, class FData>
FieldTypeID
GenericField<Mesh, Basis, FData>::field_id(type_name(-1),field_maker,field_maker_mesh);


template <class Mesh, class Basis, class FData>
void GenericField<Mesh, Basis, FData>::io(Piostream& stream)
{
  // we need to pass -1 to type_name() on SGI to fix a compile bug
  int version = stream.begin_class(type_name(-1), GENERICFIELD_VERSION);
  if (stream.backwards_compat_id()) {
    version = stream.begin_class(type_name(-1), GENERICFIELD_VERSION);
  }
  Field::io(stream);
  if (stream.error()) return;

  if (version < 2)
    mesh_->io(stream);
  else
    Pio(stream, mesh_);
  mesh_->freeze();
  if (version >= 3) { 
    basis_.io(stream);
  }
  Pio(stream, fdata_);
  freeze();

  if (stream.backwards_compat_id()) {
    stream.end_class();
  }
  stream.end_class();
}


template <class Mesh, class Basis, class FData>
GenericField<Mesh, Basis, FData>::GenericField() : 
  Field(),
  mesh_(mesh_handle_type(scinew mesh_type())),
  fdata_(0) //workaround for default variable bug on sgi.
{
  basis_order_ = basis_order();
  mesh_dimensionality_ = -1;
  if (mesh_.get_rep()) mesh_dimensionality_ = mesh_->dimensionality();
  
  if (basis_order_ != -1 && mesh_.get_rep())
  {
    resize_fdata();
  }
  
}

template <class Mesh, class Basis, class FData>
GenericField<Mesh, Basis, FData>::GenericField(mesh_handle_type mesh) : 
  Field(),
  mesh_(mesh),
  fdata_(0) //workaround for default variable bug on sgi.
{
  basis_order_ = basis_order();
  mesh_dimensionality_ = -1;
  if (mesh_.get_rep()) mesh_dimensionality_ = mesh_->dimensionality();
 
  if (basis_order_ != -1 && mesh_.get_rep())
  { 
    resize_fdata();
  }
}


template <class Mesh, class Basis, class FData>
GenericField<Mesh, Basis, FData>::~GenericField()
{
}

template <class Mesh, class Basis, class FData>
GenericField<Mesh, Basis, FData> *
GenericField<Mesh, Basis, FData>::clone() const
{
  return new GenericField<Mesh, Basis, FData>(*this);
}

template <class Mesh, class Basis, class FData>
MeshHandle
GenericField<Mesh, Basis, FData>::mesh() const
{
  return MeshHandle(mesh_.get_rep());
}

template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::mesh_detach()
{
  thaw();
  mesh_.detach();
  mesh_->thaw();
}

template <class Mesh, class Basis, class FData>
bool
GenericField<Mesh, Basis, FData>::is_scalar() const
{
  return ::SCIRun::is_scalar<value_type>();
}


template <class Mesh, class Basis, class FData>
unsigned int
GenericField<Mesh, Basis, FData>::data_size() const
{
  switch (basis_order_)
  {
  case -1:
    return 0;
    
  case 0:
    {
      typename mesh_type::Elem::size_type s;
      mesh_->size(s);
      return (unsigned int)s;
    }
  default:
    {
      typename mesh_type::Node::size_type s;
      mesh_->size(s);
      return (unsigned int)s;
    }
  }
}


// Turn off warning for CHECKARRAYBOUNDS
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1183 1506
#endif

template <class Mesh, class Basis, class FData>
bool
GenericField<Mesh, Basis, FData>::value(value_type &val, 
				 typename mesh_type::Node::index_type i) const
{
  ASSERTL3(basis_order_ >= 1 || mesh_dimensionality_ == 0);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  if (!(basis_order_ == 1 || mesh_dimensionality_ == 0)) return false;
  val = fdata_[i]; return true;
}

template <class Mesh, class Basis, class FData>
bool
GenericField<Mesh, Basis, FData>::value(value_type &val, 
				 typename mesh_type::Edge::index_type i) const
{
  ASSERTL3(basis_order_ == 0 && mesh_dimensionality_ == 1);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  if (basis_order_ != 0) return false;
  val = fdata_[i]; return true;
}

template <class Mesh, class Basis, class FData>
bool
GenericField<Mesh, Basis, FData>::value(value_type &val, 
				 typename mesh_type::Face::index_type i) const
{
  ASSERTL3(basis_order_ == 0 && mesh_dimensionality_ == 2);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  if (basis_order_ != 0) return false;
  val = fdata_[i]; return true;
}

template <class Mesh, class Basis, class FData>
bool
GenericField<Mesh, Basis, FData>::value(value_type &val, 
				 typename mesh_type::Cell::index_type i) const
{
  ASSERTL3(basis_order_ == 0 && mesh_dimensionality_ == 3);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  if (basis_order_ != 0) return false;
  val = fdata_[i]; return true;
} 

//! Required interface to support Field Concept.
template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::set_value(const value_type &val, 
				      typename mesh_type::Node::index_type i)
{
  ASSERTL3(basis_order_ >= 1 || mesh_dimensionality_ == 0);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  fdata_[i] = val;
}
template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::set_value(const value_type &val, 
			       typename mesh_type::Edge::index_type i)
{
  ASSERTL3(basis_order_ == 0 && mesh_dimensionality_ == 1);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  fdata_[i] = val;
}
template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::set_value(const value_type &val, 
				      typename mesh_type::Face::index_type i)
{
  ASSERTL3(basis_order_ == 0 && mesh_dimensionality_ == 2);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  fdata_[i] = val;
}
template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::set_value(const value_type &val, 
				      typename mesh_type::Cell::index_type i)
{
  ASSERTL3(basis_order_ == 0 && mesh_dimensionality_ == 3);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  fdata_[i] = val;
}

template <class Mesh, class Basis, class FData>
const typename GenericField<Mesh, Basis, FData>::value_type &
GenericField<Mesh, Basis, FData>::
value(typename mesh_type::Node::index_type i) const
{
  ASSERTL3(basis_order_ >= 1 || mesh_dimensionality_ == 0);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  return fdata_[i];
}

template <class Mesh, class Basis, class FData>
const typename GenericField<Mesh, Basis, FData>::value_type &
GenericField<Mesh, Basis, FData>::
value(typename mesh_type::Edge::index_type i) const
{
  ASSERTL3(basis_order_ == 0 && mesh_dimensionality_ == 1);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  return fdata_[i];
}
template <class Mesh, class Basis, class FData>
const typename GenericField<Mesh, Basis, FData>::value_type &
GenericField<Mesh, Basis, FData>::
value(typename mesh_type::Face::index_type i) const 
{
  ASSERTL3(basis_order_ == 0 && mesh_dimensionality_ == 2);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  return fdata_[i];
}
template <class Mesh, class Basis, class FData>
const typename GenericField<Mesh, Basis, FData>::value_type &
GenericField<Mesh, Basis, FData>::
value(typename mesh_type::Cell::index_type i) const 
{
  ASSERTL3(basis_order_ == 0 && mesh_dimensionality_ == 3);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  return fdata_[i];
}

// Reenable warning for CHECKARRAYBOUNDS
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1183 1506
#endif

template <class Mesh, class Basis, class FData>
typename GenericField<Mesh, Basis, FData>::fdata_type &
GenericField<Mesh, Basis, FData>::fdata()
{
  return fdata_;
}

template <class Mesh, class Basis, class FData>
const typename GenericField<Mesh, Basis, FData>::fdata_type&
GenericField<Mesh, Basis, FData>::fdata() const
{
  return fdata_;
}

template <class Mesh, class Basis, class FData>
const typename GenericField<Mesh, Basis, FData>::mesh_handle_type &
GenericField<Mesh, Basis, FData>::get_typed_mesh() const
{
  return mesh_;
}

template <class Mesh, class Basis, class FData>
const string GenericField<Mesh, Basis, FData>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 3);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNM
      + type_name(2) + FTNM + type_name(3) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("GenericField");
    return nm;
  }
  else if (n == 1)
  {
    return find_type_name((Mesh *)0);
  }
  else if (n == 2)
  {
    return find_type_name((Basis *)0);
  }
  else
  {
    return find_type_name((FData *)0);
  }
}

template <class Mesh, class Basis, class FData>
const TypeDescription *
GenericField<Mesh, Basis, FData>::get_type_description(td_info_e td) const
{
  static string name(type_name(0));
  static string namesp("SCIRun");
  static string path(__FILE__);
  const TypeDescription *sub1 = SCIRun::get_type_description((Mesh*)0);
  const TypeDescription *sub2 = SCIRun::get_type_description((Basis*)0);
  const TypeDescription *sub3 = SCIRun::get_type_description((FData*)0);

  switch (td) {
  default:
  case FULL_TD_E:
    {
      static TypeDescription* tdn1 = 0;
      if (tdn1 == 0) {
	TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(3);
	(*subs)[0] = sub1;
	(*subs)[1] = sub2;
	(*subs)[2] = sub3;
	tdn1 = scinew TypeDescription(name, subs, path, namesp, 
				      TypeDescription::FIELD_E);
      } 
      return tdn1;
    }
  case FIELD_NAME_ONLY_E:
    {
      static TypeDescription* tdn0 = 0;
      if (tdn0 == 0) {
	tdn0 = scinew TypeDescription(name, 0, path, namesp, 
				      TypeDescription::FIELD_E);
      }
      return tdn0;
    }
  case MESH_TD_E:
    {
      return sub1;
    }
  case BASIS_TD_E:
    {
      return sub2;
    }
  case FDATA_TD_E:
    {
      return sub3;
    }
  };
}




// Unfortunately even for the linear case gradients are not necessarily 
// constant. Hence provide a means of getting the real gradient at a
// location. This method is similar to interpolate but returns the local
// gradient instead.

template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::
gradient(vector<value_type>& grad, const vector<double>& coords, 
                     typename mesh_type::Elem::index_type ci) const
{
  grad.resize(3);
 
  ElemData<field_type> fcd(*this, ci);
  // derivative is constant anywhere in the linear cell
  
  // get the mesh Jacobian for the element.
  StackVector<Point,3> Jv;
  mesh_->derivate(coords, ci, Jv);

  int dim = basis_.domain_dimension();
  double J[9], Ji[9];

  // TO DO:
  // Squeeze out more STL vector operations as they require memory
  // being reserved, we should have simple C style arrays which are build
  // directly on the stack. As this is mostly used for volume data, it has 
  // only been optimized for this kind of data

  ASSERT(dim >=1 && dim <=3);
  if (dim == 3)
  {
    J[0] = Jv[0].x();
    J[1] = Jv[0].y();
    J[2] = Jv[0].z();
    J[3] = Jv[1].x();
    J[4] = Jv[1].y();
    J[5] = Jv[1].z();
    J[6] = Jv[2].x();
    J[7] = Jv[2].y();
    J[8] = Jv[2].z();        
  }
  else if (dim == 2)
  {
    Vector J2 = Cross(Jv[0].asVector(),Jv[1].asVector());
    J2.normalize();
    J[0] = Jv[0].x();
    J[1] = Jv[0].y();
    J[2] = Jv[0].z();
    J[3] = Jv[1].x();
    J[4] = Jv[1].y();
    J[5] = Jv[1].z();
    J[6] = J2.x();
    J[7] = J2.y();
    J[8] = J2.z();    
  }
  else
  {
    // The same thing as for the surface but then for a curve.
    // Again this matrix should have a positive determinant as well. It actually
    // has an internal degree of freedom, which is not being used.
    Vector J1, J2;
    Jv[0].asVector().find_orthogonal(J1,J2);
    J[0] = Jv[0].x();
    J[1] = Jv[0].y();
    J[2] = Jv[0].z();
    J[3] = J1.x();
    J[4] = J1.y();
    J[5] = J1.z();
    J[6] = J2.x();
    J[7] = J2.y();
    J[8] = J2.z();          
  }

  InverseMatrix3x3(J,Ji);
  
  StackVector<value_type,3> g;
  basis_.derivate(coords, fcd, g);  

  if (g.size() == 3)
  {
    grad[0] = static_cast<value_type>(g[0]*Ji[0])+static_cast<value_type>(g[1]*Ji[1])+static_cast<value_type>(g[2]*Ji[2]);
    grad[1] = static_cast<value_type>(g[0]*Ji[3])+static_cast<value_type>(g[1]*Ji[4])+static_cast<value_type>(g[2]*Ji[5]);
    grad[2] = static_cast<value_type>(g[0]*Ji[6])+static_cast<value_type>(g[1]*Ji[7])+static_cast<value_type>(g[2]*Ji[8]);
  }
  else if (g.size() == 2)
  {
    grad[0] = static_cast<value_type>(g[0]*Ji[0])+static_cast<value_type>(g[1]*Ji[1]);
    grad[1] = static_cast<value_type>(g[0]*Ji[3])+static_cast<value_type>(g[1]*Ji[4]);
    grad[2] = static_cast<value_type>(g[0]*Ji[6])+static_cast<value_type>(g[1]*Ji[7]);
  }
  else if (g.size() == 1)
  {
    grad[0] = static_cast<value_type>(g[0]*Ji[0]);
    grad[1] = static_cast<value_type>(g[0]*Ji[3]);
    grad[2] = static_cast<value_type>(g[0]*Ji[6]);  
  }
}


// DO NOT USE cell_gradient
// This implementation is limitted to volumetric data
// and inproperly assumes gradients to be constant
// This function only works properly for a TetVolMesh
// It is still here for compatibility reasons

template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::
cell_gradient(typename mesh_type::Elem::index_type ci,
	      DenseMatrix *&grad) const
{
  // supported for linear, should be expanded to support higher order.
  ASSERT(basis_order_ == 1);

  ElemData<field_type> fcd(*this, ci);
  // derivative is constant anywhere in the linear cell
  vector<double> coords(3);
  coords[0] = 0.0;
  coords[1] = 0.0;
  coords[2] = 0.0;
  
  // get the mesh Jacobian for the element.
  vector<Point> Jv;
  mesh_->derivate(coords, ci, Jv);


  // load the matrix with the Jacobian
  DenseMatrix J(3, Jv.size());
  int i = 0;
  vector<Point>::iterator iter = Jv.begin();
  while(iter != Jv.end()) {
    Point &p = *iter++;
    J.put(i, 0, p.x());
    J.put(i, 1, p.y());
    J.put(i, 2, p.z());
    ++i;
  }
  J.invert();

  vector<value_type> g;
  basis_.derivate(coords, fcd, g);
  unsigned int n = g.size();
  unsigned int m = get_vsize((value_type *)0);
  DenseMatrix local(n, m);
  grad = scinew DenseMatrix(n, m);
  load_partials(g, local);

  Mult(*grad, J, local);
}

// Functions for cell_gradient
// These should go as soon as cell_gradient
// has been replaced

template <class T>
unsigned int get_vsize(T*);

template <>
SCISHARE unsigned int get_vsize(Vector*);

template <>
SCISHARE unsigned int get_vsize(Tensor*);

//size for scalars
template <class T>
unsigned int get_vsize(T*)
{
  return 1;
}

template <class T>
void
load_partials(const vector<T> &grad, DenseMatrix &m);

template <>
SCISHARE void
load_partials(const vector<Vector> &grad, DenseMatrix &m);

template <>
SCISHARE void
load_partials(const vector<Tensor> &grad, DenseMatrix &m);


//scalar version
template <class T>
void
load_partials(const vector<T> &grad, DenseMatrix &m)
{
  int i = 0;
  typename vector<T>::const_iterator iter = grad.begin();
  while(iter != grad.end()) {
    const T &v = *iter++;
    m.put(i, 0, (double)v);
    ++i;
  }
}

///

template <class Mesh, class Basis, class FData>
const TypeDescription *
GenericField<Mesh, Basis, FData>::order_type_description() const
{
  const int order = basis_order_;
  const int dim = mesh_dimensionality_;
  if (order == 0 && dim == 3) 
  {
    return Mesh::cell_type_description();
  } 
  else if (order == 0 && dim == 2) 
  {
    return Mesh::face_type_description();
  } 
  else if (order == 0 && dim == 1) 
  {
    return Mesh::edge_type_description();
  } 
  else 
  {
    return Mesh::node_type_description();
  }
}


template <class Mesh, class Basis, class FData>
bool
GenericField<Mesh, Basis, FData>::has_virtual_interface()
{
  FDataResult dummy; 
  return (fdata_operation("test",dummy) && mesh_->has_virtual_interface());
}

template <class T> inline T CastFData(const char &val);
template <class T> inline T CastFData(const unsigned char &val);
template <class T> inline T CastFData(const short &val);
template <class T> inline T CastFData(const unsigned short &val);
template <class T> inline T CastFData(const int &val);
template <class T> inline T CastFData(const unsigned int &val);
template <class T> inline T CastFData(const long &val);
template <class T> inline T CastFData(const unsigned long &val);
template <class T> inline T CastFData(const long long &val);
template <class T> inline T CastFData(const unsigned long long &val);
template <class T> inline T CastFData(const float &val);
template <class T> inline T CastFData(const double &val);
template <class T> inline T CastFData(const Vector &val);
template <class T> inline T CastFData(const Tensor &val);

template <class T> inline T CastFData(const char &val) { return (static_cast<T>(val)); }
template <> inline Vector CastFData<Vector>(const char &val) { return (Vector(0,0,0)); }
template <> inline Tensor CastFData<Tensor>(const char &val) { return (Tensor(static_cast<double>(val))); }

template <class T> inline T CastFData(const unsigned char &val) { return (static_cast<T>(val)); }
template <> inline Vector CastFData<Vector>(const unsigned char &val) { return (Vector(0,0,0)); }
template <> inline Tensor CastFData<Tensor>(const unsigned char &val) { return (Tensor(static_cast<double>(val))); }

template <class T> inline T CastFData(const short &val) { return (static_cast<T>(val)); }
template <> inline Vector CastFData<Vector>(const short &val) {return (Vector(0,0,0));}
template <> inline Tensor CastFData<Tensor>(const short &val) { return (Tensor(static_cast<double>(val))); }

template <class T> inline T CastFData(const unsigned short &val) { return (static_cast<T>(val)); }
template <> inline Vector CastFData<Vector>(const unsigned short &val) { return (Vector(0,0,0));}
template <> inline Tensor CastFData<Tensor>(const unsigned short &val) { return (Tensor(static_cast<double>(val))); }

template <class T> inline T CastFData(const int &val) { return (static_cast<T>(val)); }
template <> inline Vector CastFData<Vector>(const int &val) { return (Vector(0,0,0)); }
template <> inline Tensor CastFData<Tensor>(const int &val) { return (Tensor(static_cast<double>(val))); }

template <class T> inline T CastFData(const unsigned int &val) { return (static_cast<T>(val)); }
template <> inline Vector CastFData<Vector>(const unsigned int &val) { return (Vector(0,0,0)); }
template <> inline Tensor CastFData<Tensor>(const unsigned int &val) { return (Tensor(static_cast<double>(val))); }

template <class T> inline T CastFData(const long &val) { return (static_cast<T>(val)); }
template <> inline Vector CastFData<Vector>(const long &val) { return (Vector(0,0,0)); }
template <> inline Tensor CastFData<Tensor>(const long &val) { return (Tensor(static_cast<double>(val))); }

template <class T> inline T CastFData(const unsigned long &val) { return (static_cast<T>(val)); }
template <> inline Vector CastFData<Vector>(const unsigned long &val) { return (Vector(0,0,0)); }
template <> inline Tensor CastFData<Tensor>(const unsigned long &val) { return (Tensor(static_cast<double>(val))); }

template <class T> inline T CastFData(const long long &val) { return (static_cast<T>(val)); }
template <> inline Vector CastFData<Vector>(const long long &val) { return (Vector(0,0,0)); }
template <> inline Tensor CastFData<Tensor>(const long long &val) { return (Tensor(static_cast<double>(val))); }

template <class T> inline T CastFData(const unsigned long long &val) { return (static_cast<T>(val)); }
template <> inline Vector CastFData<Vector>(const unsigned long long &val) { return (Vector(0,0,0)); }
template <> inline Tensor CastFData<Tensor>(const unsigned long long &val) { return (Tensor(static_cast<double>(val))); }

template <class T> inline T CastFData(const float &val) { return (static_cast<T>(val)); }
template <> inline Vector CastFData<Vector>(const float &val) { return (Vector(0,0,0)); }
template <> inline Tensor CastFData<Tensor>(const float &val) { return (Tensor(static_cast<double>(val))); }

template <class T> inline T CastFData(const double &val) { return (static_cast<T>(val)); }
template <> inline Vector CastFData<Vector>(const double &val) { return (Vector(0,0,0)); }
template <> inline Tensor CastFData<Tensor>(const double &val) { return (Tensor(static_cast<double>(val))); }

template <class T> inline T CastFData(const Vector &val) { return (0); }
template <> inline Vector CastFData<Vector>(const Vector &val) { return (val); }
template <> inline Tensor CastFData<Tensor>(const Vector &val) { return (Tensor(0.0)); }

template <class T> inline T CastFData(const Tensor &val) { return (0); }
template <> inline Vector CastFData<Vector>(const Tensor &val) { return (Vector(0,0,0)); }
template <> inline Tensor CastFData<Tensor>(const Tensor &val) { return (val); }


template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::get_value(int &val, SCIRun::Mesh::index_type i) const
{
//  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  val = CastFData<int>(fdata_[i]);
}

template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::get_value(double &val, SCIRun::Mesh::index_type i) const
{
//  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  val = CastFData<double>(fdata_[i]);
}

template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::get_value(Vector &val, SCIRun::Mesh::index_type i) const
{
//  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  val = CastFData<Vector>(fdata_[i]);
}

template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::get_value(Tensor &val, SCIRun::Mesh::index_type i) const
{
//  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  val = CastFData<Tensor>(fdata_[i]);
}


template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::set_value(const int &val, SCIRun::Mesh::index_type i)
{
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  fdata_[i] = CastFData<value_type>(val);
}

template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::set_value(const double &val, SCIRun::Mesh::index_type i)
{
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  fdata_[i] = CastFData<value_type>(val);
}

template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::set_value(const Vector &val, SCIRun::Mesh::index_type i)
{
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  fdata_[i] = CastFData<value_type>(val);
}

template <class Mesh, class Basis, class FData>
void
GenericField<Mesh, Basis, FData>::set_value(const Tensor &val, SCIRun::Mesh::index_type i)
{
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  fdata_[i] = CastFData<value_type>(val);
}


template <class Mesh, class Basis, class FData>
void 
GenericField<Mesh, Basis, FData>::interpolate(double &val, const vector<double> &coords, SCIRun::Mesh::index_type elem_idx) const
{
  typename mesh_type::Elem::index_type ei;
  mesh_->to_index(ei,elem_idx);
  ElemData<field_type> fcd(*this, ei);
  val = CastFData<double>(basis_.interpolate(coords, fcd));
}

template <class Mesh, class Basis, class FData>
void 
GenericField<Mesh, Basis, FData>::interpolate(Vector &val, const vector<double> &coords, SCIRun::Mesh::index_type elem_idx) const
{
  typename mesh_type::Elem::index_type ei;
  mesh_->to_index(ei,elem_idx);
  ElemData<field_type> fcd(*this, ei);
  val = CastFData<Vector>(basis_.interpolate(coords, fcd));
}

template <class Mesh, class Basis, class FData>
void 
GenericField<Mesh, Basis, FData>::interpolate(Tensor &val, const vector<double> &coords, SCIRun::Mesh::index_type elem_idx) const
{
  typename mesh_type::Elem::index_type ei;
  mesh_->to_index(ei,elem_idx);
  ElemData<field_type> fcd(*this, ei);
  val = CastFData<Tensor>(basis_.interpolate(coords, fcd));
}


template <class Mesh, class Basis, class FData>
void 
GenericField<Mesh, Basis, FData>::gradient(vector<double> &grad, const vector<double> &coords, SCIRun::Mesh::index_type elem_idx) const
{ 
  grad.resize(3);

  typename mesh_type::Elem::index_type ei;
  mesh_->to_index(ei,elem_idx); 
  ElemData<field_type> fcd(*this, ei);
  // derivative is constant anywhere in the linear cell
  
  // get the mesh Jacobian for the element.
  StackVector<Point,3> Jv;
  mesh_->derivate(coords, ei, Jv);

  int dim = basis_.domain_dimension();
  double J[9], Ji[9];

  // TO DO:
  // Squeeze out more STL vector operations as they require memory
  // being reserved, we should have simple C style arrays which are build
  // directly on the stack. As this is mostly used for volume data, it has 
  // only been optimized for this kind of data

  ASSERT(dim >=1 && dim <=3);
  if (dim == 3)
  {
    J[0] = Jv[0].x(); J[1] = Jv[0].y(); J[2] = Jv[0].z();
    J[3] = Jv[1].x(); J[4] = Jv[1].y(); J[5] = Jv[1].z();
    J[6] = Jv[2].x(); J[7] = Jv[2].y(); J[8] = Jv[2].z();        
  }
  else if (dim == 2)
  {
    Vector J2 = Cross(Jv[0].asVector(),Jv[1].asVector());
    J2.normalize();
    J[0] = Jv[0].x(); J[1] = Jv[0].y(); J[2] = Jv[0].z();
    J[3] = Jv[1].x(); J[4] = Jv[1].y(); J[5] = Jv[1].z();
    J[6] = J2.x(); J[7] = J2.y(); J[8] = J2.z();    
  }
  else
  {
    // The same thing as for the surface but then for a curve.
    // Again this matrix should have a positive determinant as well. It actually
    // has an internal degree of freedom, which is not being used.
    Vector J1, J2;
    Jv[0].asVector().find_orthogonal(J1,J2);
    J[0] = Jv[0].x(); J[1] = Jv[0].y(); J[2] = Jv[0].z();
    J[3] = J1.x(); J[4] = J1.y(); J[5] = J1.z();
    J[6] = J2.x(); J[7] = J2.y(); J[8] = J2.z();          
  }

  InverseMatrix3x3(J,Ji);
  StackVector<value_type,3> g;
  basis_.derivate(coords, fcd, g);  
  
  if (g.size() == 3)
  {
    grad[0] = CastFData<double>(g[0]*Ji[0]+g[1]*Ji[1]+g[2]*Ji[2]);
    grad[1] = CastFData<double>(g[0]*Ji[3]+g[1]*Ji[4]+g[2]*Ji[5]);
    grad[2] = CastFData<double>(g[0]*Ji[6]+g[1]*Ji[7]+g[2]*Ji[8]);
  }
  else if (g.size() == 2)
  {
    grad[0] = CastFData<double>(g[0]*Ji[0]+g[1]*Ji[1]);
    grad[1] = CastFData<double>(g[0]*Ji[3]+g[1]*Ji[4]);
    grad[2] = CastFData<double>(g[0]*Ji[6]+g[1]*Ji[7]);
  }
  else if (g.size() == 1)
  {
    grad[0] = CastFData<double>(g[0]*Ji[0]);
    grad[1] = CastFData<double>(g[0]*Ji[3]);
    grad[2] = CastFData<double>(g[0]*Ji[6]);  
  }
}


template <class Mesh, class Basis, class FData>
void 
GenericField<Mesh, Basis, FData>::gradient(vector<Vector> &grad, const vector<double> &coords, SCIRun::Mesh::index_type elem_idx) const
{ 
  grad.resize(3);

  typename mesh_type::Elem::index_type ei;
  mesh_->to_index(ei,elem_idx); 
  ElemData<field_type> fcd(*this, ei);
  // derivative is constant anywhere in the linear cell
  
  // get the mesh Jacobian for the element.
  StackVector<Point,3> Jv;
  mesh_->derivate(coords, ei, Jv);

  int dim = basis_.domain_dimension();
  double J[9], Ji[9];

  // TO DO:
  // Squeeze out more STL vector operations as they require memory
  // being reserved, we should have simple C style arrays which are build
  // directly on the stack. As this is mostly used for volume data, it has 
  // only been optimized for this kind of data

  ASSERT(dim >=1 && dim <=3);
  if (dim == 3)
  {
    J[0] = Jv[0].x(); J[1] = Jv[0].y(); J[2] = Jv[0].z();
    J[3] = Jv[1].x(); J[4] = Jv[1].y(); J[5] = Jv[1].z();
    J[6] = Jv[2].x(); J[7] = Jv[2].y(); J[8] = Jv[2].z();        
  }
  else if (dim == 2)
  {
    Vector J2 = Cross(Jv[0].asVector(),Jv[1].asVector());
    J2.normalize();
    J[0] = Jv[0].x(); J[1] = Jv[0].y(); J[2] = Jv[0].z();
    J[3] = Jv[1].x(); J[4] = Jv[1].y(); J[5] = Jv[1].z();
    J[6] = J2.x(); J[7] = J2.y(); J[8] = J2.z();    
  }
  else
  {
    // The same thing as for the surface but then for a curve.
    // Again this matrix should have a positive determinant as well. It actually
    // has an internal degree of freedom, which is not being used.
    Vector J1, J2;
    Jv[0].asVector().find_orthogonal(J1,J2);
    J[0] = Jv[0].x(); J[1] = Jv[0].y(); J[2] = Jv[0].z();
    J[3] = J1.x(); J[4] = J1.y(); J[5] = J1.z();
    J[6] = J2.x(); J[7] = J2.y(); J[8] = J2.z();          
  }

  InverseMatrix3x3(J,Ji);
  StackVector<value_type,3> g;
  basis_.derivate(coords, fcd, g);  
  
  if (g.size() == 3)
  {
    grad[0] = CastFData<Vector>(g[0]*Ji[0]+g[1]*Ji[1]+g[2]*Ji[2]);
    grad[1] = CastFData<Vector>(g[0]*Ji[3]+g[1]*Ji[4]+g[2]*Ji[5]);
    grad[2] = CastFData<Vector>(g[0]*Ji[6]+g[1]*Ji[7]+g[2]*Ji[8]);
  }
  else if (g.size() == 2)
  {
    grad[0] = CastFData<Vector>(g[0]*Ji[0]+g[1]*Ji[1]);
    grad[1] = CastFData<Vector>(g[0]*Ji[3]+g[1]*Ji[4]);
    grad[2] = CastFData<Vector>(g[0]*Ji[6]+g[1]*Ji[7]);
  }
  else if (g.size() == 1)
  {
    grad[0] = CastFData<Vector>(g[0]*Ji[0]);
    grad[1] = CastFData<Vector>(g[0]*Ji[3]);
    grad[2] = CastFData<Vector>(g[0]*Ji[6]);  
  }
}

template <class Mesh, class Basis, class FData>
void 
GenericField<Mesh, Basis, FData>::gradient(vector<Tensor> &grad, const vector<double> &coords, SCIRun::Mesh::index_type elem_idx) const
{ 
  grad.resize(3);

  typename mesh_type::Elem::index_type ei;
  mesh_->to_index(ei,elem_idx); 
  ElemData<field_type> fcd(*this, ei);
  // derivative is constant anywhere in the linear cell
  
  // get the mesh Jacobian for the element.
  StackVector<Point,3> Jv;
  mesh_->derivate(coords, ei, Jv);

  int dim = basis_.domain_dimension();
  double J[9], Ji[9];

  // TO DO:
  // Squeeze out more STL vector operations as they require memory
  // being reserved, we should have simple C style arrays which are build
  // directly on the stack. As this is mostly used for volume data, it has 
  // only been optimized for this kind of data

  ASSERT(dim >=1 && dim <=3);
  if (dim == 3)
  {
    J[0] = Jv[0].x(); J[1] = Jv[0].y(); J[2] = Jv[0].z();
    J[3] = Jv[1].x(); J[4] = Jv[1].y(); J[5] = Jv[1].z();
    J[6] = Jv[2].x(); J[7] = Jv[2].y(); J[8] = Jv[2].z();        
  }
  else if (dim == 2)
  {
    Vector J2 = Cross(Jv[0].asVector(),Jv[1].asVector());
    J2.normalize();
    J[0] = Jv[0].x(); J[1] = Jv[0].y(); J[2] = Jv[0].z();
    J[3] = Jv[1].x(); J[4] = Jv[1].y(); J[5] = Jv[1].z();
    J[6] = J2.x(); J[7] = J2.y(); J[8] = J2.z();    
  }
  else
  {
    // The same thing as for the surface but then for a curve.
    // Again this matrix should have a positive determinant as well. It actually
    // has an internal degree of freedom, which is not being used.
    Vector J1, J2;
    Jv[0].asVector().find_orthogonal(J1,J2);
    J[0] = Jv[0].x(); J[1] = Jv[0].y(); J[2] = Jv[0].z();
    J[3] = J1.x(); J[4] = J1.y(); J[5] = J1.z();
    J[6] = J2.x(); J[7] = J2.y(); J[8] = J2.z();          
  }

  InverseMatrix3x3(J,Ji);
  StackVector<value_type,3> g;
  basis_.derivate(coords, fcd, g);  
  
  if (g.size() == 3)
  {
    grad[0] = CastFData<Tensor>(g[0]*Ji[0]+g[1]*Ji[1]+g[2]*Ji[2]);
    grad[1] = CastFData<Tensor>(g[0]*Ji[3]+g[1]*Ji[4]+g[2]*Ji[5]);
    grad[2] = CastFData<Tensor>(g[0]*Ji[6]+g[1]*Ji[7]+g[2]*Ji[8]);
  }
  else if (g.size() == 2)
  {
    grad[0] = CastFData<Tensor>(g[0]*Ji[0]+g[1]*Ji[1]);
    grad[1] = CastFData<Tensor>(g[0]*Ji[3]+g[1]*Ji[4]);
    grad[2] = CastFData<Tensor>(g[0]*Ji[6]+g[1]*Ji[7]);
  }
  else if (g.size() == 1)
  {
    grad[0] = CastFData<Tensor>(g[0]*Ji[0]);
    grad[1] = CastFData<Tensor>(g[0]*Ji[3]);
    grad[2] = CastFData<Tensor>(g[0]*Ji[6]);  
  }
}

template <class Mesh, class Basis, class FData>
bool 
GenericField<Mesh, Basis, FData>::fdata_operation(const std::string& op, FDataResult& result) const
{
  return(FData_operation(op,fdata_,result));
}


} // end namespace SCIRun






#endif // Datatypes_GenericField_h
