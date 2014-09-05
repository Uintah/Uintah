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


/*
 *  GenericField.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_GenericField_h
#define Datatypes_GenericField_h

#include <Core/Datatypes/builtin.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Persistent/PersistentSTL.h>

namespace SCIRun {

template <class Mesh, class FData>
class GenericField: public Field 
{
public:
  //! Typedefs to support the Field concept.
  typedef typename FData::value_type      value_type;
  typedef Mesh                            mesh_type;
  typedef LockingHandle<mesh_type>        mesh_handle_type;
  typedef FData                           fdata_type;

  // only Pio should use this constructor
  GenericField();
  GenericField(data_location data_at);
  GenericField(mesh_handle_type mesh, data_location data_at);

  virtual ~GenericField();

  virtual GenericField<Mesh, FData> *clone() const;

  //! Required virtual functions from field base.
  virtual MeshHandle mesh() const;
  virtual void mesh_detach();

  virtual bool is_scalar() const;

  virtual const TypeDescription *data_at_type_description() const;

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

  virtual void resize_fdata();

  fdata_type& fdata();
  const fdata_type& fdata() const;

  const mesh_handle_type &get_typed_mesh() const;

  //! Persistent I/O.
  virtual void io(Piostream &stream);
  static  PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

  // -- mutability --
  virtual void freeze();
  virtual void thaw();

private:

  static Persistent *maker();

protected:

  //! A (generic) mesh.
  mesh_handle_type             mesh_;
  //! Data container.
  fdata_type                   fdata_;
}; 


template <class Mesh, class FData>
void
GenericField<Mesh, FData>::freeze()
{
  mesh_->freeze();
  // Call base class freeze..
  PropertyManager::freeze();
}

template <class Mesh, class FData>
void
GenericField<Mesh, FData>::thaw()
{
  // Call base class thaw..
  PropertyManager::thaw();
}

template <class Mesh, class FData>
void
GenericField<Mesh, FData>::resize_fdata()
{
  if (data_at() == NODE)
  {
    typename mesh_type::Node::size_type ssize;
    mesh_->synchronize(Mesh::NODES_E);
    mesh_->size(ssize);
    fdata().resize(ssize);
  }
  else if (data_at() == EDGE)
  {
    typename mesh_type::Edge::size_type ssize;
    mesh_->synchronize(Mesh::EDGES_E);
    mesh_->size(ssize);
    fdata().resize(ssize);
  }
  else if (data_at() == FACE)
  {
    typename mesh_type::Face::size_type ssize;
    mesh_->synchronize(Mesh::FACES_E);
    mesh_->size(ssize);
    fdata().resize(ssize);
  }
  else if (data_at() == CELL)
  {
    typename mesh_type::Cell::size_type ssize;
    mesh_->synchronize(Mesh::CELLS_E);
    mesh_->size(ssize);
    fdata().resize(ssize);
  }
  else if (data_at() == NONE)
  {
    // do nothing (really, we want to resize to zero)
  }
  else
  {
    ASSERTFAIL("data at unrecognized location");
  }
}


// PIO
const int GENERICFIELD_VERSION = 1;


template <class Mesh, class FData>
Persistent *
GenericField<Mesh, FData>::maker()
{
  return scinew GenericField<Mesh, FData>;
}

template <class Mesh, class FData>
PersistentTypeID 
GenericField<Mesh, FData>::type_id(type_name(-1), "Field", maker);


template <class Mesh, class FData>
void GenericField<Mesh, FData>::io(Piostream& stream)
{
  // we need to pass -1 to type_name() on SGI to fix a compile bug
  stream.begin_class(type_name(-1), GENERICFIELD_VERSION);
  Field::io(stream);
  mesh_->io(stream);
  mesh_->freeze();
  Pio(stream, fdata_);
  freeze();
  stream.end_class();
}


template <class Mesh, class FData>
GenericField<Mesh, FData>::GenericField() : 
  Field(),
  mesh_(mesh_handle_type(scinew mesh_type())),
  fdata_(0) //workaround for default variable bug on sgi.
{
  if (data_at() != NONE && mesh_.get_rep())
  {
    resize_fdata();
  }
}

template <class Mesh, class FData>
GenericField<Mesh, FData>::GenericField(data_location data_at) : 
  Field(data_at),
  mesh_(mesh_handle_type(scinew mesh_type())),
  fdata_(0) //workaround for default variable bug on sgi.
{
  if (data_at != NONE && mesh_.get_rep())
  { 
    resize_fdata();
  }
}

template <class Mesh, class FData>
GenericField<Mesh, FData>::GenericField(mesh_handle_type mesh, 
					data_location data_at) : 
  Field(data_at),
  mesh_(mesh),
  fdata_(0) //workaround for default variable bug on sgi.
{
  if (data_at != NONE && mesh_.get_rep())
  {
    resize_fdata();
  }
}

template <class Mesh, class FData>
GenericField<Mesh, FData>::~GenericField()
{
}

template <class Mesh, class FData>
GenericField<Mesh, FData> *
GenericField<Mesh, FData>::clone() const
{
  return new GenericField<Mesh, FData>(*this);
}

template <class Mesh, class FData>
MeshHandle
GenericField<Mesh, FData>::mesh() const
{
  return MeshHandle(mesh_.get_rep());
}

template <class Mesh, class FData>
void
GenericField<Mesh, FData>::mesh_detach()
{
  thaw();
  mesh_.detach();
  mesh_->thaw();
}

template <class Mesh, class FData>
bool
GenericField<Mesh, FData>::is_scalar() const
{
  return ::SCIRun::is_scalar<value_type>();
}

// Turn off warning for CHECKARRAYBOUNDS
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1183 1506
#endif

template <class Mesh, class FData>
bool
GenericField<Mesh, FData>::value(value_type &val, typename mesh_type::Node::index_type i) const
{
  ASSERTL3(data_at() == NODE);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  if (data_at() != NODE) return false; val = fdata_[i]; return true;
}

template <class Mesh, class FData>
bool
GenericField<Mesh, FData>::value(value_type &val, typename mesh_type::Edge::index_type i) const
{
  ASSERTL3(data_at() == EDGE);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  if (data_at() != EDGE) return false; val = fdata_[i]; return true;
}

template <class Mesh, class FData>
bool
GenericField<Mesh, FData>::value(value_type &val, typename mesh_type::Face::index_type i) const
{
  ASSERTL3(data_at() == FACE);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  if (data_at() != FACE) return false; val = fdata_[i]; return true;
}

template <class Mesh, class FData>
bool
GenericField<Mesh, FData>::value(value_type &val, typename mesh_type::Cell::index_type i) const
{
  ASSERTL3(data_at() == CELL);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  if (data_at() != CELL) return false; val = fdata_[i]; return true;
} 

//! Required interface to support Field Concept.
template <class Mesh, class FData>
void
GenericField<Mesh, FData>::set_value(const value_type &val, typename mesh_type::Node::index_type i)
{
  ASSERTL3(data_at() == NODE);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  fdata_[i] = val;
}
template <class Mesh, class FData>
void
GenericField<Mesh, FData>::set_value(const value_type &val, typename mesh_type::Edge::index_type i)
{
  ASSERTL3(data_at() == EDGE);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  fdata_[i] = val;
}
template <class Mesh, class FData>
void
GenericField<Mesh, FData>::set_value(const value_type &val, typename mesh_type::Face::index_type i)
{
  ASSERTL3(data_at() == FACE);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  fdata_[i] = val;
}
template <class Mesh, class FData>
void
GenericField<Mesh, FData>::set_value(const value_type &val, typename mesh_type::Cell::index_type i)
{
  ASSERTL3(data_at() == CELL);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  fdata_[i] = val;
}

template <class Mesh, class FData>
const typename GenericField<Mesh, FData>::value_type &
GenericField<Mesh, FData>::value(typename mesh_type::Node::index_type i) const
{
  ASSERTL3(data_at() == NODE);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  return fdata_[i];
}
template <class Mesh, class FData>
const typename GenericField<Mesh, FData>::value_type &
GenericField<Mesh, FData>::value(typename mesh_type::Edge::index_type i) const
{
  ASSERTL3(data_at() == EDGE);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  return fdata_[i];
}
template <class Mesh, class FData>
const typename GenericField<Mesh, FData>::value_type &
GenericField<Mesh, FData>::value(typename mesh_type::Face::index_type i) const 
{
  ASSERTL3(data_at() == FACE);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  return fdata_[i];
}
template <class Mesh, class FData>
const typename GenericField<Mesh, FData>::value_type &
GenericField<Mesh, FData>::value(typename mesh_type::Cell::index_type i) const 
{
  ASSERTL3(data_at() == CELL);
  CHECKARRAYBOUNDS(i, 0, fdata_.size());
  return fdata_[i];
}

// Reenable warning for CHECKARRAYBOUNDS
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1183 1506
#endif


template <class Mesh, class FData>
typename GenericField<Mesh, FData>::fdata_type &
GenericField<Mesh, FData>::fdata()
{
  return fdata_;
}

template <class Mesh, class FData>
const typename GenericField<Mesh, FData>::fdata_type&
GenericField<Mesh, FData>::fdata() const
{
  return fdata_;
}

template <class Mesh, class FData>
const typename GenericField<Mesh, FData>::mesh_handle_type &
GenericField<Mesh, FData>::get_typed_mesh() const
{
  return mesh_;
}

template <class Mesh, class FData>
const string GenericField<Mesh, FData>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 2);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNM
      + type_name(2) + FTNE;
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
  else
  {
    return find_type_name((FData *)0);
  }
}

template <class Mesh, class FData>
const TypeDescription *
GenericField<Mesh, FData>::get_type_description(int /*n*/) const
{
  ASSERTFAIL("TD MUST BE AT LEAF LEVEL OF INHERITENCE");
}

template <class Mesh, class FData>
const TypeDescription *
GenericField<Mesh, FData>::data_at_type_description() const
{
  switch(data_at())
  {
  case NODE:
    return SCIRun::get_type_description((typename Mesh::Node *)0);

  case EDGE:
    return SCIRun::get_type_description((typename Mesh::Edge *)0);

  case FACE:
    return SCIRun::get_type_description((typename Mesh::Face *)0);

  case CELL:
    return SCIRun::get_type_description((typename Mesh::Cell *)0);

  case NONE:
    // Default to least common denominator.
    return SCIRun::get_type_description((typename Mesh::Node *)0);
  }
  return 0;
}

} // end namespace SCIRun

#endif // Datatypes_GenericField_h
















