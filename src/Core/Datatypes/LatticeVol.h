
#ifndef Datatypes_LatticeVol_h
#define Datatypes_LatticeVol_h

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array3.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <string>

namespace SCIRun {

using std::string;


template <class Data>
class FData3d : public Array3<Data> {
public:
  typedef Data value_type;
  typedef Data * iterator;

  Data *begin() { return &(*this)(0,0,0); }
  Data *end() { return &((*this)(dim1()-1,dim2()-1,dim3()-1))+1; }
    
  FData3d():Array3<Data>(){}
  virtual ~FData3d(){}
  
  const value_type &operator[](typename LatVolMesh::cell_index idx) const 
    { return operator()(idx.i_,idx.j_,idx.k_); } 
  value_type operator[](typename LatVolMesh::face_index) const
    { return (Data)0; }
  value_type operator[](typename LatVolMesh::edge_index) const 
    { return (Data)0; }
  const value_type &operator[](typename LatVolMesh::node_index idx) const
    { return operator()(idx.i_,idx.j_,idx.k_); }

  value_type &operator[](typename LatVolMesh::cell_index idx)
    { return operator()(idx.i_,idx.j_,idx.k_); } 
  value_type operator[](typename LatVolMesh::face_index idx)
    { return (Data)0; }
  value_type operator[](typename LatVolMesh::edge_index idx)
    { return (Data)0; }
  value_type &operator[](typename LatVolMesh::node_index idx)
    { return operator()(idx.i_,idx.j_,idx.k_); }

  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  void resize(const LatVolMesh::node_size_type &size)
    { newsize(size.i_, size.j_, size.k_); }
  void resize(LatVolMesh::edge_size_type) {}
  void resize(LatVolMesh::face_size_type) {}
  void resize(const LatVolMesh::cell_size_type &size)
    { newsize(size.i_, size.j_, size.k_); }
};

template <class Data>
const string
FData3d<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    return "FData3d";
  }
  else
  {
    return find_type_name((Data *)0);
  }
}

template <class Data>
class LatticeVol : public GenericField< LatVolMesh, FData3d<Data> > { 

public:

  LatticeVol() :
    GenericField<LatVolMesh, FData3d<Data> >() {}
  LatticeVol(Field::data_location data_at) :
    GenericField<LatVolMesh, FData3d<Data> >(data_at) {}
  LatticeVol(LatVolMeshHandle mesh, Field::data_location data_at) : 
    GenericField<LatVolMesh, FData3d<Data> >(mesh, data_at) {}
  
  virtual ~LatticeVol(){}

  virtual LatticeVol<Data> *clone() const 
    { return new LatticeVol<Data>(*this); }
 
  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

private:
  static Persistent* maker();
};

#define LATTICEVOL_VERSION 1

template <class Data>
Persistent* 
LatticeVol<Data>::maker()
{
  return scinew LatticeVol<Data>;
}

template <class Data>
PersistentTypeID
LatticeVol<Data>::type_id(type_name(),
		GenericField<LatVolMesh, FData3d<Data> >::type_name(),
                maker); 

template <class Data>
void
LatticeVol<Data>::io(Piostream &stream)
{
  stream.begin_class(type_name().c_str(), LATTICEVOL_VERSION);
  GenericField<LatVolMesh, FData3d<Data> >::io(stream);
  stream.end_class();                                                         
}


template <class Data>
const string
LatticeVol<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "LatticeVol";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 

} // end namespace SCIRun

#endif // Datatypes_LatticeVol_h
















