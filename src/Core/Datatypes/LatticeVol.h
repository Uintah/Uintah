
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
  value_type &operator[](typename LatVolMesh::face_index idx)
    { return operator()(idx, 1, 1); }
  value_type &operator[](typename LatVolMesh::edge_index idx)
    { return operator()(idx, 1, 1); }
  value_type &operator[](typename LatVolMesh::node_index idx)
    { return operator()(idx.i_,idx.j_,idx.k_); }

  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  void resize(const LatVolMesh::node_index &size)
  { newsize(size.i_, size.j_, size.k_); }
  void resize(LatVolMesh::edge_index size)
  { newsize(size, 1, 1); }
  void resize(LatVolMesh::face_index size)
  { newsize(size, 1, 1); }
  void resize(const LatVolMesh::cell_index &size)
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
class LatticeVol: public GenericField< LatVolMesh, FData3d<Data> > { 

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

#if 0
template<class Data> int
LatticeVol::get_size(FIELD::data_location data_at) const
{
  int nx = mesh_.get_nx();
  int ny = mesh_.get_ny();
  int nz = mesh_.get_nz();

  switch(data_at) {
  case FIELD::NODE :
    return nx*ny*nz;
  case FIELD::EDGE :
    return 0; // don't know this yet
  case FIELD::FACE :
    return 0; // don't know this yet
  case FIELD::CELL :
    return (nx-1)*(ny-1)*(nz-1);
  default :
    // unknown location
    return 0;
  }
}

template<class Data> bool
LatticeVol<Data>::set_fdata(fdata_type *fdata) {
  if (get_size(location_)==fdata.size()) {
    fdata_ = fdata;
    return true;
  }
  
  return false;
}

template <class Data> value_type
LatticeVol<Data>::operator[](int index) {
  return fdata_[index];
}

const double LATTICE_VOL_VERSION = 1.0;

template <class Data>
void LatticeVol<Data>::io(Piostream& stream){

  stream.begin_class(typeName().c_str(), LATTICE_VOL_VERSION);
  Field::io(stream);
  Pio(stream, mesh_.get_rep());
  Pio(stream, fdata_);
  stream.end_class();
}

template <class Data> template <class Functor>
void 
LatticeVol<Data>::interpolate(const Point &p, Functor &f) {

  if (f.wieghts_ == 0)
    f.wieghts_ = new double[8]; // eight nodes for hex
  MeshRG::cell_index ci;
  mesh_->locate_cell(ci, p, f.weights_);

  switch (data_at()) {
  case Field::NODE :
    {
      int i = 0;
      MeshRG::node_array nodes;
      get_nodes(nodes, ci);
      MeshRG::node_array::iterator iter = nodes.begin();
      while (iter != nodes.end()) {
	f(*data_, *iter);
	++iter; ++i;
      }
    }
  break;
  case Field::EDGE:
    {
    }
    break;
  case Field::FACE:
    {
    }
    break;
  case Field::CELL:
    {
    }
    break;
  } 
} 

/* this should go into the field base class 

template <class Data>
struct InterpFunctor {
  typedef Data data_type;
  typedef typename Data::value_type value_type;

  InterpFunctor(int num_weights = 0) :
    result_(0),
    weights_(0) 
  {
    if (num_weights > 0) {
      weights_ = new double[num_weights];
    }
  }

  virtual ~InterpFunctor() {
    if (weights_) { delete[] weights; }
  }

  double         *weights_;
  value_type      result_;
};

// sample interp functor.
template <class Data, class Index>
struct LinearInterp : public InterpFunctor<Data> {
  
  LinearInterp(int num_weights) :
    InterpFunctor<Data>(num_weights) {}

  void 
  operator()(const Data &data, Index idx, int widx) {
      result_ += data[idx] * weights_[widx];
      cout << "linear interping :)" << endl;
    }
};

*/  // go into base field class
#endif
} // end namespace SCIRun

#endif // Datatypes_TetVol_h
















