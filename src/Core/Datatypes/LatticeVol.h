
#ifndef Datatypes_LatticeVol_h
#define Datatypes_LatticeVol_h

#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array3.h>
#include <vector>

namespace SCIRun {

using namespace vector;

template <class Data>
  class FData3d : public Array3<Data>

  typedef Data value_type;

  FData3d():Array3(){}
  virtual ~FData3d(){}

  value_type operator[](typename LatVolMesh::cell_index idx) const
    { return this(idx.x_,idx.y_,idx.z_); } 
  value_type operator[](typename LatVolMesh::face_index idx) const
    { return this(idx,idx,idx); }
  value_type operator[](typename LatVolMesh::edge_index idx) const
    { return this(idx,idx,idx); }
  value_type operator[](typename LatVolMesh::node_index idx) const
    { return this(idx.x_,idx.y_,idx.z_); }
};

template <class Data>
class LatticeVol: public GenericField< LatVolMesh, FData3d<Data> > { 

public:

  LatticeVal() :
    GenericField<LatVolMesh, FData3d<Data> >() {};
  LatticeVol(data_location data_at) :
    GenericField<LatVolMesh, FData3d<Data> >(data_at) {};
  virtual ~LatticeVol(){};

  static const string type_name(int );
};

template <class Data>
const string
LatticeVol<Data>::type_name(int )
{
  const static string name =  "LatticeVol<" + find_type_name((Data *)0) + ">";
  return name;
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

} // end namespace SCIRun
#endif

#endif // Datatypes_TetVol_h
















