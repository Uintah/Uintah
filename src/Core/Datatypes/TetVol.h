
#ifndef Datatypes_TetVol_h
#define Datatypes_TetVol_h

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/MeshTet.h>
#include <Core/Containers/LockingHandle.h>
#include <vector>

namespace SCIRun {

template <class Data>
class TetVol: public Field 
{
public:
  //! Typedefs to support the Field concept.
  typedef Data            value_type;
  typedef MeshTet                  mesh_type;
  typedef vector<Data>    fdata_type;


  TetVol();
  TetVol(data_location data_at);
  virtual ~TetVol();

  //! Required virtual functions from field base.
  virtual MeshBaseHandle get_mesh() const;

  //! Required interfaces from field base.
  virtual InterpolateToScalar* query_interpolate_to_scalar() const;



  //! Required interface to support Field Concept.
  value_type operator[] (int);
  
  template <class Functor>
  void interpolate(const Point &p, Functor &f);
  
  MeshTetHandle get_tet_mesh(); 

  //! attempt to set data, succeeds if sizes match.
  bool set_fdata(fdata_type *fdata) { fdata_ = fdata; }



  //! Persistent I/O.
  void    io(Piostream &stream);
  static  PersistentTypeID type_id;
  static  const string type_name(int);
  virtual const string get_type_name(int n) const;
private:
  //! A Tetrahedral Mesh.
  MeshTetHandle                mesh_;
  //! Data container.
  fdata_type                   fdata_;
  
};

const double TET_VOL_VERSION = 1.0;

template <class Data>
void TetVol<Data>::io(Piostream& stream){

  stream.begin_class(typeName().c_str(), TET_VOL_VERSION);
  Field::io(stream);
  Pio(stream, mesh_.get_rep());
  Pio(stream, fdata_);
  stream.end_class();
}

template <class Data> template <class Functor>
void 
TetVol<Data>::interpolate(const Point &p, Functor &f) {

  if (f.wieghts_ == 0)
    f.wieghts_ = new double[4]; // four nodes in tets
  MeshTet::cell_index ci;
  mesh_->locate_cell(ci, p, f.weights_);

  switch (data_at()) {
  case Field::NODE :
    {
      int i = 0;
      MeshTet::node_array nodes;
      get_nodes_from_cell(nodes, ci);
      MeshTet::node_array::iterator iter = nodes.begin();
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

} // end namespace SCIRun

#endif // Datatypes_TetVol_h
















