/*
 *  FieldAlgo.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_FieldAlgo_h
#define Datatypes_FieldAlgo_h

namespace SCIRun {

  
template <class Field, class Functor>
void 
interpolate_vol(const Field &fld, const Point &p, Functor &f) {
  typedef typename Field::mesh_type Mesh;
  
  if (f.wieghts_ == 0) //FIX_ME get weights....
    f.wieghts_ = new double[4]; // four nodes in tets

  typename Mesh::cell_index ci;
  fld.locate(ci, p, f.weights_);

  switch (fld.data_at()) {
  case Field::NODE :
    {
      int i = 0;
      typename Mesh::node_array nodes;
      get_nodes(nodes, ci);
      typename Mesh::node_array::iterator iter = nodes.begin();
      while (iter != nodes.end()) {
	f(fld, *iter, i);
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
#endif //Datatypes_FieldAlgo_h
