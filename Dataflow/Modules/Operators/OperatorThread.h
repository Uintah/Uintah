#if ! defined(UINTAH_OPERATOR_THREAD_H)
#define UINTAH_OPERATOR_THREAD_H

#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Mutex.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  using std::cerr;
  using std::endl;
  using namespace SCIRun;

template<class Data, class ScalarField, class Op>
class OperatorThread : public Runnable
{
public:
  OperatorThread( Array3<Data>&  data, ScalarField *scalarField,
		  IntVector offset, Op op,
		  Semaphore *sema ) :
    data_(data), sf_(scalarField), offset_(offset),
    op_(op), sema_(sema) {}

  void run()
{
  typename ScalarField::mesh_type *m =
    sf_->get_typed_mesh().get_rep();
  typename Array3<Data>::iterator it(data_.begin());
  typename Array3<Data>::iterator it_end(data_.end());
  IntVector min(data_.getLowIndex() - offset_);
  IntVector size(data_.getHighIndex() - min - offset_);
  if( sf_->data_at() == Field::CELL ) {
    typename ScalarField::mesh_type mesh(m, min.x(), min.y(), min.z(),
					 size.x()+1, size.y()+1, size.z()+1);
    typename ScalarField::mesh_type::Cell::iterator s_it; mesh.begin(s_it);
    typename ScalarField::mesh_type::Cell::iterator s_it_end; mesh.end(s_it_end);
    for(; s_it != s_it_end; ++it, ++s_it){
      IntVector idx((*s_it).i_, (*s_it).j_, (*s_it).k_);
      idx += offset_;
      //sf_->fdata()[*s_it] = op_( data_[idx] );
      sf_->fdata()[*s_it] = op_( *it );
    }
  } else {
    typename ScalarField::mesh_type mesh(m, min.x(), min.y(), min.z(),
					 size.x(), size.y(), size.z());
    typename ScalarField::mesh_type::Node::iterator s_it; mesh.begin(s_it);
    typename ScalarField::mesh_type::Node::iterator s_it_end; mesh.end(s_it_end);
    for(; s_it != s_it_end; ++it, ++s_it){
      IntVector idx((*s_it).i_, (*s_it).j_, (*s_it).k_);
      idx += offset_;
      //      sf_->fdata()[*s_it] = op_( data_[idx] );
      sf_->fdata()[*s_it] = op_( *it );
    }
  }
  sema_->up();
}
private:
  Array3<Data>& data_;
  ScalarField *sf_;
  IntVector offset_;
  Op op_;
  Semaphore *sema_;
  //  Mutex* lock_;
};


template<class Data, class ScalarField >
class AverageThread : public Runnable
{
public:
  AverageThread( Array3<Data>&  data, ScalarField *scalarField,
		  IntVector offset, double& aveVal,
		  Semaphore *sema, Mutex *m) :
      data_(data), sf_(scalarField), offset_(offset),
      aveVal_(aveVal), sema_(sema), lock_(m) {}
  
  void run()
{
  typename ScalarField::mesh_type *m =
    sf_->get_typed_mesh().get_rep();
  typename Array3<Data>::iterator it(data_.begin());
  typename Array3<Data>::iterator it_end(data_.end());
  IntVector min(data_.getLowIndex() - offset_);
  IntVector size(data_.getHighIndex() - min - offset_);
  double ave = 0;
  int counter = 0;
  if( sf_->data_at() == Field::CELL ) {
    typename ScalarField::mesh_type mesh(m, min.x(), min.y(), min.z(),
					 size.x()+1, size.y()+1, size.z()+1);
    typename ScalarField::mesh_type::Cell::iterator s_it; mesh.begin(s_it);
    typename ScalarField::mesh_type::Cell::iterator s_it_end; mesh.end(s_it_end);
    for(; s_it != s_it_end; ++it, ++s_it){
      sf_->fdata()[*s_it] = (sf_->fdata()[*s_it]+(*it))/2.0 ;
      ave += sf_->fdata()[*s_it];
      ++counter;
    }
  } else {
    typename ScalarField::mesh_type mesh(m, min.x(), min.y(), min.z(),
					 size.x(), size.y(), size.z());
    typename ScalarField::mesh_type::Node::iterator s_it; mesh.begin(s_it);
    typename ScalarField::mesh_type::Node::iterator s_it_end; mesh.end(s_it_end);
    for(; s_it != s_it_end; ++it, ++s_it){
      sf_->fdata()[*s_it] = (sf_->fdata()[*s_it]+(*it))/2.0 ;
      ave += sf_->fdata()[*s_it];
      ++counter;
    }
  }
  lock_->lock();
  cerr<<"sum  = "<<ave<<", counter = "<<counter
      <<", ave = "<<ave/(double)counter<<endl;

  aveVal_ += ( ave/(double)counter );
  lock_->unlock();
  sema_->up();
}
private:
  Array3<Data>& data_;
  ScalarField *sf_;
  IntVector offset_;
  double& aveVal_;
  Semaphore *sema_;
  Mutex* lock_;
};

} // namespace Uintah
#endif //UINTAH_OPERATOR_THREAD_H
