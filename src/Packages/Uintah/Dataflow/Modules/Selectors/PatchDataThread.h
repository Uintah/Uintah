#ifndef UINTAH_PATCHDATATHREAD_H
#define UINTAH_PATCHDATATHREAD_H

#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Runnable.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Util/Endian.h>
#include <Packages/Uintah/Core/Grid/Variable.h>
#include <Packages/Uintah/Core/Grid/Array3.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/CCA/Ports/DataArchive.h>
#include <Core/Util/Timer.h>

namespace SCIRun {
  void swapbytes( Uintah::Matrix3& m);
} //end namespace SCIRun

#include <string>
#include <iostream>
namespace Uintah {
using std::string;
using std::cerr;
using std::endl;
using SCIRun::Thread;
using SCIRun::ThreadGroup;
using SCIRun::Semaphore;
using SCIRun::Runnable;
using SCIRun::LatVolField;
using SCIRun::LatVolMesh;


template <class Var, class Iter>
class PatchDataThread : public Runnable {
public:  
  PatchDataThread(DataArchive& archive, 
		  Iter iter,
		  const string& varname,
		  int matnum,
		  const Patch* patch,
		  double time, Semaphore* sema,
		  bool swapbytes = false) :
    archive_(archive),
    iter_(iter),
    name_(varname),
    mat_(matnum),
    patch_(patch),
    time_(time),
    sema_(sema),
    swapbytes_(swapbytes){}

  void run() 
    {
      Var v; 
      archive_.query( v, name_, mat_, patch_, time_); 
      *iter_ = v; 
      if( swapbytes_){
	typename Var::iterator it(v.begin()), it_end(v.end()); 
	for(; it !=  it_end; ++it)
	  swapbytes( *it );
      }
      sema_->up();
    }
  
private:

  PatchDataThread(){}

  DataArchive& archive_;
  Iter iter_;
  const string& name_;
  int mat_;
  const Patch *patch_;
  double time_;
  Semaphore *sema_;
  bool swapbytes_;
};
  

template <class Var, class Data>
class PatchDataToLatVolFieldThread : public Runnable {
public:  
  PatchDataToLatVolFieldThread(DataArchive& archive, 
			      LatVolField<Data> *fld,
			      IntVector& offset,
			      const string& varname,
			      int matnum,
			      const Patch* patch,
			      double time, Semaphore* sema,
			      bool swapbytes = false) :
    archive_(archive),
    fld_(fld),
    offset_(offset),
    name_(varname),
    mat_(matnum),
    patch_(patch),
    time_(time),
    sema_(sema),
    swapbytes_(swapbytes){}

  void run() 
    {
      Var v; 
      WallClockTimer TIMER;
      TIMER.start();
      cerr<<"Start time = "<<TIMER.time()<<endl;
      archive_.query( v, name_, mat_, patch_, time_);
      cerr<<"Done Reading Data\n";
      cerr<<"Time = "<<TIMER.time()<<endl;
      LatVolMesh *m = fld_->get_typed_mesh().get_rep();
      

      if( fld_->data_at() == Field::CELL){

	IntVector lo(patch_->getCellLowIndex() - offset_);
	IntVector hi(patch_->getCellHighIndex() - offset_);
	// Get an iterator over a subgrid of the mesh
	LatVolMesh::Cell::range_iter it(m, lo.x(), lo.y(), lo.z(),
					hi.x(), hi.y(), hi.z());
	// The end iterator is just a cell iterator
	LatVolMesh::Cell::iterator it_end;
	// See Core/Datatypes/LatVolMesh.cc
	if( lo.z() != hi.z() )
	  it_end = LatVolMesh::Cell::iterator(m, lo.x(), lo.y(), hi.z());
	else
	  it_end = LatVolMesh::Cell::iterator(m, lo.x(), lo.y(), hi.z()+1);

	const Array3<Data> &vals = v;
	Array3<Data>::const_iterator vit = vals.begin();
	cerr<<"Done with setup\n";
	cerr<<"Time = "<<TIMER.time()<<endl;
	if(swapbytes_){
	  for(;it != it_end; ++it){
	    IntVector idx = vit.getIndex() - offset_;
	    fld_->fdata()[*it] = *vit;
	    swapbytes( fld_->fdata()[*it]);
	    ++vit;
	  }
	} else {
	  for(;it != it_end; ++it){
	    IntVector idx = vit.getIndex() - offset_;
	    fld_->fdata()[*it] = *vit;
	    ++vit;
	  }
	}
      } else {

	IntVector lo(patch_->getNodeLowIndex() - offset_);
	IntVector hi(patch_->getNodeHighIndex() - offset_);
	// Get an iterator over a subgrid of the mesh
	LatVolMesh::Node::range_iter it(m, lo.x(), lo.y(), lo.z(),
					hi.x(), hi.y(), hi.z());
	// The end iterator is just a node iterator
	LatVolMesh::Node::iterator it_end;
	// See Core/Datatypes/LatVolMesh.cc
	if( lo.z() != hi.z() )
	  it_end = LatVolMesh::Node::iterator(m, lo.x(), lo.y(), hi.z());
	else
	  it_end = LatVolMesh::Node::iterator(m, lo.x(), lo.y(), hi.z()+1);

	const Array3<Data> &vals = v;
	Array3<Data>::const_iterator vit = vals.begin();
	if(swapbytes_){
	  for(;it != it_end; ++it){
	    //	    IntVector idx = vit.getIndex() - offset_;
	    fld_->fdata()[*it] = *vit;
	    swapbytes( fld_->fdata()[*it]);
	    ++vit;
	  }
	} else {
	  for(;it != it_end; ++it){
	    //	    IntVector idx = vit.getIndex() - offset_;
	    fld_->fdata()[*it] = *vit;
	    ++vit;
	  }
	}
      }
      cerr<<"Finished:  Time = "<<TIMER.time()<<endl;
      TIMER.stop();
      sema_->up();
    }
  
private:
  
  PatchDataToLatVolFieldThread(){}
  DataArchive& archive_;
  LatVolField<Data> *fld_;
  IntVector offset_;
  const string& name_;
  int mat_;
  const Patch *patch_;
  double time_;
  Semaphore *sema_;
  bool swapbytes_;
};
  



} // end namespace Uintah
#endif
