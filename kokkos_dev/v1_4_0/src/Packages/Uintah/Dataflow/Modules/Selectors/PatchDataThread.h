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

namespace SCIRun {
  void swapbytes( Uintah::Matrix3& m);
} //end namespace SCIRun

#include <string>
#include <iostream>
using std::string;
using std::cerr;
using std::endl;

namespace Uintah {

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
      IntVector levelLo, LevelHi, lo, hi, sz;
      Var v; 
      archive_.query( v, name_, mat_, patch_, time_);
      
      LatVolMesh *m = fld_->get_typed_mesh().get_rep();
      

      if( fld_->data_at() == Field::CELL){
	lo = patch_->getCellLowIndex() - offset_;
	hi = patch_->getCellHighIndex() - offset_;
	sz = hi - lo;
	LatVolMesh mesh(m, lo.x(), lo.y(), lo.z(), 
			sz.x()+1, sz.y()+1, sz.z()+1);
	LatVolMesh::CellIter it; mesh.begin(it);
	LatVolMesh::CellIter it_end; mesh.end(it_end);
	const Array3<Data> &vals = v;
	Array3<Data>::const_iterator vit = vals.begin();
	if(swapbytes_){
	  Data val;
	  for(;it != it_end; ++it){
	    IntVector idx = vit.getIndex() - offset_;
	    val  = *vit;
	    swapbytes( val );
	    fld_->fdata()[*it] = val;
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
	lo = patch_->getNodeLowIndex() - offset_;
	hi = patch_->getNodeHighIndex() - offset_;
	sz = hi - lo;
	LatVolMesh mesh(m, lo.x(), lo.y(), lo.z(),
			sz.x(), sz.y(), sz.z());
	LatVolMesh::Node::iterator it; mesh.begin(it);
	LatVolMesh::Node::iterator it_end; mesh.begin(it_end);
	const Array3<Data> &vals = v;
	Array3<Data>::const_iterator vit = vals.begin();
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
      }
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
