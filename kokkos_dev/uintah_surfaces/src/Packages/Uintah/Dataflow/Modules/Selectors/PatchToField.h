#ifndef UINTAH_PATCHDATATHREAD_H
#define UINTAH_PATCHDATATHREAD_H

#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Runnable.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Packages/Uintah/Core/Grid/Variables/Variable.h>
#include <Packages/Uintah/Core/Grid/Variables/Array3.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Core/Util/Timer.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::string;
using SCIRun::Thread;
using SCIRun::ThreadGroup;
using SCIRun::Semaphore;
using SCIRun::Runnable;
using SCIRun::LatVolField;
using SCIRun::LatVolMesh;

template <class Var, class Data>
class PatchToFieldThread : public Runnable {
public:
  PatchToFieldThread(LatVolField<Data> *fld,
                     Var& patchData,
                     IntVector offset,
                     IntVector min_i,
                     IntVector max_i,
                     Semaphore* sema = 0):
    fld_(fld),
    offset_(offset),
    min_(min_i),
    max_(max_i),
    sema_(sema)
  {
    var_.copyPointer(patchData);
  }
  void run()
    {
      LatVolMesh *mesh = fld_->get_typed_mesh().get_rep();
      
      if( fld_->basis_order() == 0){
        IntVector lo(min_ - offset_);
        IntVector hi(max_ - offset_);
#if 1
        // Get an iterator over a subgrid of the mesh
        LatVolMesh::Cell::range_iter it(mesh,
					lo.x(), lo.y(), lo.z(),
                                        hi.x(), hi.y(), hi.z());
        // The end iterator is just a cell iterator
        LatVolMesh::Cell::iterator it_end; it.end(it_end);

        typename Array3<Data>::iterator vit(&var_, min_);

	for(;it != it_end; ++it){
	  fld_->fdata()[*it] = *vit;
	  ++vit;
	}
#else
        typename Array3<Data>::iterator vit(&var_, min_);

	for (int k = lo.k_; k <= hi.k_; k++)
	  for (int j = lo.j_; j <= hi.j_; j++)
	    for (int i = lo.i_; i <= hi.i_; i++)
	      {
		LatVolMesh::Cell::index_type  idx(m, i, j, k); 
		
		fld_->fdata()[idx] = *vit;
		++vit;
	      }
#endif
      } else {

        IntVector lo(min_ - offset_);
        IntVector hi(max_ - offset_);
        // Get an iterator over a subgrid of the mesh
        LatVolMesh::Node::range_iter it(mesh,
					lo.x(), lo.y(), lo.z(),
                                        hi.x(), hi.y(), hi.z());
        // The end iterator is just a node iterator
        LatVolMesh::Node::iterator it_end; it.end(it_end);
        typename Array3<Data>::iterator vit(&var_, min_);
	for(;it != it_end; ++it){
	  fld_->fdata()[*it] = *vit;
	  ++vit;
	}
      }
      if (sema_) sema_->up();
    }
private:
  PatchToFieldThread(){}

  LatVolField<Data> *fld_;
  Var var_;
  IntVector offset_;
  IntVector min_;
  IntVector max_;
  Semaphore* sema_;
};


} // end namespace Uintah
#endif
