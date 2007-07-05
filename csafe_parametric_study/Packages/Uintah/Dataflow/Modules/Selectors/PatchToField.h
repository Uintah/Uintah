#ifndef UINTAH_PATCHDATATHREAD_H
#define UINTAH_PATCHDATATHREAD_H


#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Runnable.h>
#include <Core/Util/Timer.h>
#include <Packages/Uintah/Core/Grid/Variables/Variable.h>
#include <Packages/Uintah/Core/Grid/Variables/Array3.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>


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
using SCIRun::LatVolMesh;

template <class Var, class Data, class FIELD>
class PatchToFieldThread : public Runnable {
public:
  typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;

  PatchToFieldThread(FIELD *fld,
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
    LVMesh *mesh = fld_->get_typed_mesh().get_rep();
      
    IntVector lo(min_ - offset_);
    IntVector hi(max_ - offset_);

    if( fld_->basis_order() == 0){
      // Get an iterator over a subgrid of the mesh
      LVMesh::Cell::range_iter it(mesh,
                                  lo.x(), lo.y(), lo.z(),
                                  hi.x(), hi.y(), hi.z());
      // The end iterator is just a cell iterator
      LVMesh::Cell::iterator it_end; it.end(it_end);
      typename Array3<Data>::iterator vit(&var_, min_);

      for(;it != it_end; ++it){
        fld_->fdata()[*it] = *vit;
        ++vit;
      }
    } else {
      // Get an iterator over a subgrid of the mesh
      LVMesh::Node::range_iter it(mesh,
                                  lo.x(), lo.y(), lo.z(),
                                  hi.x(), hi.y(), hi.z());
      // The end iterator is just a node iterator
      LVMesh::Node::iterator it_end; it.end(it_end);
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

  FIELD *fld_;
  Var var_;
  IntVector offset_;
  IntVector min_;
  IntVector max_;
  Semaphore* sema_;
};


} // end namespace Uintah
#endif
