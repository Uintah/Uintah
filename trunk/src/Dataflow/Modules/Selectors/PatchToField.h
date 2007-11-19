#ifndef UINTAH_PATCHDATATHREAD_H
#define UINTAH_PATCHDATATHREAD_H


#include <SCIRun/Core/Basis/HexTrilinearLgn.h>
#include <SCIRun/Core/Datatypes/LatVolMesh.h>
#include <SCIRun/Core/Datatypes/GenericField.h>
#include <SCIRun/Core/Thread/Thread.h>
#include <SCIRun/Core/Thread/ThreadGroup.h>
#include <SCIRun/Core/Thread/Semaphore.h>
#include <SCIRun/Core/Thread/Runnable.h>
#include <SCIRun/Core/Util/Timer.h>
#include <Core/Grid/Variables/Variable.h>
#include <Core/Grid/Variables/GridVariable.h>
#include <Core/Grid/Variables/Array3.h>
#include <Core/Grid/Patch.h>
#include <Core/DataArchive/DataArchive.h>


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

template <class Data, class FIELD>
class PatchToFieldThread : public Runnable {
public:
  typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;

  PatchToFieldThread(FIELD *fld,
                     GridVariable<Data>* patchData,
                     IntVector offset,
                     IntVector min_i,
                     IntVector max_i,
                     Semaphore* sema = 0):
    fld_(fld),
    offset_(offset),
    min_(min_i),
    max_(max_i),
    sema_(sema),
    var_(patchData) 
  {
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
      typename Array3<Data>::iterator vit(var_, min_);

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
      typename Array3<Data>::iterator vit(var_, min_);
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
  GridVariable<Data>* var_;
  IntVector offset_;
  IntVector min_;
  IntVector max_;
  Semaphore* sema_;
};


} // end namespace Uintah
#endif
