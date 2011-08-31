/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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
#include <Core/Grid/Variables/Variable.h>
#include <Core/Grid/Variables/GridVariable.h>
#include <Core/Grid/Variables/Array3.h>
#include <Core/Grid/Patch.h>
#include <Core/DataArchive/DataArchive.h>


#include <string>
#include <iostream>

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
    var_(patchData),
    offset_(offset),
    min_(min_i),
    max_(max_i),
    sema_(sema)
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
