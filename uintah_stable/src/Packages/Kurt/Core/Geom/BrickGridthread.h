/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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


#ifndef BRICK_GRID_THREAD_H
#define BRICK_GRID_THREAD_H

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Container/Array3.h>
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>

namespace Kurt {

using SCIRun::Runnable;
using SCIRun::Semaphore;
using SCIRun::Array3;
using Uintah::ShareAssignArray3.h;
  

template <class Data>
class BrickGridThread : public Runnable {
public:
  BrickGridThread(ShareAssignArray3<Data>& data,
		  const Array3<GridBrick*> *bricks,
		  const IntVector& offset,
		  Semaphore *sema) :
    data_(data), bricks_(bricks), offset_(offset), sema_(sema){}

  virtual void run()
    {
      ShareAssignArray3<Data>::iterator it(data_.begin());
      ShareAssignArray3<Data>::iterator it_end(data_.end());
      IntVector min(data_.getLowIndex() - offset);
      IntVector max(data_.getHighIndex() - offset);
      
      vector<GridBrick *> intersects;
      get_intersecting_bricks( intersects, min, max );
      vector<GridBrick *>::iterator gbit;
      vector<GridBrick *>::iterator gbit_end = intersects.end();
      for(; it != it_end; ++it){
	gbit = intersects.begin();
	for(; gbit != gbit_end; ++gbit ){
	  SCIRun::Array3<unsigned char>* bdata = (*gbit)->texture();
	  IntVector bmin;
	  IntVector bmax;
	  (*gbit)->get_index_range( bmin, bmax );
	  IntVector index = it.getIndex() - offset;
	  if( bmin.x() <= index.x() && index.x() <= bmax.x() &&
	      bmin.y() <= index.y() && index.y() <= bmax.y() &&
	      bmin.z() <= index.z() && index.z() <= bmax.z() ){
	    (*bdata)(index.z() - bmin.z(),
		     index.y() - bmin.y(),
		     index.x() - bmin.x()) = SETVAL( *it );
	  }
	}
      }
      sema_->up();
    }
      
private:
  ShareAssignArray3<Data>& data_;
  const Array3<GridBrick*> *bricks_;
  IntVector offset_;
  Semaphore* sema_;

  get_intersecting_bricks(vector<GridBrick *>& list,
					 const IntVector& min, 
					 const IntVector& max )
    {  
      GridBrick *brick;
      int i,j,k;
      for(i = 0; i < nx(); i++ )
	for(j = 0; j < ny(); j++ )
	  for(k = 0; k < nz(); k++) {
	    IntVector bmin;
	    IntVector bmax;
	    brick = (*bricks_)(i, j, k);
	    brick->get_index_range( bmin, bmax );
	    if( bmin.x() < max.x() && bmax.x() > min.x() &&
		bmin.y() < max.y() && bmax.y() > min.y() &&
		bmin.z() < max.z() && bmax.z() > min.z() ){
	      list.push_back(brick);
	    }
	  }
    }

  
};
#endif
