#include <Packages/Kurt/Core/Geom/SCIRexCompositer.h>
#include <Packages/Kurt/Core/Geom/SCIRexWindow.h>
#include <Packages/Kurt/Core/Geom/SCIRexRenderData.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Thread.h>
#include <iostream>
using std::cerr;
using std::endl;

using namespace Kurt;
using SCIRun::Thread;
using SCIRun::Barrier;

SCIRexCompositer::SCIRexCompositer( SCIRexRenderData *rd) :
  render_data_(rd), die_( false )
{

}

SCIRexCompositer::~SCIRexCompositer()
{

}

void
SCIRexCompositer::run()
{
  do{
    render_data_->mutex_->lock(); 
    cerr<<"check for Exit "<<my_thread_->getThreadName()<<endl;
    render_data_->mutex_->unlock();
    render_data_->barrier_->wait(render_data_->waiters_);
    if( die_ ){  
      render_data_->mutex_->lock(); 
      cerr<<"Returning from thread "<<my_thread_->getThreadName()<<endl;
      render_data_->mutex_->unlock();
      break;
    }
    render_data_->mutex_->lock(); 
    cerr<<"update info for "<<my_thread_->getThreadName()<<endl;
    render_data_->mutex_->unlock();
    render_data_->barrier_->wait( render_data_->waiters_);
    render_data_->mutex_->lock(); 
    cerr<<"render windows "<<my_thread_->getThreadName()<<endl;
    render_data_->mutex_->unlock();
    render_data_->barrier_->wait( render_data_->waiters_);
    render_data_->mutex_->lock(); 
    cerr<<"wait on compositers "<<my_thread_->getThreadName()<<endl;
    render_data_->mutex_->unlock();
    doComposite();
    render_data_->barrier_->wait( render_data_->waiters_);
    render_data_->mutex_->lock(); 
    cerr<<"wait on Display "<<my_thread_->getThreadName()<<endl;
    render_data_->mutex_->unlock();
    render_data_->barrier_->wait( render_data_->waiters_);
    
  }while(true);
}
void 
SCIRexCompositer::add( SCIRexWindow* r)
{
  renderers.push_back( r);
}
void
SCIRexCompositer::doComposite()
{
  vector<SCIRexWindow*>::iterator it;
  int size = renderers.size();
  
  unsigned char *row = new unsigned char[ 3* (xmax - xmin) * (ymax - ymin) ];
  unsigned char *write_buffer_pointer = render_data_->write_buffer_;
  unsigned char *write_buffer_end = render_data_->write_buffer_ + 4 *
    render_data_->viewport_x_ * render_data_->viewport_y_;
  unsigned char *read_buffer_pointer;
  write_buffer_pointer += offset;
  
  int i,j,u,v;
  for(it = renderers.begin(); it != renderers.end(); it++){
  unsigned char *write_buffer_pointer = render_data_->write_buffer_;
  unsigned char *write_buffer_end = render_data_->write_buffer_ + 4 *
    render_data_->viewport_x_ * render_data_->viewport_y_;
  unsigned char *read_buffer_pointer = (*it)->getBuffer();
  write_buffer_pointer += offset;
  read_buffer_pointer += offset;
    // copy pixels one by one for now.
  while( write_buffer_pointer != write_buffer_end){
    *write_buffer_pointer = *read_buffer_pointer;
    write_buffer_pointer++;
    read_buffer_pointer++;
  }
//     (*it)->readFB(buffer_pointer, xmin, ymin, xmax, ymax);
//     for(j = ymin, v = 0; j < ymax; j++, v++){
//       (*it)->readFB(render_data_->write_buffer_, xmin, j, xmax, j);
//       for(i = xmin, u = 0; i < xmax; i++, u++){
	
//       }
//  }
  }
} 


void
SCIRexCompositer::SetFrame(int xmn, int ymn, int xmx, int ymx)
{
  xmin = xmn;
  ymin = ymn;
  xmax = xmx;
  ymax = ymx;
}

void SCIRexCompositer::SetOffset( int o )
{
  offset = o;
}
