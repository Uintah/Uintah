#include <Packages/Kurt/Core/Geom/SCIRexCompositer.h>
#include <Packages/Kurt/Core/Geom/SCIRexWindow.h>
#include <Packages/Kurt/Core/Geom/SCIRexRenderData.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Thread.h>
#include <Core/Math/MinMax.h>
#include <iostream>
#include <ios>
using std::cerr;
using std::endl;
using std::hex;
using std::oct;
using std::ios;
using namespace Kurt;
using SCIRun::Thread;
using SCIRun::Barrier;
using SCIRun::Min;

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
  for(;;){
    int i = 0;
//     render_data_->mutex_->lock(); 
//     cerr<<"check for Exit "<<my_thread_->getThreadName()<<"   "<<i++<<endl;
//     render_data_->mutex_->unlock();
    for(;;){
      render_data_->barrier_->wait(render_data_->waiters_);
      if( die_ ){  
	render_data_->mutex_->lock(); 
	cerr<<"Returning from thread "<<my_thread_->getThreadName()<<endl;
	render_data_->mutex_->unlock();
	return;
      } else if(!render_data_->waiters_changed_){
	break;
      }
    }
//     render_data_->mutex_->lock(); 
//     cerr<<"update info for "<<my_thread_->getThreadName()<<endl;
//     render_data_->mutex_->unlock();
    render_data_->barrier_->wait( render_data_->waiters_);
//     render_data_->mutex_->lock(); 
//     cerr<<"render windows "<<my_thread_->getThreadName()<<endl;
//     render_data_->mutex_->unlock();
    render_data_->barrier_->wait( render_data_->waiters_);
//     render_data_->mutex_->lock(); 
//     cerr<<"wait on compositers "<<my_thread_->getThreadName()<<endl;
//     render_data_->mutex_->unlock();
    doComposite();
    render_data_->barrier_->wait( render_data_->waiters_);
//     render_data_->mutex_->lock(); 
//     cerr<<"wait on Display "<<my_thread_->getThreadName()<<endl;
//     render_data_->mutex_->unlock();
    render_data_->barrier_->wait( render_data_->waiters_);
    
  }
}
void 
SCIRexCompositer::add( SCIRexWindow* r)
{
  renderers.push_back( r);
}
void
SCIRexCompositer::doComposite()
{
  //  vector<SCIRexWindow*>::iterator it;
  int size = renderers.size();

//   if(strcmp(my_thread_->getThreadName(), "Compositer-0") == 0)
//     return;
  
//  unsigned char *row = new unsigned char[ 3* (xmax - xmin) * (ymax - ymin) ];
  
  int i,j,u,v;
  //  for(it = renderers.begin(); it != renderers.end(); it++){
  for( i = 0; i < renderers.size(); i++){
//     cerr<<"compositing window-"<<render_data_->comp_order_[i]<<endl;
    SCIRexWindow *win = renderers[render_data_->comp_order_[i]];
//      cerr<<"i = "<<i<<", comp_order_[i] = "<<render_data_->comp_order_[i]<<endl;
    unsigned char *write_buffer_pointer = render_data_->write_buffer_;
    unsigned char *write_buffer_end = write_buffer_pointer + end_offset;
    unsigned char *read_buffer_pointer = (win)->getBuffer();
//     cerr<<"readptr = "<<(int)read_buffer_pointer<<", ";
    write_buffer_pointer += begin_offset;
    read_buffer_pointer += begin_offset;
//     cerr<<"readptr + offset = "<<(int)read_buffer_pointer<<endl;
    // copy pixels one by one for now.
    while( write_buffer_pointer != write_buffer_end){
      unsigned char* wb = write_buffer_pointer;
      unsigned char* alpha_pointer = read_buffer_pointer + 3;
      double alpha = *alpha_pointer/255.0;
//       cerr<<"alpha = "<<alpha<<" \t";
      unsigned char r,g,b,a;

//       r = Min((*read_buffer_pointer * alpha) +
// 	      ((1.0 - alpha) * *wb), 255.0);
      *write_buffer_pointer = Min((*read_buffer_pointer /* * alpha */) +
				  ((1.0 - alpha) * *wb), 255.0);
      write_buffer_pointer++; wb++; read_buffer_pointer++;

//       g = Min((*read_buffer_pointer /* * alpha */) +
// 	      ((1.0 - alpha) * *wb), 255.0);
      *write_buffer_pointer = Min((*read_buffer_pointer /* * alpha */) +
				  ((1.0 - alpha) * *wb), 255.0);
      write_buffer_pointer++; wb++; read_buffer_pointer++;

//       b = Min((*read_buffer_pointer /* * alpha */) +
// 	      ((1.0 - alpha) * *wb), 255.0);
      *write_buffer_pointer = Min((*read_buffer_pointer /* * alpha */) +
				  ((1.0 - alpha) * *wb), 255.0);
      write_buffer_pointer++; wb++; read_buffer_pointer++;
      
//       a = Min(*alpha_pointer + ((1.0 - alpha) * *wb),
// 				  255.0);
      *write_buffer_pointer = Min(*read_buffer_pointer + ((1.0 - alpha) * *wb),
				  255.0);
//       if( r !=0 && g!=0 && b!=0 && a != 255)
// 	cerr<<"rgba = ("<<r*1<<","<<g*1<<","<<b*1<<","<<a*1<<") \t";
      write_buffer_pointer++; wb++; read_buffer_pointer++;
    }
  } 
//   cerr<<endl;
}

void
SCIRexCompositer::SetFrame(int xmn, int ymn, int xmx, int ymx)
{
  xmin = xmn;
  ymin = ymn;
  xmax = xmx;
  ymax = ymx;
  begin_offset = 4*ymin*xmax;
  end_offset = 4*ymax*xmax;
}

