/*
 *  Graph.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   July 20001
 *
 */



#include <tcl.h>
#include <tk.h>

#include <Core/2d/Graph.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/GuiInterface/TclObj.h>
#include <Core/GuiInterface/GuiInterface.h>

namespace SCIRun {

class GraphHelper : public Runnable {
  Graph *graph_;
  Mutex lock_;

public:
  GraphHelper( Graph *g ) : graph_(g), lock_("GraphHelper") {}
  virtual ~GraphHelper() {}

  virtual void run();
};


void
GraphHelper::run()
{
  for (;;) {
    graph_->has_work_.wait( lock_ );
    graph_->update();
  }
}



Graph::Graph(GuiInterface* gui, const string &id )
  : DrawGui(gui, id, "Graph" ), has_work_("GraphLock")
{
  obj_ = 0;

  set_id( id );

  helper_ = scinew GraphHelper(this);
  Thread* t=new Thread(helper_, id.c_str() );
  t->detach();
  
}

void
Graph::set_window( const string &window )
{
  TclObj::set_window( window );
  if ( obj_ ) {
    obj_->set_windows( window+".menu", 
		       window+".f.tb",
		       window+".ui",
		       window+".f.gl" );
  }
}

void
Graph::add( const string &, DrawGui *d )
{
  if ( obj_ ) delete obj_;

  obj_ = d;

  obj_->set_parent( this );
  obj_->set_id( id() + "-obj" );

  if ( window() != "" ) {
    obj_->set_windows( window()+".menu", 
		       window()+".f.tb",
		       window()+".ui",
		       window()+".f.gl");
  }
}


void
Graph::need_redraw() 
{
  has_work_.conditionSignal();
}

void
Graph::update()
{
  if ( !obj_ ) 
    return; 

  if ( has_window() )
    obj_->redraw();
}

void
Graph::tcl_command(GuiArgs& args, void*)
{
  if ( args[1] == "redraw" ) 
    update();
}


#define GRAPH_VERSION 1

void
Graph::io(Piostream& stream)
{
  stream.begin_class( "Graph", GRAPH_VERSION);


  stream.end_class();
}

void Graph::lock()
{
  gui->lock();
}

void Graph::unlock()
{
  gui->unlock();
}
} // End namespace SCIRun

