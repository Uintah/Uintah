/*
 *  EditPath.cc:  edit new or existing path datatype; iterates along camera path
 *
 *  Written by:
 *   David Weinstein & Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   July 1994, July 2000
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/PathPort.h>
#include <SCICore/Geom/View.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Containers/HashTable.h>
#include <sys/timeb.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <iostream>
#include <SCICore/Datatypes/Path.h>
#include <SCICore/Thread/Semaphore.h>
#include <SCICore/Thread/Mutex.h>
#include <iosfwd>
#include <SCICore/Util/Timer.h>
#include <SCICore/Thread/Time.h>

using std::cout;
using namespace std;

#undef mr

#define msg(m) cout << m  << endl;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Datatypes;

using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::TclInterface;
using namespace SCICore::Datatypes;
using SCICore::Thread::Semaphore;
using SCICore::Thread::Mutex;
using SCICore::Thread::Time;
using namespace std;

class EditPath : public Module {

    enum ExecMsg { Default=1, add_vp, rem_vp, ins_vp, init, init_new, 
		   init_exist, test_path, stop_test, test_views, save_path, get_to_view, 
		   calc_path, refr_tcl_info, update_path, switch_loop, switch_dir, set_rate, set_step_size};
    
  //  HashTable<clString, ExecMsg>    msg_lookup;
  //    void         RegisterMessages();
  
    TCLvarint    tcl_is_new, tcl_curr_view, tcl_num_views,
                 tcl_intrp_type, tcl_speed_mode, tcl_acc_mode, 
                 tcl_is_looped, tcl_is_backed, tcl_auto_start;

    TCLvarint    tcl_msg_box, tcl_curr_roe, tcl_acc_pat;   
    TCLvardouble tcl_step_size, tcl_speed_val, tcl_acc_val, tcl_rate;
    TCLvarint    UI_Init;

    double       acc_val, speed_val, rate, step_size;
    int          curr_view, speed_mode, acc_mode, acc_patt;
    int          curr_roe; 
    bool         is_changed, is_new, is_init, is_auto;
    ExecMsg      exec_msg;
    View         c_view;
  
    Semaphore    sem;
    Mutex        exec_lock;
  
    PathIPort*   ipath;
    PathOPort*   opath;
    GeometryOPort* ogeom;
    PathHandle   ext_path_h, new_path_h, curr_path_h;
   
public:
    EditPath(const clString& id);
    virtual ~EditPath();
    virtual void execute();
    virtual void tcl_command(TCLArgs&, void*);
    bool init_new_path();
    bool init_exist_path(PathHandle);
    void update_tcl_var(ExecMsg msg=Default);
    bool Msg_Box(const clString&, const clString&);
};

extern "C" Module* make_EditPath(const clString& id)
{
    return scinew EditPath(id);
}

EditPath::EditPath(const clString& id)
: Module("EditPath", id, Filter),
  tcl_is_new("tcl_is_new", id, this), 
  tcl_rate("tcl_rate", id, this), 
  tcl_curr_view("tcl_curr_view", id, this), 
  tcl_num_views("tcl_num_views", id, this), 
  tcl_intrp_type("tcl_intrp_type", id, this), 
  tcl_speed_mode("tcl_speed_mode", id, this), 
  tcl_acc_mode("tcl_acc_mode", id, this), 
  tcl_is_looped("tcl_is_looped", id, this),
  tcl_is_backed("tcl_is_backed", id, this),
  tcl_msg_box("tcl_msg_box", id, this),
  tcl_curr_roe("tcl_curr_roe", id, this),
  tcl_acc_pat("tcl_acc_pat", id, this),
  tcl_step_size("tcl_step_size", id, this), 
  tcl_speed_val("tcl_speed_val", id, this), 
  tcl_acc_val("tcl_acc_val", id, this),
  tcl_auto_start("tcl_auto_start", id, this),
  UI_Init("UI_Init", id, this),
  curr_view(0),
  speed_mode(0),
  acc_mode(0),
  is_changed(false),
  is_auto(false),
  curr_roe(0),        // no Roe yet
  acc_val(0),
  speed_val(0),
  acc_patt(0),
  rate(1),
  step_size(0.01),
  exec_msg(init),
  sem("EditPath Semaphore", 0),
  exec_lock("exec_lock")
{
    // Create the input port
    msg("Entering constructor")
    ipath=scinew PathIPort(this, "Path", PathIPort::Atomic);
    add_iport(ipath);
    opath=scinew PathOPort(this, "Path", PathIPort::Atomic);
    add_oport(opath);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    
    is_new=true;
    new_path_h=curr_path_h=scinew Path();
    new_path_h->set_step(step_size);
    
    //    RegisterMessages();

    UI_Init.set(0);
    update_tcl_var();
    
    is_init=false;

    cout << "Leaving constructor" << endl;
}

EditPath::~EditPath()
{
}

// void EditPath::RegisterMessages(){
//     msg_lookup.insert("Default", Default);
//     msg_lookup.insert("add_vp", add_vp);
//     msg_lookup.insert("rem_vp", rem_vp);
//     msg_lookup.insert("ins_vp", ins_vp);
//     msg_lookup.insert("init", init);
//     msg_lookup.insert("init_new", init_new);
//     msg_lookup.insert("init_exist", init_exist);
//     msg_lookup.insert("test_path", test_path);
//     msg_lookup.insert(" stop_test",  stop_test);
//     msg_lookup.insert("test_views", test_views);
//     msg_lookup.insert("save_path", save_path);
//     msg_lookup.insert("get_to_view", get_to_view);
//     msg_lookup.insert("calc_path", calc_path);
//     msg_lookup.insert("refr_tcl_info", refr_tcl_info);
//     msg_lookup.insert("update_path", update_path);
//     msg_lookup.insert("switch_loop", switch_loop);
//     msg_lookup.insert("switch_dir", switch_dir);
//     msg_lookup.insert("set_rate", set_rate);
//     msg_lookup.insert("set_step_size", set_step_size);
//     //     msg_lookup.insert("", );
//     //     msg_lookup.insert("", );
//     //     msg_lookup.insert("", );

// }


void EditPath::execute()
{
  exec_lock.lock();

  sem.tryDown();
  { 
    PathHandle p;
    GeometryData* data=0;
    
    if (!is_init){
      is_init=true;
      if (!ipath->get(p)){
	init_new_path();
	update_tcl_var();
      }
      else {
	if (init_exist_path(p) && (is_auto=tcl_auto_start.get())) {
	  update_tcl_var();
	  exec_msg=test_path;
	  want_to_execute();
	  // !!! no sem.up() here - no certain UI parts interference
	  exec_lock.unlock();
	  return;
	}
      }
    }
    
    switch (exec_msg) {
    case init_new:
      if (init_new_path()){
	opath->send(curr_path_h);
      }
      update_tcl_var();
      break;
      
    case init_exist:
      if (ipath->get(p)){
	if (init_exist_path(p)) {
	  opath->send(curr_path_h);
	}
      }
      update_tcl_var();
      break;
      
    case add_vp:
      data=ogeom->getData(0, 1);

      if (data && data->view){
	curr_path_h->add_keyF(*(data->view));
	curr_view=(curr_path_h->get_num_views()-1);
	is_changed=true;
	update_tcl_var();
      }
      break;
      
    case rem_vp:    
      data=ogeom->getData(0, 1);
      if (data && data->view){
	if (curr_path_h->get_keyF(curr_view, c_view) && *(data->view)==c_view){
	  curr_path_h->del_keyF(curr_view);
	  if (curr_path_h->get_keyF(--curr_view, c_view))
	    ogeom->setView(0, c_view);
	  is_changed=true;
	  update_tcl_var();
	}
	else{ 
	  // attempt to delete not existing view message !!!
	}
      }
      
      break;
      
    case ins_vp:
      data=ogeom->getData(0, 1);
      if (data && data->view){
	curr_path_h->ins_keyF(++curr_view, *(data->view));
	is_changed=true;
	update_tcl_var(); 
      }
      break;
      
    case test_path:
      // self-messaging mode; sem is down all the time;
      // exit could be done by "on-fly" exec_msg changing in tcl_command()


      bool is_next=curr_path_h->get_nextPP(c_view, curr_view, speed_val, acc_val);
      
  
      ogeom->setView(0, c_view);  
      if (is_next){ 
	  exec_msg=test_path;
	  Time::waitFor((double)rate);
	  want_to_execute();
	  // !!! no sem.up() here - no certain UI parts interference
	  exec_lock.unlock();
	  return;
	}
      else {
	curr_path_h->seek_start();
	sem.up();
      }
      break;

    case test_views:
      // self-messaging mode; sem is down all the time;
      // exit could be done by "on-fly" exec_msg changing in tcl_command()
      if(curr_path_h->get_nextKF(c_view, curr_view, speed_val, acc_patt)){
          ogeom->setView(0, c_view);  
	  exec_msg=test_views;
	  Time::waitFor((double)(rate*3)); 
	  want_to_execute();
	  // !!! no sem.up() here - no certain UI parts interference
	  exec_lock.unlock();
	  return;
      }
      else {
	curr_path_h->seek_start(); 
	sem.up();
      }
      break;

    case get_to_view:
      int cv=tcl_curr_view.get();
      if (curr_path_h->get_keyF(cv, c_view)){
	ogeom->setView(0, c_view);
	curr_view=cv;
      }
      break;

    case calc_path:
      break;
    case refr_tcl_info:
      break;
    case update_path:
      break;
    case switch_loop:
      break;
    case stop_test:
      break;
    case save_path:
      opath->send(curr_path_h);
      break;
    default:
      break;
    }
  }
  exec_msg=Default;
  sem.up();
  exec_lock.unlock();
}

void EditPath::tcl_command(TCLArgs& args, void* userdata)
{   
 //  ExecMsg em=(msg_lookup.lookup(args[1], em))?em:Default;
//   cout << "tcl_command function call: "  << args[1] << endl;
  
//   // messages to be executed one after another
//   switch(em){
//   case add_vp:
//   case rem_vp:
//   case ins_vp:
//   case init:
//   case init_new:
//   case init_exist:
//   case test_path:
//   case test_views:
//   case save_path:
//   case get_to_view:
//     if (exec_lock.tryLock() && sem.tryDown()){      // previos msg handled and execute() isn't running
//       exec_msg=em;
//       want_to_execute();
//     }
//     return;
//   }

//   // messages to set "run-time" parameters
//   switch(em){
//   case stop_test:
//     // substitute active message assuming that execute() is in loop
//     exec_lock.lock();
//     if (exec_msg==test_path || exec_msg==test_views){  
//       exec_msg=stop_test;
//     }
//     exec_lock.unlock();
//   case switch_loop:
//     exec_lock.lock();
//       curr_path_h->set_loop(tcl_is_looped.get());
//     exec_lock.unlock();
//   case set_rate:
//     exec_lock.lock();
//       rate=tcl_rate.get();
//     exec_lock.unlock();
//     return;
//   case switch_dir:
//     exec_lock.lock();
//      curr_path_h->set_back(tcl_is_backed.get());
//     exec_lock.unlock();
//     return;
//   case set_step_size:
//     exec_lock.lock();
//      curr_path_h->set_step(step_size=tcl_step_size.get());
//      is_changed=true;
//     exec_lock.unlock();
//     return;
//   default:
//     Module::tcl_command(args, userdata);
//   }
// }


    if (args[1] == "add_vp"){
       if(sem.tryDown()){
 	exec_msg=add_vp;
 	want_to_execute();
       }
     }  
     else if (args[1] == "rem_vp"){
       if(sem.tryDown()){
 	exec_msg=rem_vp;
 	want_to_execute();
       }
     }
     else if (args[1] == "ins_vp"){
       if(sem.tryDown()){
 	exec_msg=ins_vp;
 	want_to_execute();
       }
     }
     else if (args[1] == "init"){
       if(sem.tryDown()){
 	exec_msg=init;
 	want_to_execute();
       }
     }
     else if (args[1] == "init_new"){ 
       if(sem.tryDown()){
        exec_msg=init_new; 
        want_to_execute();
       }
     }
     else if (args[1] == "init_exist"){
       if(sem.tryDown()){
        exec_msg=init_exist;
        want_to_execute();
       }
     }
     else if (args[1] == "test_path"){
       msg ("In test_path TCL handler");
       if(sem.tryDown()){
 	msg ("Sending exec message");
 	exec_msg=test_path;
 	want_to_execute();
       }
     }
     else if (args[1] == "stop_test"){
 	// substitute active message assuming that execute() is in loop
 	exec_lock.lock();
 	  exec_msg=stop_test;
 	exec_lock.unlock();
     }
     else if (args[1] == "test_views"){
 	if(sem.tryDown()){
           exec_msg=test_views;
 	  want_to_execute();
 	}
     }
     else if (args[1] == "save_path"){
 	if(sem.tryDown()){
           exec_msg=save_path;
 	  want_to_execute();
 	}
     } 
     else if (args[1] == "get_to_view"){
       if(sem.tryDown()){
 	exec_msg=get_to_view;
 	curr_view=tcl_curr_view.get();
 	want_to_execute();
       }
     }
     else if (args[1] == "calc_path"){
       exec_msg=calc_path;
     }
     else if (args[1] == "refr_tcl_info"){
       exec_msg=refr_tcl_info;
     }
     else if (args[1] == "update_path"){
       exec_msg=update_path;
     }
     else if (args[1] == "switch_loop"){
       exec_lock.lock();
 	curr_path_h->set_loop(tcl_is_looped.get());
       exec_lock.unlock();
     }
     else if (args[1] == "switch_dir"){
       exec_lock.lock();
 	curr_path_h->set_back(tcl_is_backed.get());
       exec_lock.unlock();
     }
     else if (args[1] == "set_rate"){
       exec_lock.lock();
 	rate=tcl_rate.get();
       exec_lock.unlock();
     }
     else if (args[1] == "set_step_size"){
       exec_lock.lock();
         curr_path_h->set_step(step_size=tcl_step_size.get());
 	is_changed=true;
       exec_lock.unlock();
     }
     else if (args[1] == "" || args[1] == "Default"){
       //exec_msg=Default;
     }
     else{
       Module::tcl_command(args, userdata);
     }
}

bool EditPath::init_new_path(){
  if (is_changed && !Msg_Box("Modified Buffer Exists", "There is modified buffer. Do you want to discard it?"))
      return false;
  is_new=true;
  is_changed=false;
  new_path_h->reset();
  curr_path_h=new_path_h;
  return true;
}

bool EditPath::init_exist_path(PathHandle p){
  if (p.get_rep()){
    if (is_changed && !Msg_Box("Modified Buffer Exists", "There is modified buffer. Do you want to discard it?"))
      return false;
    
    is_new=false;
    is_changed=false;
    curr_path_h=p;
    return true;
  }
  else {
    // empty path handle!!!
    return false;
  }
}

void EditPath::update_tcl_var(ExecMsg msg){
  msg ("Updating TCL variables");
 
  switch(msg){
  
//   case ins_vp:
//   case rem_vp:
//   case add_vp:
//     tcl_curr_view.set(curr_view+1);
    
//     break;
 
  case init:
  default:  
    tcl_is_new.set(is_new);
    tcl_rate.set(rate);
    tcl_curr_view.set(curr_view+1);  // adjustion for zero-indexed arrays
    tcl_num_views.set(curr_path_h->get_num_views());
    tcl_intrp_type.set(curr_path_h->get_path_type());
    tcl_speed_mode.set(speed_mode);
    tcl_acc_mode.set(acc_mode);
    tcl_is_looped.set(curr_path_h->is_looped());
    tcl_is_backed.set(curr_path_h->is_backed());
    tcl_msg_box.set(0);
    tcl_step_size.set(curr_path_h->get_step());
    tcl_speed_val.set(speed_val);
    tcl_acc_val.set(acc_val);
    tcl_acc_pat.set(curr_path_h->get_acc_patt(curr_view));
    tcl_curr_roe.set(curr_roe);
    tcl_auto_start.set(is_auto);
  }

  if (UI_Init.get()){
    TCLTask::lock();
    TCL::execute(id+" refresh ");
    TCLTask::unlock();
  }
}

bool EditPath::Msg_Box(const clString& title, const clString& message){
  tcl_msg_box.set(0);
  if (UI_Init.get()){
     TCLTask::lock();
         TCL::execute(id+" EraseWarn "+ "\""+title +"\""+ " " + "\""+message+"\"");
     TCLTask::unlock();
  }

  if (tcl_msg_box.get()!=0){
//     MSG ("MSGBOX returning true");
    return true;
  }
  else {
//     MSG ("MSGBOX returning false");
    return false;
  }
}

} // End namespace Modules
} // End namespace PSECommon


//
// $Log$
// Revision 1.3  2000/07/19 19:28:53  samsonov
// Moving from DaveW
//
// Revision 1.2  2000/07/19 19:27:16  samsonov
// Moving from DaveW package
//
// Revision 1.1  2000/07/18 23:09:49  samsonov
// *** empty log message ***
//
// Revision 1.4  2000/07/14 23:38:44  samsonov
// Rewriting for creating and editing camera paths; initial submit for testing Yarden's module
//
// Revision 1.3  2000/03/17 09:25:54  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.2  1999/12/03 00:29:23  dmw
// improved the module for John Day...
//
// Revision 1.1  1999/12/02 21:57:33  dmw
// new camera path datatypes and modules
//









