/*----------------------------------------------------------------------
CLASS
    EditPath

    Interactive tool editing and playbacking camera path in Salmon module

GENERAL INFORMATION

    Created by:
    Alexei Samsonov
    Department of Computer Science
    University of Utah
    July 2000
    
    Copyright (C) 2000 SCI Group

KEYWORDS
    Quaternion

DESCRIPTION
   EditPath module provides a set of interactive tools for creation and manipulation of
   camera path in Salmon module. The basic features of the modules are as follows:
    
   - Interactive adding/inserting/deleting of key frames of current camera position in edited path
   - Navigation through existing key frames</para>
   - Three interpolation modes - Linear, Cubic and no interpolation (key frames only playback)
   - Two acceleration modes - smooth start/end(in interpolated mode) and no acceleration
   - Path step(smoothness) and sampling rate specification for current path
   - Automatic generation of circle path based on current camera position and position of reference widget
   - Looped and reversed path modes

PATTERNS

WARNING    
    
POSSIBLE REVISIONS
    Adding additional interpolation modes (subject to experiment)
    Adding support of PathWidget
----------------------------------------------------------------------*/


#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/PathPort.h>
#include <PSECore/Datatypes/CameraViewPort.h>
#include <PSECore/Widgets/CrosshairWidget.h>

#include <SCICore/Geom/View.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Containers/HashTable.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/Datatypes/Path.h>

#include <SCICore/Util/Timer.h>
#include <SCICore/Thread/Time.h>
#include <SCICore/Thread/CrowdMonitor.h>
#include <SCICore/Thread/Semaphore.h>
#include <SCICore/Thread/Mutex.h>

#include <SCICore/Geometry/Quaternion.h>
#include <SCICore/Geometry/Transform.h>
#include <math.h>

#include <iostream>
using std::cout;
using namespace std;


namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::TclInterface;
using namespace SCICore::Datatypes;

using PSECore::Widgets::CrosshairWidget;
using SCICore::Thread::Semaphore;
using SCICore::Thread::Mutex;
using SCICore::Thread::Time;
  
  class EditPath : public Module {
  
    enum ExecMsg { Default=1, add_vp, rem_vp, ins_vp, rpl_vp, init_new, 
		   init_exist, test_path, save_path, get_to_view, 
		   prev_view, next_view, set_step_size, mk_circle_path, w_show, 
		   set_acc_mode, set_path_t};

    enum { to_ogeom=0, to_oview };

    TCLint     tcl_num_views, tcl_is_looped, tcl_is_backed;
    
    TCLint    tcl_curr_roe;   
    TCLdouble tcl_step_size, tcl_acc_val, tcl_rate;
    TCLvardouble tcl_speed_val;
    TCLint    UI_Init, tcl_send_dir;
    TCLvarint tcl_msg_box, tcl_intrp_type, tcl_acc_mode, tcl_widg_show, tcl_curr_view, tcl_is_new, tcl_stop;
    TCLstring tcl_info;
    
    double       acc_val, speed_val, rate;
    int          curr_view, acc_mode;
    int          curr_roe; 
    bool         is_changed, is_new, is_init;
    ExecMsg      exec_msg;
    View         c_view;
    clString     message;
    
    CrosshairWidget* cross_widget;
    CrowdMonitor     widget_lock;
    GeomID           cross_id;
    
    Semaphore    sem;
    
    PathIPort*   ipath;
    PathOPort*   opath;
    CameraViewOPort* ocam_view;
    GeometryOPort* ogeom;
    PathHandle   ext_path_h, new_path_h, curr_path_h;
    CameraViewHandle cv_h;
    CameraView       camv;
    
public:
    EditPath(const clString& id);
    virtual ~EditPath();
    virtual void execute();
    virtual void tcl_command(TCLArgs&, void*);
    bool init_new_path();
    bool init_exist_path(PathHandle);
    void update_tcl_var();
    void init_tcl_update();
    bool Msg_Box(const clString&, const clString&);
    void send_view();
};

extern "C" Module* make_EditPath(const clString& id)
{
    return scinew EditPath(id);
}

EditPath::EditPath(const clString& id)
: Module("EditPath", id, Filter),
  tcl_stop("tcl_stop", id, this),
  tcl_msg_box("tcl_msg_box", id, this),
  tcl_widg_show("tcl_widg_show", id, this),
  tcl_is_new("tcl_is_new", id, this), 
  tcl_rate("tcl_rate", id, this), 
  tcl_curr_view("tcl_curr_view", id, this),
  tcl_acc_mode("tcl_acc_mode", id, this),
  tcl_intrp_type("tcl_intrp_type", id, this),   
  tcl_is_looped("tcl_is_looped", id, this),
  tcl_is_backed("tcl_is_backed", id, this),
  tcl_step_size("tcl_step_size", id, this),
  tcl_curr_roe("tcl_curr_roe", id, this),
  tcl_num_views("tcl_num_views", id, this),
  tcl_speed_val("tcl_speed_val", id, this), 
  tcl_acc_val("tcl_acc_val", id, this),
  tcl_info("tcl_info", id, this),
  UI_Init("UI_Init", id, this),
  tcl_send_dir("tcl_send_dir", id, this),
  curr_view(0),
  acc_mode(0),
  is_changed(false),
  curr_roe(0),        // no Roe yet
  acc_val(0),
  speed_val(0),
  rate(1),
  exec_msg(Default),
  widget_lock("EditPath module widget lock"),
  sem("EditPath Semaphore", 0),
  message(""),
  camv(),
  c_view(Point(1, 1, 1), Point(0, 0, 0), Vector(-1, -1, 1), 20)
{
  ipath=scinew PathIPort(this, "Path", PathIPort::Atomic);
  add_iport(ipath);
  opath=scinew PathOPort(this, "Path", PathIPort::Atomic);
  add_oport(opath);
  ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(ogeom);
  ocam_view=scinew CameraViewOPort(this, "Camera View",  CameraViewIPort::Atomic);
  add_oport(ocam_view);

  cv_h=&camv;

  cross_widget =scinew CrosshairWidget(this, &widget_lock, 0.01);
  cross_widget->SetState(0);
  cross_widget->SetPosition(Point(0, 0, 0));
  cross_id=0;
  
  new_path_h=curr_path_h=scinew Path();
  is_new=true;
  
  UI_Init.set(0);
 
  init_tcl_update();
    
  is_init=false;
  need_execute=1;
  
  sem.up();
}

EditPath::~EditPath()
{
}

void EditPath::execute()
{

  sem.tryDown();
  { 
    PathHandle p;
    GeometryData* data=0;
    int cv;
    bool is_next=false;

    // first - time initialization
    if (!is_init){
      is_init=true;
      cross_id=ogeom->addObj(cross_widget->GetWidget(), clString("Crosshair"), &widget_lock);
      if (ipath->get(p))
	init_exist_path(p);
      else 
	init_new_path();
    }
    else {
      // execute request from upstream module
      if (ipath->get(p))
	if (ext_path_h.get_rep()==0 || p->generation!=ext_path_h->generation){ // new path appeared on the input
	  if (init_exist_path(p)) {
	    opath->send(curr_path_h);
	    exec_msg=Default;
	    sem.up();
	    return;
	  }
	}
    }
 
    switch (exec_msg) {

      // ******************************************************
      case init_new:
	if (init_new_path()){
	  opath->send(curr_path_h);
	}
	break;
	
      case init_exist:
	if (ipath->get(p)){
	  if (init_exist_path(p)) {
	    opath->send(curr_path_h);
	  }
	}
	else {
	  message="Cann't get path from outside";
	  update_tcl_var();
	}
	break;
      
      // ******************************************************	
      case mk_circle_path:{
	
	data=ogeom->getData(0, 1);
	if (data && data->view){
	  c_view=*(data->view);
	  if (!cross_widget->GetState() && !Msg_Box("No visible widget message","No active widget is visible for specifying path center. Would you like to use its current position?")){
	    break;
	  }

	  const Point wpos=cross_widget->GetPosition();
	  Point eye=c_view.eyep();
	  Point lookp=c_view.lookat();
	  Vector lookdir=lookp-eye;
	  Vector rdir=wpos-eye;
	  const double proj=Dot(lookdir.normal(), rdir.normal())*rdir.length();
	  const double radius=(proj>0)?proj:-proj;
	  if (rdir.length()<10e-7|| radius<10e-5){
	    message="Bad Geometry: No circle";
	    update_tcl_var();
	    break;
	  }
 
	  if (init_new_path()){
	    const int npts=40;
	    const double PI=3.14159265358979323846;
	    
	    Vector tang=Cross(lookdir, c_view.up());
	    Vector u=Cross(tang, lookdir);
	    double f=c_view.fov();
	    Point center=eye+lookdir.normal()*proj;	   
	    Array1<Point> pts;
	    pts.resize(npts);
       
	    u.normalize();
	    double angle=0;
	    for (int i=0; i<npts-1; i++){ // first and last frames don't coincide ???
	      angle=(i*PI)/(npts-1);
	      Quaternion rot(cos(angle), u*sin(angle));
	      Vector dir=rot.rotate(eye-center);
	      pts[i]=center+dir;
	      curr_path_h->add_keyF(View(pts[i], center, u, f));
	    }
	    
	    is_changed=true;
	    curr_path_h->set_path_t(KEYFRAMED);
	    curr_path_h->set_acc_t(SMOOTH);
	    init_tcl_update();
	    opath->send(curr_path_h);
	  }
	}
	break;
      }

      // ******************************************************	
      case add_vp:
	data=ogeom->getData(0, 1);
    
	if (data && data->view){
	  c_view=*(data->view);
	  speed_val=tcl_speed_val.get();
	  if (curr_path_h->add_keyF(c_view, speed_val)){
	    curr_view=(curr_path_h->get_num_views()-1);
	    is_changed=true;
	    send_view();
	    message="Key frame added";
	  }
	  else {
	    message="Cann't add keyframe at the same position";
	  }
	}
	else {
	  message="Cann't get view";
	}
	update_tcl_var();
	break;
      
      // ******************************************************		
      case rem_vp:    
	data=ogeom->getData(0, 1);
	if (data && data->view){
	  if (curr_path_h->get_keyF(curr_view, c_view, speed_val)){
	      //      && *(data->view)==c_view){
	    curr_path_h->del_keyF(curr_view);

	    if (curr_view==curr_path_h->get_num_views())  // last view deleted
	      curr_view--;
	    
	    if (curr_path_h->get_keyF(curr_view, c_view, speed_val)){
	      send_view();
	    }
	      
	    is_changed=true;
	    message="Key frame removed";
	  }
	  else{ 
	    // attempt to delete not existing view message !!!
	    message="Cann't delete non-active view";
	  }
	}
	else {
	  message="Cann't get view";
	}
	update_tcl_var();
	break;

      // ******************************************************		
      case ins_vp:
	data=ogeom->getData(0, 1);
	if (data && data->view){
	  speed_val=tcl_speed_val.get();
	  if (curr_path_h->ins_keyF(curr_view, *(data->view), speed_val)){
	    send_view();
	    is_changed=true;
	    message="Key frame inserted";
	  }
	  else {
	    message="No insertion";
	  }
	}
	else {
	  message="Cann't get view";
	}
	update_tcl_var();
	break;

      // ******************************************************		
      case rpl_vp:
	data=ogeom->getData(0, 1);
	if (data && data->view){
	  if (curr_path_h->get_keyF(curr_view, c_view, speed_val)){ 
	    curr_path_h->del_keyF(curr_view);

	    if (curr_view==curr_path_h->get_num_views()) // last view deleted
	      if (curr_path_h->add_keyF(*(data->view), tcl_speed_val.get())){
		message="Last keyframe replaced";
	      }
	      else {
		message="Cann't replace (neighboor keyframe at the same position)";
		curr_path_h->add_keyF(c_view, tcl_speed_val.get());
	      }
	    else {
	      if (curr_path_h->ins_keyF(curr_view, *(data->view), tcl_speed_val.get())){
		message="Keyframe replaced";
	      }
	      else {
		message="Cann't replace (neighboor keyframe at the same position)";
		curr_path_h->ins_keyF(curr_view, c_view, tcl_speed_val.get());
	      }
	    }
	    curr_path_h->get_keyF(curr_view, c_view, speed_val);
	    send_view();
	    is_changed=true;
	  }
	  else{ 
	    // attempt to delete not existing view message !!!
	    message="Cann't delete view";
	  }
	}      
	else {
	  message="Cann't get view";
	}
	
	update_tcl_var();
	break;

      // ******************************************************		
      case test_path: {
	// self-messaging mode; sem is down all the time;

	if (curr_path_h->set_step(tcl_step_size.get())
	    || curr_path_h->set_loop(tcl_is_looped.get())
	    || curr_path_h->set_back(tcl_is_backed.get())
	    || curr_path_h->set_path_t(tcl_intrp_type.get())
	    || curr_path_h->set_acc_t(tcl_acc_mode.get())) {
	  is_changed=true;
	}
	
	if (!curr_path_h->is_started()){
	  tcl_stop.set(0);
	}
					 
	if (!tcl_stop.get()){
	  double olds=speed_val;
	  is_next=curr_path_h->get_nextPP(c_view, cv, speed_val, acc_val);  
	 
	  if (is_next){
	    send_view();
	    acc_val=(speed_val-olds)/rate;
	    
	    curr_view=cv;
	    update_tcl_var();
	    exec_msg=test_path;
	    Time::waitFor(rate=tcl_rate.get());
	    want_to_execute();
	    // !!! no sem.up() here - no certain UI parts interference
	    return;
	  }
	  else {
	    acc_val=0;
	    curr_path_h->seek_start();
	    curr_view=cv;
	    update_tcl_var();
	  }
	}
     
	curr_path_h->stop();
      }
      break;
    
      //********************************************************			
      case get_to_view:
	cv=tcl_curr_view.get();
	if (curr_path_h->get_keyF(cv, c_view, speed_val)){
	  send_view();
	  curr_view=cv;
	  update_tcl_var();
	}
	break;

      // ******************************************************
      case next_view:
	cv=curr_view+1;
	if (curr_path_h->get_keyF(cv, c_view, speed_val)){
	  curr_view=cv;
	  send_view();
	}
	else {
	  if (curr_path_h->get_keyF(curr_view, c_view, speed_val)){
	    send_view();
	  }
	}
	update_tcl_var();
	break;

      // ******************************************************
      case prev_view:
	cv=curr_view-1;
	if (curr_path_h->get_keyF(cv, c_view, speed_val)){
	  curr_view=cv;
	  send_view();
	}
	else {
	  if (curr_path_h->get_keyF(curr_view, c_view, speed_val)){
	    send_view();
	  }
	}
	update_tcl_var();
	break;

      // ******************************************************
      case set_path_t:
      case set_acc_mode:  	
	if (curr_path_h->set_path_t(tcl_intrp_type.get())
	    || curr_path_h->set_acc_t(tcl_acc_mode.get()))
	  opath->send(curr_path_h);
	break;
	
      // ******************************************************
      case save_path:
	if (curr_path_h->build_path()){
	  curr_path_h->set_path_t(tcl_intrp_type.get());
	  curr_path_h->set_acc_t(tcl_acc_mode.get());
	  curr_path_h->set_step(tcl_step_size.get());
	  curr_path_h->set_loop(tcl_is_looped.get());
	  curr_path_h->set_back(tcl_is_backed.get());
	  opath->send(curr_path_h);
	  message="Path Saved";
	  update_tcl_var();
	}
	break;

      // ******************************************************
      default:
	break;
    }
  }
  exec_msg=Default;
  sem.up();
}

void EditPath::send_view(){
  
  cv_h->set_view(c_view);

  switch (tcl_send_dir.get()){
  case to_ogeom:
    ogeom->setView(0, c_view);
    break;
  case to_oview:
    ocam_view->send(cv_h);
    break;
  default:
    break;
  }
}


void EditPath::tcl_command(TCLArgs& args, void* userdata)
{   
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
  else if (args[1] == "rpl_vp"){
    if(sem.tryDown()){
      exec_msg=rpl_vp;
      want_to_execute();
    }
  }
  else if (args[1] == "init_new"){ 
    if(sem.tryDown()){
      exec_msg=init_new; 
      want_to_execute();
    }
    else {
      update_tcl_var();
    }
  }
  else if (args[1] == "init_exist"){
    if(sem.tryDown()){
      exec_msg=init_exist;
      want_to_execute();
    }
    else {
      update_tcl_var();
    }
  }
  else if (args[1] == "test_path"){
    if(sem.tryDown()){
      tcl_stop.set(0);
      exec_msg=test_path;
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
 	want_to_execute();
    }
    else {
      update_tcl_var();
    }
  }
  else if (args[1] == "next_view"){
    if(sem.tryDown()){
      exec_msg=next_view;
      want_to_execute();
    }
    else {
      update_tcl_var();
    }
  }
  else if (args[1] == "prev_view"){
    if(sem.tryDown()){
      exec_msg=prev_view;
      want_to_execute();
    }
     else {
      update_tcl_var();
    }
  }
  else if (args[1] == "w_show"){
    cout << "entering widget lock" << endl;
    widget_lock.writeLock();
    cout << " behind widget lock " << endl;
    cross_widget->SetState(tcl_widg_show.get());
    ogeom->flushViews();
    widget_lock.writeUnlock();
    cout << " out of w_show handler" << endl;
  }
  else if (args[1] == "mk_circle_path"){
    if(sem.tryDown()){
      exec_msg=mk_circle_path;
      want_to_execute();
    }
  }
  else{
    Module::tcl_command(args, userdata);
  }
}

bool EditPath::init_new_path(){
 
  if (is_changed && !Msg_Box("Modified Buffer Exists", "There is modified buffer. Do you want to discard it?")){
    update_tcl_var();
    return false;
  }
  is_new=true;
  is_changed=false;
  curr_view=0;
  
  new_path_h->reset();
  curr_path_h=new_path_h;

  curr_path_h->set_acc_t(SMOOTH);
  curr_path_h->set_path_t(CUBIC);
  message="Editing new path";
  init_tcl_update();

  return true;
}

bool EditPath::init_exist_path(PathHandle p){

  if (is_changed && !Msg_Box("Modified Buffer Exists", "There is modified buffer. Do you want to discard it?")){
    update_tcl_var();
    return false;
  }

  is_new=false;
  is_changed=false;
  curr_path_h=ext_path_h=p;
  message="Editing existing path";
  init_tcl_update();
  return true;  
}

// setting tcl vars to initial state
void EditPath::init_tcl_update(){
  tcl_num_views.set(curr_path_h->get_num_views());
  tcl_intrp_type.set(curr_path_h->get_path_t());
  tcl_acc_mode.set(curr_path_h->get_acc_t());
  tcl_is_looped.set(curr_path_h->is_looped());
  tcl_is_backed.set(curr_path_h->is_backed());
  tcl_step_size.set(curr_path_h->get_step());
  tcl_curr_roe.set(curr_roe);
  tcl_rate.set(1);
  tcl_info.set(message);
  tcl_send_dir.set(0);

  widget_lock.readLock();
  tcl_widg_show.set(cross_widget->GetState());
  widget_lock.readUnlock();

  tcl_is_new.set(is_new);
  tcl_speed_val.set(speed_val);
  tcl_acc_val.set(acc_val);
  tcl_msg_box.set(0);
  tcl_curr_view.set(curr_view);
  
  if (UI_Init.get()){
    TCLTask::lock();
    TCL::execute(id+" refresh ");
    TCLTask::unlock();
  }
}

void EditPath::update_tcl_var(){
  TCL::reset_vars();  
  tcl_is_new.set(is_new);
  tcl_speed_val.set(speed_val);
  tcl_acc_val.set(acc_val);
  tcl_curr_view.set(curr_view);
  tcl_num_views.set(curr_path_h->get_num_views());
  tcl_info.set(message);
  message="";
  
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

  if (tcl_msg_box.get()>0){
    return true;
  }
  else {
    return false;
  }
}

} // End namespace Modules
} // End namespace PSECommon


//
// $Log$
// Revision 1.6.4.1  2000/10/19 05:17:12  sparker
// Merge changes from main branch into csafe_risky1
//
// Revision 1.7  2000/09/29 08:45:35  samsonov
// Added camera speed support
//
// Revision 1.6  2000/09/15 21:49:26  samsonov
// added output switch and send_view() function
//
// Revision 1.5  2000/08/20 04:24:20  samsonov
// added CameraView outport
//
// Revision 1.4  2000/08/09 08:01:45  samsonov
// final version
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









