/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*----------------------------------------------------------------------
CLASS
    EditPath

    Interactive tool editing and playbacking camera path in Viewer

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
   camera path in Viewer. The basic features of the modules are as follows:
    
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


#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/PathPort.h>
#include <Dataflow/Widgets/CrosshairWidget.h>

#include <Core/Geom/View.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Containers/HashTable.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geom/Path.h>

#include <Core/Util/Timer.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Mutex.h>

#include <Core/Geometry/Quaternion.h>
#include <Core/Geometry/Transform.h>
#include <math.h>

#include <iostream>
using namespace std;


namespace SCIRun {


  
  class EditPath : public Module {
  
    enum ExecMsg { Default=1, add_vp, rem_vp, ins_vp, rpl_vp, init_new, 
		   init_exist, test_path, save_path, get_to_view, 
		   prev_view, next_view, set_step_size, mk_circle_path, w_show, 
		   set_acc_mode, set_path_t};

    enum { to_ogeom=0, to_oview };

    GuiInt     tcl_num_views, tcl_is_looped, tcl_is_backed;
    
    GuiInt    tcl_curr_viewwindow;   
    GuiDouble tcl_step_size, tcl_acc_val, tcl_rate;
    GuiDouble tcl_speed_val;
    GuiInt    UI_Init, tcl_send_dir;
    GuiInt    tcl_msg_box, tcl_intrp_type, tcl_acc_mode, tcl_widg_show, tcl_curr_view, tcl_is_new, tcl_stop;
    GuiString tcl_info;
    
    double       acc_val, speed_val, rate;
    int          curr_view, acc_mode;
    int          curr_viewwindow; 
    bool         is_changed, is_new, is_init;
    ExecMsg      exec_msg;
    View         c_view;
    string     message;
    
    CrosshairWidget* cross_widget;
    CrowdMonitor     widget_lock;
    GeomID           cross_id;
    
    Semaphore    sem;
    
    PathIPort*   ipath;
    PathOPort*   opath;
    PathOPort*   ocam_view;
    GeometryOPort* ogeom;
    PathHandle   ext_path_h, new_path_h, curr_path_h;
    
public:
    EditPath(GuiContext* ctx);
    virtual ~EditPath();
    virtual void execute();
    virtual void tcl_command(GuiArgs&, void*);
    bool init_new_path();
    bool init_exist_path(PathHandle);
    void update_tcl_var();
    void init_tcl_update();
    bool Msg_Box(const string&, const string&);
    void send_view();
};

DECLARE_MAKER(EditPath)
EditPath::EditPath(GuiContext* ctx)
: Module("EditPath", ctx, Filter, "Render", "SCIRun"),
  tcl_num_views(ctx->subVar("tcl_num_views")),
  tcl_is_looped(ctx->subVar("tcl_is_looped")),
  tcl_is_backed(ctx->subVar("tcl_is_backed")),
  tcl_curr_viewwindow(ctx->subVar("tcl_curr_viewwindow")),
  tcl_step_size(ctx->subVar("tcl_step_size")),
  tcl_acc_val(ctx->subVar("tcl_acc_val")),
  tcl_rate(ctx->subVar("tcl_rate")), 
  tcl_speed_val(ctx->subVar("tcl_speed_val")), 
  UI_Init(ctx->subVar("UI_Init")),
  tcl_send_dir(ctx->subVar("tcl_send_dir")),
  tcl_msg_box(ctx->subVar("tcl_msg_box")),
  tcl_intrp_type(ctx->subVar("tcl_intrp_type")),   
  tcl_acc_mode(ctx->subVar("tcl_acc_mode")),
  tcl_widg_show(ctx->subVar("tcl_widg_show")),
  tcl_curr_view(ctx->subVar("tcl_curr_view")),
  tcl_is_new(ctx->subVar("tcl_is_new")), 
  tcl_stop(ctx->subVar("tcl_stop")),
  tcl_info(ctx->subVar("tcl_info")),
  acc_val(0),
  speed_val(0),
  rate(1),
  curr_view(0),
  acc_mode(0),
  curr_viewwindow(0),        // no Viewwindow yet
  is_changed(false),
  exec_msg(Default),
  c_view(Point(1, 1, 1), Point(0, 0, 0), Vector(-1, -1, 1), 20),
  message(""),
  widget_lock("EditPath module widget lock"),
  sem("EditPath Semaphore", 0)
{
  cross_widget =scinew CrosshairWidget(this, &widget_lock, 0.01);
  cross_widget->Connect((GeometryOPort *)get_oport("Geometry"));
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
  ipath = (PathIPort *)get_iport("Path");
  opath = (PathOPort *)get_oport("Path");
  ogeom = (GeometryOPort *)get_oport("Geometry");
  ocam_view = (PathOPort *)get_oport("Camera View");
  
  sem.tryDown();
  { 
    PathHandle p;
    GeometryData* data=0;
    int cv;
    bool is_next=false;

    // first - time initialization
    if (!is_init){
      is_init=true;
      cross_id=ogeom->addObj(cross_widget->GetWidget(), string("Crosshair"), &widget_lock);
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
	
	data=ogeom->getData(0, 0, 1);
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
	    vector<Point> pts(npts);
       
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
	data=ogeom->getData(0, 0, 1);
    
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
	data=ogeom->getData(0, 0, 1);
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
	data=ogeom->getData(0, 0, 1);
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
	data=ogeom->getData(0, 0, 1);
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
  
  switch (tcl_send_dir.get()){
  case to_ogeom:
    ogeom->setView(0, 0, c_view);
    break;
  case to_oview:
    {
    Path* cv=new Path;
    PathHandle cv_h(cv);
    cv_h->add_keyF(c_view);
    ocam_view->send(cv_h);
    }
    break;
  default:
    break;
  }
}


void EditPath::tcl_command(GuiArgs& args, void* userdata)
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
    widget_lock.writeLock();
    cross_widget->SetState(tcl_widg_show.get());
    ogeom->flushViews();
    widget_lock.writeUnlock();
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
  tcl_curr_viewwindow.set(curr_viewwindow);
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
    gui->lock();
    gui->execute(id+" refresh ");
    gui->unlock();
  }
}

void EditPath::update_tcl_var(){
  reset_vars();
  tcl_is_new.set(is_new);
  tcl_speed_val.set(speed_val);
  tcl_acc_val.set(acc_val);
  tcl_curr_view.set(curr_view);
  tcl_num_views.set(curr_path_h->get_num_views());
  tcl_info.set(message);
  message="";
  
  if (UI_Init.get()){
    gui->lock();
    gui->execute(id+" refresh ");
    gui->unlock();
  }
}

bool EditPath::Msg_Box(const string& title, const string& message){
  tcl_msg_box.set(0);
  if (UI_Init.get()){
     gui->lock();
         gui->execute(id+" EraseWarn "+ "\""+title +"\""+ " " + "\""+message+"\"");
     gui->unlock();
  }

  if (tcl_msg_box.get()>0){
    return true;
  }
  else {
    return false;
  }
}

} // End namespace SCIRun

