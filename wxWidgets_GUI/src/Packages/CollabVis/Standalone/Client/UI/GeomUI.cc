#include <UI/GeomUI.h>
#include <UI/View.h>
#include <UI/Transform.h>
#include <UI/Ball.h>

#include <Rendering/MouseEvent.h>
#include <Rendering/GeometryRenderer.h>

namespace SemotusVisum {

GeomUI::GeomUI( GeometryRenderer *parent) : parent(parent), redraw(false) {
  Log::log( ENTER, "[GeomUI::GeomUI] entered" );
  
  ball = new BallData();
  ball->Init();

  // Set default initial values for all member variables

  
  // Current view
  //view = View(Point3d(0, 0, 10), Point3d(0, 0, 0), Vector(0, 1, 0), 16);   
  //lastview = view;

  // Home view
  //homeview = view;

  // Are we doing inertial rotation?
  inertia_mode = 0;

  angular_v = 0;		// angular velocity for inertia
  // rot_view - automatically initialized to 0 
  // prev_trans - automatically initialized to identity
  eye_dist = 0;
  total_scale = 0;
  prev_time[0] = 0;
  prev_time[1] = 0;
  prev_time[2] = 0;		// history for quaternions and time
  // prev_quat - should be automatically initialized to 0
  redraw = 0;
  last_x = 0;
  last_y = 0;
  total_x = 0;
  total_y = 0;
  total_z = 0;
  rot_point = Point3d(0, 0, 0);
  rot_point_valid = 0;
  last_time = 0;
  xres = 0;
  yres = 0;
  

  Log::log( LEAVE, "[GeomUI::GeomUI] leaving" );
}

GeomUI::~GeomUI() {
  delete ball;
}

void
GeomUI::rotate( int action, int x, int y, int time ) {
  Log::log( ENTER, "[GeomUI::rotate] entered" );
  switch(action){
  case START:
    {
      if(inertia_mode){
	inertia_mode=0;
	redraw = true;
      }
      
      last_x=x;
      last_y=y;
      
      // Find the center of rotation...
      View tmpview(view);
      double aspect=double(xres)/double(yres);
      double znear, zfar;
      rot_point_valid=0;
      parent->compute_depth(tmpview, znear, zfar);

      double zmid=(znear+zfar)/2.;
 
      Point3d ep(0, 0, zmid);
      rot_point=tmpview.eyespace_to_objspace(ep, aspect);

      rot_point = tmpview.lookat();
      rot_view=tmpview;
      rot_point_valid=1;

      double rad = 0.8;
      HVect center(0,0,0,1.0);
	
      // we also want to keep the old transform information
      // around (so stuff correlates correctly)
      // OGL uses left handed coordinate system!
	
      Vector3d z_axis,y_axis,x_axis;

      y_axis = tmpview.up();
      z_axis = tmpview.eyep() - tmpview.lookat();
      eye_dist = z_axis.normalize();
      x_axis = Vector3d::cross(y_axis,z_axis);
      x_axis.normalize();
      y_axis = Vector3d::cross(z_axis,x_axis);
      y_axis.normalize();
      tmpview.up(y_axis); // having this correct could fix something?

      prev_trans.load_frame(Point3d(0.0,0.0,0.0),x_axis,y_axis,z_axis);

      ball->Init();
      ball->Place(center,rad);
      HVect mouse((2.0*x)/xres - 1.0,2.0*(yres-y*1.0)/yres - 1.0,0.0,1.0);
      ball->Mouse(mouse);
      ball->BeginDrag();

      prev_time[0] = time;
      prev_quat[0] = mouse;
      prev_time[1] = prev_time[2] = -100;
      ball->Update();
      last_time=time;
      inertia_mode=0;
      redraw = true;
    }
    break;
  case DRAG:
    {
      if(!rot_point_valid)
	break;

      HVect mouse((2.0*x)/xres - 1.0,2.0*(yres-y*1.0)/yres - 1.0,0.0,1.0);
      prev_time[2] = prev_time[1];
      prev_time[1] = prev_time[0];
      prev_time[0] = time;
      ball->Mouse(mouse);
      ball->Update();

      prev_quat[2] = prev_quat[1];
      prev_quat[1] = prev_quat[0];
      prev_quat[0] = mouse;

      // now we should just sendthe view points through
      // the rotation (after centerd around the ball)
      // eyep lookat and up

      View tmpview(rot_view);

      Transform tmp_trans;
      HMatrix mNow;
      ball->Value(mNow);
      tmp_trans.set(&mNow[0][0]);

      Transform prv = prev_trans;
      prv.post_trans(tmp_trans);

      HMatrix vmat;
      prv.get(&vmat[0][0]);

      Vector3d y_a(vmat[0][1],vmat[1][1],vmat[2][1]);
      Point3d z_a(vmat[0][2],vmat[1][2],vmat[2][2]);

      tmpview.up(y_a);
      tmpview.eyep(Point3d((z_a*(eye_dist)) + tmpview.lookat()));

      view = tmpview;

      cerr << "GeomUI::rotate DRAG" << endl;

      redraw=true;

      last_time=time;
      inertia_mode=0;
    }
    break;
  case END:
    if(time-last_time < 20){
      // now setup the normalized quaternion
      View tmpview(rot_view);
	    
      Transform tmp_trans;
      HMatrix mNow;
      ball->Value(mNow);
      tmp_trans.set(&mNow[0][0]);
	    
      Transform prv = prev_trans;
      prv.post_trans(tmp_trans);
	    
      HMatrix vmat;
      prv.get(&vmat[0][0]);
	    
      Vector3d y_a(vmat[0][1],vmat[1][1],vmat[2][1]);
      Point3d z_a(vmat[0][2],vmat[1][2],vmat[2][2]);
	    
      tmpview.up(y_a);
      tmpview.eyep(Point3d((z_a*(eye_dist)) + tmpview.lookat()));
	    
      view = tmpview;
      cerr << "GeomUI::rotate END" << endl;
      prev_trans = prv;

      // now you need to use the history to 
      // set up the arc you want to use...

      ball->Init();
      double rad = 0.8;
      HVect center(0,0,0,1.0);

      ball->Place(center,rad);

      int index=2;

      if (prev_time[index] == -100)
	index = 1;

      ball->vDown = prev_quat[index];
      ball->vNow  = prev_quat[0];
      ball->dragging = 1;
      ball->Update();
	    
      ball->qNorm = ball->qNow.Conj();
      double mag = ball->qNow.VecMag();

      // Go into inertia mode...
      inertia_mode=1;
      redraw=true;

      if (mag < 0.00001) { // arbitrary ad-hoc threshold
	inertia_mode = 0;
	redraw = true;
      }
      else {
	double c = 1.0/mag;
	double dt = prev_time[0] - prev_time[index];// time between last 2 events
	ball->qNorm.x *= c;
	ball->qNorm.y *= c;
	ball->qNorm.z *= c;
	angular_v = 2*acos(ball->qNow.w)*1000.0/dt;
//	cerr << dt << endl;
      }
    } else {
      inertia_mode=0;
    }
    ball->EndDrag();
    rot_point_valid = 0; // so we don't have to draw this...
    redraw = true;     // always update this...
    break;
  }
  Log::log( LEAVE, "[GeomUI::rotate] leaving" );
}


void 
GeomUI::translate( int action, int x, int y ) {
  Log::log( ENTER, "[GeomUI::translate] entered" );
  switch(action){
  case START:
    {
      if (inertia_mode) {
	inertia_mode=0;
	redraw = true;
      }
      last_x=x;
      last_y=y;
      total_x = 0;
      total_y = 0;
    }
    break;
  case DRAG:
    {
      double xmtn=double(last_x-x)/double(xres);
      double ymtn=-double(last_y-y)/double(yres);
      last_x = x;
      last_y = y;
      // Get rid of roundoff error for the display...
      if (fabs(total_x) < .001) total_x = 0;
      if (fabs(total_y) < .001) total_y = 0;

      View tmpview(view);
      double aspect=double(xres)/double(yres);
      double znear, zfar;
      parent->compute_depth(tmpview, znear, zfar);

      Log::log( DEBUG, "[GeomUI::translate] znear = " + mkString(znear) );
      Log::log( DEBUG, "[GeomUI::translate] zfar = " + mkString(zfar) );
      
      double zmid=(znear+zfar)/2.;
      Vector3d u,v;
      tmpview.get_viewplane(aspect, zmid, u, v);
      double ul=u.length();
      double vl=v.length();

      Log::log( DEBUG, "[GeomUI::translate] xmtn = " + mkString(xmtn) + ", ymtn = " + mkString(ymtn) );
      Vector3d trans(u*xmtn+v*ymtn);

      total_x+=ul*xmtn;
      total_y+=vl*ymtn;

      // Translate the view...
      tmpview.eyep(Point3d(tmpview.eyep()+trans));
      tmpview.lookat(Point3d(tmpview.lookat()+trans));

      Log::log( DEBUG, "[GeomUI::translate] u: (" + mkString(u.x) + ", " + mkString(u.y) + ", " + mkString(u.z) + ")" );
      Log::log( DEBUG, "[GeomUI::translate] v: (" + mkString(v.x) + ", " + mkString(v.y) + ", " + mkString(v.z) + ")" );
      Log::log( DEBUG, "[GeomUI::translate] trans: (" + mkString(trans.x) + ", " + mkString(trans.y) + ", " + mkString(trans.z) + ")" );
      Log::log( DEBUG, "[GeomUI::translate] new eyep: (" + mkString(tmpview.eyep().x) + ", " + mkString(tmpview.eyep().y) + ", " + mkString(tmpview.eyep().z) + ")" );
      Log::log( DEBUG, "[GeomUI::translate] new lookat: (" + mkString(tmpview.lookat().x) + ", " + mkString(tmpview.lookat().y) + ", " + mkString(tmpview.lookat().z) + ")" );

      // Put the view back...
      view = tmpview;
      Log::log( DEBUG, "[GeomUI::translate] DRAG" );
      
      redraw=true;
    }
    break;
  case END:
    break;
  }
  Log::log( LEAVE, "[GeomUI::translate] leaving" );
}

void 
GeomUI::scale( int action, int x, int y ) {
  Log::log( ENTER, "[GeomUI::scale] entered" );
  switch(action){
  case START:
    {
      if (inertia_mode) {
	inertia_mode=0;
	redraw=true;
      }
      last_x=x;
      last_y=y;
      total_scale=1;
    }
    break;
  case DRAG:
    {
      double scl;
      double xmtn=last_x-x;
      double ymtn=last_y-y;
      xmtn/=30;
      ymtn/=30;
      last_x = x;
      last_y = y;
      if (fabs(xmtn)>fabs(ymtn)) scl=xmtn; else scl=ymtn;
      if (scl<0) scl=1/(1-scl); else scl+=1;
      total_scale*=scl;
      
      View tmpview(view);
      tmpview.fov(RtoD(2*atan(scl*tan(DtoR(tmpview.fov()/2.)))));

      view = tmpview;
      cerr << "GeomUI::scale DRAG" << endl;
      redraw=true;
    }
    break;
  case END:
    break;
  }	
  Log::log( LEAVE, "[GeomUI::scale] leaving" );
}
  

}














