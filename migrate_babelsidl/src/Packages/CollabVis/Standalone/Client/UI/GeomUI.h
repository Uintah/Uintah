#ifndef __geom_ui_h_
#define __geom_ui_h_

#include <UI/View.h>
#include <UI/Transform.h>
#include <UI/BallAux.h>

namespace SemotusVisum {

class BallData;
class GeometryRenderer;

class GeomUI {
public:
  GeomUI( GeometryRenderer *parent);
  ~GeomUI();

  inline void setRes( const int x, const int y ) {
    xres = x; yres = y;
  }
  void rotate( int action, int x, int y, int time );
  void translate( int action, int x, int y );
  void scale( int action, int x, int y );

  inline void setLastView() { lastview = view; }
  inline View& lastView() { return lastview; }
  inline View& getView() { return view; }
  inline void setHomeView( const View& v ) { homeview = v; }
  inline View getHomeView() { return homeview; }
  inline void goHome() { cerr << "GeomUI->goHome" << endl; view = homeview; }
  
protected:
  /// Geometry renderer
  GeometryRenderer *parent;
  
  /// Arcball
  BallData * ball;

  /// Current view
  View view, lastview;

  /// Home view
  View homeview;

  /// Are we doing inertial rotation?
  int inertia_mode;

  double angular_v;		// angular velocity for inertia
  View rot_view;		// pre-rotation view
  Transform prev_trans;
  double eye_dist;
  double total_scale;
  int prev_time[3];		// history for quaternions and time
  HVect prev_quat[3];
  bool redraw;
  int last_x, last_y;
  double total_x, total_y, total_z;
  Point3d rot_point;
  int rot_point_valid;
  int last_time;
  int xres, yres;
};

}

#endif










