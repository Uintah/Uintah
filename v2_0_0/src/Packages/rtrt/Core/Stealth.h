

#ifndef STEALTH_H
#define STEALTH_H 1

#include <iostream>
#include <vector>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Point;

using std::cout;
using std::vector;
using std::string;

// "Stealth" comes from the DoD term for an invisible watcher on the
// simulated battlefield.  The stealths are used in relation to
// cameras.  They contain movement information, but the camera
// contains direction information.
  
class Stealth {

  ////////////////////////////////////////////////////////
  // Stealth Movement
  //
  //   This camera does NOT move like an airplane.
  // The driver (user) can specify movement in the following
  // manner.
  //
  //     +  Accelerate Forward
  //     -  Accelerate Backward
  //     <- Begin turning left 
  //     -> Begin turning right
  //     ^  Begin pitching forward (look down)
  //     v  Begin pitching backward (look up)
  //     7 (keypad) Accelerate to the left
  //     9 (keypad) Accelerate to the right
  //     0 (keypad) Stop ALL movement
  //     . (keypad) Slow down (in all movements (including turns))
  //
  //   Acceleration is in respect to the direction that the eye is
  // looking.  

public:

  // rotate_scale of '4' is good 
  // translate_scale of '100' is good if the eye is 3 units (or so) from
  // the area of interest and the area of interest is ~6 units across.
  Stealth( double translate_scale, double rotate_scale, double gravity_force );
  ~Stealth();

  // Tells the eye (camera) to update its position based on its current
  // velocity vector.
  void updatePosition();

  inline double getSpeed( int direction ) const {
    switch( direction ) {
    case 0:
      return speed_;
    case 1:
      return horizontal_speed_;
    case 2:
      return vertical_speed_;
    case 3:
      return pitch_speed_;
    case 4:
      return rotate_speed_;
    default:
      cout << "Error in Stealth::getSpeed, bad direction " << direction 
	   << "\n";
      exit( 1 );
      return 0.0;
    }
  }

  // Slows down in all dimensions (pitch, speed, turn, etc);
  void slowDown();
  void stopAllMovement();
  void stopPitchAndRotate();
  void stopPitch();
  void stopRotate();

  void slideLeft();
  void slideRight();

  void goUp();
  void goDown();

  void accelerate();
  void decelerate();
  void turnRight();
  void turnLeft();
  void pitchUp();
  void pitchDown();

  void updateRotateSensitivity( double scalar );
  void updateTranslateSensitivity( double scalar );

  // Display the Stealth's speeds, etc.
  void print();

  // Returns next location in the path and the new view vector.
  void getNextLocation( Point & point, Point & look_at );

  // Moves the stealth to the next Marker in the path, point/lookat
  // are set to the correct locations.  Index of that pnt is returned.
  // Returns -1 if no route!

  //void using_catmull_rom(vector<Point> &points, vector<Point> &f_points);
  Point using_catmull_rom(vector<Point> &points, int i, float t);

  int  getNextMarker( Point & point, Point & look_at );
  int  getPrevMarker( Point & point, Point & look_at );
  int  goToBeginning( Point & point, Point & look_at );
  void getRouteStatus( int & pos, int & numPts );

  // Clear out path stealth is to follow.
  void clearPath();
  // Adds point to the end of the route
  void addToPath( const Point & eye, const Point & look_at );
  // Adds point in front of the current marker.
  void addToMiddleOfPath( const Point & eye, const Point & look_at );
  void deleteCurrentMarker();

  // Returns the name of the route (for GUI to display)
  string loadPath( const string & filename );
  void   newPath(  const string & routeName );
  void   savePath( const string & filename );

  // Choose the current path to follow.
  void   selectPath( int index );

  // If gravity is on, the stealth/camera will "fall" towards the ground.
  void toggleGravity();

  // Stealths and Cameras are highly integrated right now... perhaps
  // stealth should be a sub class in camera?  This needs more thought.

  bool   gravityIsOn() { return gravity_on_; }
  double getGravityForce() { return gravity_force_; }

  bool   moving(); // Returns true if moving in any direction

private:

  void increase_a_speed( double & speed, int & accel_cnt, double scale, double base, double max );
  void decrease_a_speed( double & speed, int & accel_cnt, double scale, double base, double min );

  void displayCurrentRoute();

  // Scale is based on the size of the "universe".  It effects how fast
  // the stealth will move.

  double baseTranslateScale_;
  double baseRotateScale_;

  double translate_scale_;
  double rotate_scale_;

  // Speeds (in units per frame)
  double speed_;            // + forward, - backward
  double horizontal_speed_; // + right, - left
  double vertical_speed_;   // + up, - down
  double pitch_speed_;      // + down, - up
  double rotate_speed_;     // + right, - left

  // Acceleration counts represent the number of velocity change 
  // requests that the user has made.  They are used to judge how
  // much velocity to add to the velocity vector.  The more requests,
  // the faster we "accelerate" for the next similar request.
  int accel_cnt_;            // + forward, - backward. 
  int horizontal_accel_cnt_; // + right, - left
  int vertical_accel_cnt_;   // + up, - down
  int pitch_accel_cnt_;      // + up, - down
  int rotate_accel_cnt_;     // + right, - left

  // Path information  (rough draft)

  vector< vector<Point> >   paths_;    // Array of routes.
  vector< vector<Point> >   lookAts_;
  vector< string >          routeNames_;

  vector<Point>    * currentPath_;

  vector<Point>    * currentLookAts_;

  string           * currentRouteName_;
  double        segment_percentage_;

  bool          gravity_on_;
  double        gravity_force_;
};

} // end namespace rtrt

#endif
