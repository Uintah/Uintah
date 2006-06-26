
#include <Packages/rtrt/Core/Stealth.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Persistent/Pstreams.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <stdio.h>
#include <math.h>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

static double translate_base = 1.5;      
static double translate_max_count = 8;   
static double rotate_base = 1.5;     
static double rotate_max_count = 8;  

Stealth::Stealth( double translate_scale, double rotate_scale, 
		  double gravity_force ) :
  baseTranslateScale_( translate_scale ), baseRotateScale_( rotate_scale ),
  translate_scale_( translate_scale ), rotate_scale_(rotate_scale), 
  speed_( 0 ), horizontal_speed_( 0 ), vertical_speed_( 0 ),
  pitch_speed_( 0 ), rotate_speed_( 0 ),
  accel_cnt_( 0 ), horizontal_accel_cnt_( 0 ),
  vertical_accel_cnt_( 0 ), pitch_accel_cnt_( 0 ), rotate_accel_cnt_( 0 ),
  currentPath_( NULL ),
  currentRouteName_( NULL ), segment_percentage_( 0 ), gravity_on_(0), 
  gravity_force_( gravity_force )
{
}

Stealth::~Stealth()
{
}

void
Stealth::print()
{
  cerr << "Stealth print not done yet!\n";
}

void
Stealth::slideRight()
{
  increase_a_speed( horizontal_speed_, horizontal_accel_cnt_,
		    translate_scale_, translate_base, translate_max_count );
}

void
Stealth::slideLeft()
{
  decrease_a_speed( horizontal_speed_, horizontal_accel_cnt_,
		    translate_scale_, translate_base, translate_max_count );
}

void
Stealth::goUp()
{
  increase_a_speed( vertical_speed_, vertical_accel_cnt_,
		    translate_scale_, translate_base, translate_max_count );
}

void
Stealth::goDown()
{
  decrease_a_speed( vertical_speed_, vertical_accel_cnt_,
		    translate_scale_, translate_base, translate_max_count );
}

void
Stealth::accelerate()
{
  increase_a_speed( speed_, accel_cnt_, translate_scale_,
		    translate_base, translate_max_count );
}

void
Stealth::decelerate()
{
  decrease_a_speed( speed_, accel_cnt_, translate_scale_,
		    translate_base, translate_max_count );
}

void
Stealth::increase_a_speed( double & speed, int & accel_cnt,
			   double scale, double base, double max_count)
{
  if( accel_cnt > max_count ) {
    cout << "Going too fast... can't speed up\n";
    return;
  }

  accel_cnt++;

  // Amount of acceleration doubles for each consecutive request.
  double amount;

  cout << speed << ", " << accel_cnt << ", " << scale << ", " << base
       << ", " << max_count << "\n";

  if( accel_cnt == 0 )
    amount = 1 / scale;
  else if( accel_cnt > 0 )
    amount = pow(base, (double)accel_cnt - 1.0) / scale;
  else
    amount = pow(base, (double)(-accel_cnt)) / scale;

  printf("accelerating by %lf\n", amount);

  speed += amount;
}

void
Stealth::decrease_a_speed( double & speed, int & accel_cnt, double scale, double base, double max_count )
{
  if( accel_cnt < -max_count ) {
    cout << "Going too fast (in reverse)... can't speed up\n";
    return;
  }

  accel_cnt--;

  // Amount of acceleration doubles for each consecutive request.
  double amount; // = pow(base, (double)abs(accel_cnt)) / scale_;

  if( accel_cnt >= 0 )
    amount = pow(base, accel_cnt) / scale;
  else
    amount = pow(base, (double)(-accel_cnt) - 1.0) / scale;

  printf("reversing by %lf\n", amount);

  speed -= amount;
}

void
Stealth::turnRight()
{
  increase_a_speed( rotate_speed_, rotate_accel_cnt_,
		    rotate_scale_, rotate_base, rotate_max_count );
}

void
Stealth::turnLeft()
{
  decrease_a_speed( rotate_speed_, rotate_accel_cnt_,
		    rotate_scale_, rotate_base, rotate_max_count );
}

void
Stealth::pitchUp()
{

  cout << "pitch accel cnt: " << pitch_accel_cnt_ << "\n";
  cout << "2: pitch accel cnt: " << &pitch_accel_cnt_ << "\n";

  increase_a_speed( pitch_speed_, pitch_accel_cnt_,
		    rotate_scale_, rotate_base, rotate_max_count );
  cout << "PITCH ACCEL CNT: " << pitch_accel_cnt_ << "\n";
  cout << "2: PITCH ACCEL CNT: " << &pitch_accel_cnt_ << "\n";
}

void
Stealth::pitchDown()
{
  decrease_a_speed( pitch_speed_, pitch_accel_cnt_,
		    rotate_scale_, rotate_base, rotate_max_count );
}

void
Stealth::stopAllMovement()
{
  speed_ = 0;
  horizontal_speed_ = 0;
  vertical_speed_ = 0;
  pitch_speed_ = 0;
  rotate_speed_ = 0;

  accel_cnt_ = 0;
  horizontal_accel_cnt_ = 0;
  vertical_accel_cnt_ = 0;
  pitch_accel_cnt_ = 0;
  rotate_accel_cnt_ = 0;

  cout << "stop all movement\n";
}

void
Stealth::stopPitch()
{
  pitch_speed_ = 0;
  pitch_accel_cnt_ = 0;

  cout << "stop pitch\n";
}

void
Stealth::stopRotate()
{
  rotate_speed_ = 0;
  rotate_accel_cnt_ = 0;
}

void
Stealth::stopPitchAndRotate()
{
  cout << "Stop pitch and rotate\n";

  stopPitch();
  stopRotate();
}

double
Stealth::getSpeed( int direction ) const {
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

void
Stealth::slowDown()
{
  if( speed_ > 0 ) {
    decrease_a_speed( speed_, accel_cnt_, translate_scale_,
		      translate_base, translate_max_count );
  } else if( speed_ < 0 ) {
    increase_a_speed( speed_, accel_cnt_, translate_scale_,
		      translate_base, translate_max_count );
  }

  if( horizontal_speed_ > 0 ) {
    decrease_a_speed( horizontal_speed_, horizontal_accel_cnt_,
		      translate_scale_, translate_base, translate_max_count );
  } else if( horizontal_speed_ < 0 ) {
    increase_a_speed( horizontal_speed_, horizontal_accel_cnt_,
		      translate_scale_, translate_base, translate_max_count );
  }

  if( vertical_speed_ > 0 ) {
    decrease_a_speed( vertical_speed_, vertical_accel_cnt_,
		      translate_scale_, translate_base, translate_max_count );
  } else if( vertical_speed_ < 0 ) {
    increase_a_speed( vertical_speed_, vertical_accel_cnt_,
		      translate_scale_, translate_base, translate_max_count );
  }

  if( pitch_speed_ > 0 ) {
    decrease_a_speed( pitch_speed_, pitch_accel_cnt_, rotate_scale_,
		      rotate_base, rotate_max_count );
  } else if( pitch_speed_ < 0 ) {
    increase_a_speed( pitch_speed_, pitch_accel_cnt_, rotate_scale_,
		      rotate_base, rotate_max_count );
  }

  if( rotate_speed_ > 0 ) {
    decrease_a_speed( rotate_speed_, rotate_accel_cnt_, rotate_scale_,
		      rotate_base, rotate_max_count );
  } else if( rotate_speed_ < 0 ) {
    increase_a_speed( rotate_speed_, rotate_accel_cnt_, rotate_scale_,
		      rotate_base, rotate_max_count );
  }
}



void
Stealth::toggleGravity()
{
  gravity_on_ = !gravity_on_; 
  cout << "Gravity is now: " << gravity_on_ << "\n";
}

bool
Stealth::moving()
{
  bool moven = speed_ != 0 || horizontal_speed_ != 0 || vertical_speed_ != 0 ||
    pitch_speed_ != 0 || rotate_speed_ != 0;

  return moven;
}

void
Stealth::updateRotateSensitivity( double scalar )
{
  rotate_scale_ = baseRotateScale_ / scalar;
}

void
Stealth::updateTranslateSensitivity( double scalar )
{
  translate_scale_ = baseTranslateScale_ / scalar;
}



// new code added .... 
// the purpose of this code is to use the catmull-rom splines for the
// path and not linear interpolation. this makes the camera path smooth.

Camera Stealth::using_catmull_rom(vector<Camera> &cameras, int i,  float t) {
  Camera d0,d1;
  Camera p0,p1,q0,q1;

  q0 = cameras[i];
  q1 = cameras[i+1];

  if(i == 0)
    d0 = q1 - q0;
  else
    d0 = (q1 - cameras[i-1])*0.5;
  
  if(i == cameras.size()-2)
    d1 = q1 - q0;
  else
    d1 = (cameras[i+2] - q0)*0.5;

  
  //p0
  p0 = q0 + (d0*(1/3));
  
  //p1
  p1 = q1 - (d1*(1/3));
  
  //calculation of interpolation
  return 
    q0 * ((1-t)*(1-t)*(1-t)) +
    p0 * (3*(1-t)*(1-t)*t) +
    p1 * (3*(1-t)*t*t) +
    q1 * (t*t*t);
}

Point Stealth::using_catmull_rom(vector<Point> &points, int i,  float t)

{

      
  //finding the tangent vectors;
  
  float d0_x, d0_y, d0_z;
  float d1_x, d1_y, d1_z;
  
  //for the eye......
  

  if(i == 0)
    {
      d0_x = ((points[i+1]).x()-(points[i]).x())/1.0f;
      d0_y = ((points[i+1]).y()-(points[i]).y())/1.0f;
      d0_z = ((points[i+1]).z()-(points[i]).z())/1.0f;
    }
  
  else{
    
    d0_x = ((points[i+1]).x()-(points[i-1]).x())/2.0f;
    d0_y = ((points[i+1]).y()-(points[i-1]).y())/2.0f;
    d0_z = ((points[i+1]).z()-(points[i-1]).z())/2.0f;
  }
  
  
  if(i == points.size()-2)
    {
      d1_x = ((points[i+1]).x()-(points[i]).x())/1.0f;
      
      d1_y = ((points[i+1]).y()-(points[i]).y())/1.0f;
      d1_z = ((points[i+1]).z()-(points[i]).z())/1.0f;
    }
  
  else
    {
      d1_x = ((points[i+2]).x()-(points[i]).x())/2.0f;
      d1_y = ((points[i+2]).y()-(points[i]).y())/2.0f;
      d1_z = ((points[i+2]).z()-(points[i]).z())/2.0f;
    }
  
  
  
  //p0
  float p0_x = (points[i]).x()+((1/3.0f)*d0_x);
  float p0_y = (points[i]).y()+((1/3.0f)*d0_y);
  float p0_z = (points[i]).z()+((1/3.0f)*d0_z);
  
  
  //p1
  float p1_x = (points[i+1]).x()-((1/3.0f)*d1_x);
  float p1_y = (points[i+1]).y()-((1/3.0f)*d1_y);
  float p1_z = (points[i+1]).z()-((1/3.0f)*d1_z);
  
  
  
  float q0_x = points[i].x();
  float q0_y = points[i].y();
  float q0_z = points[i].z();
  
  float q1_x = points[i+1].x();
  float q1_y = points[i+1].y();
  float q1_z = points[i+1].z();
  
  //calculation of points:
  
  float point_gen_eyex = (pow((1-t),3)*q0_x
                          + 3*(1.0f-t)*(1.0f-t)*t*(p0_x)
                          + 3*(1.0f-t)*t*t*(p1_x)
                          + t*t*t*q1_x);
  
  float point_gen_eyey = (pow((1-t),3)*q0_y
                          + 3*(1.0f-t)*(1.0f-t)*t*(p0_y)
                          + 3*(1.0f-t)*t*t*(p1_y)
			  + t*t*t*q1_y);
  
  float point_gen_eyez = (pow((1-t),3)*q0_z
                          + 3*(1.0f-t)*(1.0f-t)*t*(p0_z)
                          + 3*(1.0f-t)*t*t*(p1_z)
			  + t*t*t*q1_z);
  
  
  Point new_eye(point_gen_eyex, point_gen_eyey, point_gen_eyez);
  
  
  return (new_eye);
  
}




void
Stealth::getNextLocation( Camera* new_camera )
{
  if( !currentPath_ || currentPath_->size() < 2 ) {
    cerr << "Not enough path enformation!\n";
    return;
  }

  // This used to use the accel_cnt_ for stepping segment_percentage_.
  // However, this value was an integer the counted the number of
  // times you pressed accelerate.  If you wanted to set the speed
  // between these intervals you were out of luck.  Now, since using
  // the actuall speed_ variable (which is scaled by
  // translate_scale_), you can get slow and really fast speeds.  I'm
  // multiplying by 100 to make the default close to what it was with
  // the old method.
  segment_percentage_ += speed_*100.0;

  if( segment_percentage_ < 0 ) // Moving backward
    {
      // This will start us at the end of the path.
      cout << "got to the beginning of the path... going to end\n";
      segment_percentage_ = (((int)(currentPath_->size())-1) * 100) - 1;
    }
  int begin_index = (int)( segment_percentage_ / 100.0 );
  int end_index = begin_index+1;
  float t_catmull = ( segment_percentage_ / 100.0f )-begin_index;
    
  if( end_index >= (int)currentPath_->size() )
    {
      cout << "got to the end of the path... restarting\n";
      segment_percentage_ = 1;
      begin_index = 0; end_index = 1;
    }
  //eye
  *new_camera = using_catmull_rom(*currentPath_, begin_index, t_catmull);
  new_camera->setup();
}



void
Stealth::clearPath()
{
  cout << "Clearing Path\n";
  currentPath_->clear();
}

void
Stealth::deleteCurrentMarker()
{
  displayCurrentRoute();

  if( currentPath_->size() == 0 )
    return;

  int pos = (int)(segment_percentage_ / 100.0);

  currentPath_->erase( currentPath_->begin() + pos);

  segment_percentage_ -= 100; 

  displayCurrentRoute();
}

void
Stealth::addToPath( const Camera * new_camera )
{
  currentPath_->push_back( *new_camera );
}

void
Stealth::addToMiddleOfPath( const Camera * new_camera )
{
  displayCurrentRoute();

  unsigned int pos = (unsigned int)(segment_percentage_ / 100.0);

  if( pos == currentPath_->size() ) {
    // If at last point, add to end.
    cout << "at last point, adding to end of route\n";
    addToPath( new_camera );
    segment_percentage_ += 100; 
    return;
  }

  currentPath_->insert( currentPath_->begin() + pos+1, *new_camera );

  segment_percentage_ += 100; 

  displayCurrentRoute();
}

void
Stealth::displayCurrentRoute()
{
  vector<Camera>::iterator pIter = currentPath_->begin();

  cout << "Path: (" << segment_percentage_ << ")\n";

  for( ; pIter != currentPath_->end(); pIter++)
    {
      (*pIter).print();
    }

}

void
Stealth::newPath( const string & routeName )
{
  unsigned long index = paths_.size();

  vector<Camera> path;

  char name[1024];
  sprintf( name, "%s", routeName.c_str() );
  routeNames_.push_back( name );
  paths_.push_back( path );

  currentPath_        = &paths_[ index ];
  currentRouteName_   = &routeNames_[ index ];
}

// Returns 1 for error, 0 if everthing went well.
int Stealth::loadOldPath(const string& filename) {
  FILE * fp = fopen( filename.c_str(), "r" );
  if( fp == NULL )
    {
      string path = "/usr/sci/data/Geometry/paths/";
      cout << "Looked for " << filename << " in local path... now looking "
	   << "in " << path << "\n";
      fp = fopen( (path+filename).c_str(), "r" );
      if( fp == NULL )
	{
	  cout << "Did not find it... not loading path\n";
	  return 1;
	}
    }

  unsigned long index = paths_.size();

  vector<Camera> path;
  vector<Point> eye;
  vector<Point> lookat;

  char name[1024];
  fgets( name, 1024, fp );

  routeNames_.push_back( name );
  paths_.push_back( path );

  currentPath_        = &paths_[ index ];
  currentRouteName_   = &routeNames_[ index ];

  int size = 0;
  fscanf( fp, "%u\n", &size );

  // Default values for up and fov
  Vector up (0,1,0);
  double fov = 45;
  
  for( int cnt = 0; cnt < size; cnt++ ) {
    double ex,ey,ez, lx,ly,lz;
    fscanf( fp, "%lf %lf %lf  %lf %lf %lf\n", &ex,&ey,&ez, &lx,&ly,&lz );
    
    Point eye(ex,ey,ez);
    Point lookat(lx,ly,lz);
    Camera camera(eye, lookat, up, fov);
    camera.setup();
    addToPath( &camera );
  }
  fclose( fp );

  return 0;
}

// This needs a lot of work to get it to work with the new cameras
string
Stealth::loadPath( const string & filename )
{
  bool do_old_path = false;
  // Do something to determing the old path stuff
  if (do_old_path) {
    if (loadOldPath(filename)) {
      // there was an error
      return "";
    }
  }

  Piostream *stream = auto_istream(filename);
  if (!stream) {
    cerr << "Error opening route file '" << filename << "'.\n";
    return "";
  }
  
  vector<Camera> path;
  string name;
  int size;

  Pio(*stream, name);
  if (stream->error()) {
    cerr << "Error loading name of route\n";
    delete stream;
    return "";
  }
  Pio(*stream, size);
  if (stream->error()) {
    cerr << "Error loading size of route\n";
    delete stream;
    return "";
  }

  routeNames_.push_back( name );
  paths_.push_back( path );

  currentPath_        = &(paths_.back());
  currentRouteName_   = &(routeNames_.back());

  for(int i = 0; i < size; i++) {
    Camera camera;
    Pio(*stream, camera);
    if (stream->error()) {
      cerr << "Error loading camera ["<<i<<"] of route\n";
      delete stream;
      return "";
    }
    camera.setup();
    addToPath( &camera );
  }
  
  delete stream;

  return *currentRouteName_;
}

void
Stealth::saveOldPath( const string & filename )
{
  FILE * fp = fopen( filename.c_str(), "w" );
  if( fp == NULL )
    {
      cout << "error opening file " << filename << ".  Did not save path.\n";
      return;
    }

  cout << "Saving path\n";

  fprintf( fp, "%s\n", (*currentRouteName_).c_str() );

  fprintf( fp, "%d\n", (int)currentPath_->size() );
  for( size_t cnt = 0; cnt < currentPath_->size(); cnt++ )
    {
      Point eye = (*currentPath_)[cnt].get_eye();
      Point lookat = (*currentPath_)[cnt].get_lookat();
      fprintf( fp, "%2.2lf %2.2lf %2.2lf  %2.2lf %2.2lf %2.2lf\n",
	       eye.x(), eye.y(), eye.z(),
               lookat.x(), lookat.y(), lookat.z() );
    }
  fclose( fp );
}

void
Stealth::savePath( const string & filename ) {
  if( currentPath_->size() < 2 )
    {
      cout << "no path. not saving\n";
      return;
    }

  bool do_old_save = false;
  if (do_old_save) {
    saveOldPath(filename);
    return;
  }

  // Open up the output stream
  Piostream* stream;
  stream = new TextPiostream(filename, Piostream::Write);
  if (stream->error()) {
    cerr << "Could not open file '"<<filename<<"' for writing\n";
    return;
  }

  // Write the file
  string name = *currentRouteName_;
  int size = (int)currentPath_->size();

  Pio(*stream, name);
  Pio(*stream, size);

  for(int i = 0; i < size; i++) {
    Pio(*stream, (*currentPath_)[i]);
  }
  
  delete stream;
}

int
Stealth::getNextMarker( Camera * new_camera )
{
  // If no route...
  if( currentPath_->size() == 0 )
    return -1;

  size_t end_index  = (size_t)(segment_percentage_ / 100.0 + 1);

  if( end_index >= currentPath_->size() )
    end_index = 0;

  segment_percentage_ = (double)(end_index * 100);
  *new_camera = (*currentPath_)[ end_index ];

  return end_index;
}

int
Stealth::getPrevMarker( Camera * new_camera )
{
  // If no route...
  if( currentPath_->size() == 0 )
    return -1;

  int begin_index;

  if( segment_percentage_ == 0 ) {
    begin_index = (int)currentPath_->size() - 1;
  } else {
    begin_index = (int)(segment_percentage_ / 100.0);
    // If we are on a marker point, then back up to the previous one.
    if( ((int)segment_percentage_) % 100 == 0 ) 
      begin_index--;
  }
  segment_percentage_ = begin_index * 100;
  *new_camera = (*currentPath_)[ begin_index ];

  return begin_index;
}

int
Stealth::goToBeginning( Camera * new_camera )
{
  // If no route...
  if( currentPath_->size() == 0 )
    return -1;

  segment_percentage_ = 0;
  *new_camera = (*currentPath_)[ 0 ];

  return 0;
}

void
Stealth::selectPath( int index )
{
  if( index < (int)paths_.size() && index >= 0 ) {
    currentPath_      = &paths_[ index ];
    currentRouteName_ = &routeNames_[ index ];
    cout << "route id " << index << ", name " << (*currentRouteName_) <<"\n";
  }
}

void
Stealth::getRouteStatus( int & pos, int & numPts )
{
  if( !currentPath_ || currentPath_->size() == 0 ) {
    pos = -1;
    numPts = 0;
  } else {
    pos    = (int)(segment_percentage_ / 100.0);
    numPts = (int)currentPath_->size();
  }
}

