
#include <Packages/rtrt/Core/Stealth.h>
#include <stdio.h>
#include <math.h>

using namespace rtrt;
using namespace SCIRun;
using std::cerr;

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
  currentPath_( NULL ), currentLookAts_( NULL ), currentRouteName_( NULL ),
  segment_percentage_( 0 ), gravity_on_(0), gravity_force_( gravity_force )
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
Stealth::getNextLocation( Point & point, Point & look_at )
{
  if( !currentPath_ || currentPath_->size() < 2 ) {
    cout << "Not enough path enformation!\n";
    return;
  }

  segment_percentage_ += accel_cnt_;

  if( segment_percentage_ < 0 ) // Moving backward
    {
      // This will start us at the end of the path.
      cout << "got to the beginning of the path... going to end\n";
      segment_percentage_ = (((int)(currentPath_->size())-1) * 100) - 1;
    }

  int begin_index = (int)( segment_percentage_ / 100.0 );
  int end_index   = begin_index + 1;

  if( end_index == (int)currentPath_->size() )
    {
      cout << "got to the end of the path... restarting\n";
      segment_percentage_ = 1;
      begin_index = 0; end_index = 1;
    }

  Point & begin = (*currentPath_)[ begin_index ];
  Point & end   = (*currentPath_)[ end_index ];

  double local_percent = segment_percentage_ - (100*begin_index);

  point = begin + ((end - begin) * (local_percent / 100.0));

  Point & begin_at = (*currentLookAts_)[ begin_index ];
  Point & end_at   = (*currentLookAts_)[ end_index ];

  look_at = begin_at + ((end_at - begin_at) * (local_percent / 100.0));
}


void
Stealth::clearPath()
{
  cout << "Clearing Path\n";
  currentPath_->clear();
  currentLookAts_->clear();
}

void
Stealth::deleteCurrentMarker()
{
  displayCurrentRoute();

  if( currentPath_->size() == 0 )
    {
      return;
    }

  int pos = (int)(segment_percentage_ / 100.0);

  vector<Point>::iterator pIter = currentPath_->begin();
  vector<Point>::iterator lIter = currentLookAts_->begin();

  for( int cnt = 0; cnt < pos; cnt++ )
    {
      pIter++;
      lIter++;
    }
  currentLookAts_->erase( lIter );
  currentPath_->erase( pIter );
  segment_percentage_ -= 100; 

  displayCurrentRoute();

}

void
Stealth::addToPath( const Point & point, const Point & look_at )
{
  currentPath_->push_back( point );
  currentLookAts_->push_back( look_at );
}

void
Stealth::addToMiddleOfPath( const Point & eye, const Point & look_at )
{
  displayCurrentRoute();

  unsigned int pos = (unsigned int)(segment_percentage_ / 100.0);

  if( pos == currentPath_->size() ) // If at last point, add to end.
    {
      cout << "at last point, adding to end of route\n";
      addToPath( eye, look_at );
      segment_percentage_ += 100; 
      return;
    }

  vector<Point>::iterator pIter = currentPath_->begin();
  vector<Point>::iterator lIter = currentLookAts_->begin();

  for( unsigned long cnt = 0; cnt <= pos; cnt++ )
    {
      pIter++;
      lIter++;
    }
  currentLookAts_->insert( lIter, look_at );
  currentPath_->insert( pIter, eye );
  segment_percentage_ += 100; 

  displayCurrentRoute();

}

void
Stealth::displayCurrentRoute()
{
  vector<Point>::iterator pIter = currentPath_->begin();
  vector<Point>::iterator lIter = currentLookAts_->begin();

  cout << "Path: (" << segment_percentage_ << ")\n";

  for( ; pIter != currentPath_->end(); pIter++,lIter++ )
    {
      cout << *pIter << ", " << *lIter << "\n";
    }

}

void
Stealth::newPath( const string & routeName )
{
  unsigned long index = paths_.size();

  vector<Point> path;
  vector<Point> lookat;

  char name[1024];
  sprintf( name, "%s", routeName.c_str() );
  routeNames_.push_back( name );
  paths_.push_back( path );
  lookAts_.push_back( lookat );

  currentPath_        = &paths_[ index ];
  currentLookAts_     = &lookAts_[ index ];
  currentRouteName_   = &routeNames_[ index ];
}

string
Stealth::loadPath( const string & filename )
{
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
	  return "";
	}
    }

  unsigned long index = paths_.size();

  vector<Point> path;
  vector<Point> lookat;

  char name[1024];
  fgets( name, 1024, fp );

  routeNames_.push_back( name );
  paths_.push_back( path );
  lookAts_.push_back( lookat );

  currentPath_        = &paths_[ index ];
  currentLookAts_     = &lookAts_[ index ];
  currentRouteName_   = &routeNames_[ index ];

  int size = 0;
  fscanf( fp, "%u\n", &size );
  for( int cnt = 0; cnt < size; cnt++ )
    {
      double ex,ey,ez, lx,ly,lz;
      fscanf( fp, "%lf %lf %lf  %lf %lf %lf\n", &ex,&ey,&ez, &lx,&ly,&lz );

      Point eye(ex,ey,ez);
      Point lookat(lx,ly,lz);
      addToPath( eye, lookat );
    }
  fclose( fp );

  return *currentRouteName_;
}

void
Stealth::savePath( const string & filename )
{
  if( currentPath_->size() < 2 )
    {
      cout << "no path. not saving\n";
      return;
    }

  FILE * fp = fopen( filename.c_str(), "w" );
  if( fp == NULL )
    {
      cout << "error opening file " << filename << ".  Did not save path.\n";
      return;
    }

  cout << "Saving path\n";

  fprintf( fp, "%s\n", (*currentRouteName_).c_str() );

  fprintf( fp, "%d\n", (int)currentPath_->size() );
  for( unsigned int cnt = 0; cnt < currentPath_->size(); cnt++ )
    {
      fprintf( fp, "%2.2lf %2.2lf %2.2lf  %2.2lf %2.2lf %2.2lf\n",
	       (*currentPath_)[cnt].x(), (*currentPath_)[cnt].y(),
	       (*currentPath_)[cnt].z(), (*currentLookAts_)[cnt].x(),
	       (*currentLookAts_)[cnt].y(), (*currentLookAts_)[cnt].z() );
    }
  fclose( fp );
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


int
Stealth::getNextMarker( Point & point, Point & look_at )
{
  unsigned int end_index  = (int)(segment_percentage_ / 100.0 + 1);

  // If no route...
  if( currentPath_->size() == 0 ) { return -1; }

  if( end_index >= currentPath_->size() )
    {
      end_index = 0;
    }

  segment_percentage_ = end_index * 100;
  point   = (*currentPath_)[ end_index ];
  look_at = (*currentLookAts_)[ end_index ];

  return end_index;
}

int
Stealth::getPrevMarker( Point & point, Point & look_at )
{
  int begin_index;

  // If no route...
  if( currentPath_->size() == 0 ) { return -1; }

  if( segment_percentage_ == 0 )
    {
      begin_index = (int)currentPath_->size() - 1;
    }
  else
    {
      begin_index = (int)(segment_percentage_ / 100.0);
      // If we are on a marker point, then back up to the previous one.
      if( ((int)segment_percentage_) % 100 == 0 )
	{
	  begin_index--;
	}
    }
  segment_percentage_ = begin_index * 100;
  point   = (*currentPath_)[ begin_index ];
  look_at = (*currentLookAts_)[ begin_index ];

  return begin_index;
}

int
Stealth::goToBeginning( Point & point, Point & look_at )
{
  // If no route...
  if( currentPath_->size() == 0 ) { return -1; }

  segment_percentage_ = 0;
  point   = (*currentPath_)[ 0 ];
  look_at = (*currentLookAts_)[ 0 ];

  return 0;
}

void
Stealth::selectPath( int index )
{
  if( index < (int)paths_.size() && index >= 0 )
    {
      currentPath_      = &paths_[ index ];
      currentLookAts_   = &lookAts_[ index ];
      currentRouteName_ = &routeNames_[ index ];
      cout << "route id " << index << ", name " << (*currentRouteName_) <<"\n";
    }
}

void
Stealth::getRouteStatus( int & pos, int & numPts )
{
  if( !currentPath_ || currentPath_->size() == 0 )
    {
      pos = -1; numPts = 0;
    }
  else
    {
      pos    = (int)(segment_percentage_ / 100.0);
      numPts = (int)currentPath_->size();
    }
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
