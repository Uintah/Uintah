
#include <Packages/rtrt/Core/Stealth.h>
#include <stdio.h>
#include <math.h>

using namespace rtrt;
using namespace SCIRun;
using std::cerr;

Stealth::Stealth( double scale, double gravity_force ) :
  speed_( 0 ), horizontal_speed_( 0 ), vertical_speed_( 0 ),
  pitch_speed_( 0 ), rotate_speed_( 0 ),
  accel_cnt_( 0 ), horizontal_accel_cnt_( 0 ),
  vertical_accel_cnt_( 0 ), pitch_accel_cnt_( 0 ), rotate_accel_cnt_( 0 ),
  scale_( scale ), segment_percentage_( 0 ), gravity_force_( gravity_force )
{
  cout << "scale is " << scale << "\n";

  path_.push_back( Point(-5.5,-5,5) );
  path_.push_back( Point(4.5,5.3,1) );
  path_.push_back( Point(0,10.2,5.1) );
  path_.push_back( Point(5.4,4.5,1.2) );
  path_.push_back( Point(-5.2,-5.3,5.4) );

  look_ats_.push_back( Point( 0, 0, 0 ) );
  look_ats_.push_back( Point( 0, 0, 0 ) );
  look_ats_.push_back( Point( 0, 0, 0 ) );
  look_ats_.push_back( Point( 0, 0, 0 ) );
  look_ats_.push_back( Point( 0, 0, 0 ) );
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
  increase_a_speed( horizontal_speed_, horizontal_accel_cnt_ );
}

void
Stealth::slideLeft()
{
  decrease_a_speed( horizontal_speed_, horizontal_accel_cnt_ );
}

void
Stealth::goUp()
{
  increase_a_speed( vertical_speed_, vertical_accel_cnt_ );
}

void
Stealth::goDown()
{
  decrease_a_speed( vertical_speed_, vertical_accel_cnt_ );
}

void
Stealth::accelerate()
{
  increase_a_speed( speed_, accel_cnt_ );
}

void
Stealth::decelerate()
{
  decrease_a_speed( speed_, accel_cnt_ );
}

static double base = 1.8; // 2.0 gets too big to quickly... i think
static int    max_accel_rate = 8;
void
Stealth::increase_a_speed( double & speed, int & accel_cnt )
{
  if( accel_cnt > max_accel_rate ) {
    cout << "Going too fast... can't speed up\n";
    return;
  }

  accel_cnt++;

  // Amount of acceleration doubles for each consecutive request.
  double amount;

  if( accel_cnt == 0 )
    amount = 1 / scale_;
  else if( accel_cnt > 0 )
    amount = pow(base, (double)accel_cnt - 1.0) / scale_;
  else
    amount = pow(base, (double)(-accel_cnt)) / scale_;

  printf("accelerating by %lf\n", amount);

  speed += amount;
}

void
Stealth::decrease_a_speed( double & speed, int & accel_cnt )
{
  if( accel_cnt < -max_accel_rate ) {
    cout << "Going too fast (in reverse)... can't speed up\n";
    return;
  }

  accel_cnt--;

  // Amount of acceleration doubles for each consecutive request.
  double amount; // = pow(base, (double)abs(accel_cnt)) / scale_;

  if( accel_cnt >= 0 )
    amount = pow(base, accel_cnt) / scale_;
  else
    amount = pow(base, (double)(-accel_cnt) - 1.0) / scale_;

  printf("reversing by %lf\n", amount);

  speed -= amount;
}

void
Stealth::turnRight()
{
  increase_a_speed( rotate_speed_, rotate_accel_cnt_ );
}

void
Stealth::turnLeft()
{
  decrease_a_speed( rotate_speed_, rotate_accel_cnt_ );
}

void
Stealth::pitchUp()
{
  increase_a_speed( pitch_speed_, pitch_accel_cnt_ );
}

void
Stealth::pitchDown()
{
  decrease_a_speed( pitch_speed_, pitch_accel_cnt_ );
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
}

void
Stealth::stopPitch()
{
  pitch_speed_ = 0;
  pitch_accel_cnt_ = 0;
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
    decrease_a_speed( speed_, accel_cnt_ );
  } else if( speed_ < 0 ) {
    increase_a_speed( speed_, accel_cnt_ );
  }

  if( horizontal_speed_ > 0 ) {
    decrease_a_speed( horizontal_speed_, horizontal_accel_cnt_ );
  } else if( horizontal_speed_ < 0 ) {
    increase_a_speed( horizontal_speed_, horizontal_accel_cnt_ );
  }

  if( vertical_speed_ > 0 ) {
    decrease_a_speed( vertical_speed_, vertical_accel_cnt_ );
  } else if( vertical_speed_ < 0 ) {
    increase_a_speed( vertical_speed_, vertical_accel_cnt_ );
  }

  if( pitch_speed_ > 0 ) {
    decrease_a_speed( pitch_speed_, pitch_accel_cnt_ );
  } else if( pitch_speed_ < 0 ) {
    increase_a_speed( pitch_speed_, pitch_accel_cnt_ );
  }

  if( rotate_speed_ > 0 ) {
    decrease_a_speed( rotate_speed_, rotate_accel_cnt_ );
  } else if( rotate_speed_ < 0 ) {
    increase_a_speed( rotate_speed_, rotate_accel_cnt_ );
  }
}

void
Stealth::getNextLocation( Point & point, Point & look_at )
{
  if( path_.size() < 2 ) {
    cout << "Not enough path enformation!\n";
    return;
  }

  segment_percentage_ += accel_cnt_;

  if( segment_percentage_ < 0 ) // Moving backward
    {
      // This will start us at the end of the path.
      cout << "got to the beginning of the path... going to end\n";
      segment_percentage_ = ((path_.size()-1) * 100) - 1;
    }

  int begin_index = segment_percentage_ / 100;
  int end_index   = (segment_percentage_ / 100) + 1;

  if( end_index == path_.size() )
    {
      cout << "got to the end of the path... restarting\n";
      segment_percentage_ = 1;
      begin_index = 0; end_index = 1;
    }

  Point & begin = path_[ begin_index ];
  Point & end = path_[ end_index ];

  double local_percent = segment_percentage_ - (100*begin_index);

  point = begin + ((end - begin) * (local_percent / 100.0));

  Point & begin_at = look_ats_[ begin_index ];
  Point & end_at   = look_ats_[ end_index ];

  look_at = begin_at + ((end_at - begin_at) * (local_percent / 100.0));
}


void
Stealth::clearPath()
{
  cout << "Clearing Path\n";
  path_.clear();
  look_ats_.clear();
}

void
Stealth::addToPath( const Point & point, const Point & look_at )
{
  cout << "Adding point to path\n";
  path_.push_back( point );
  look_ats_.push_back( look_at );
}

void
Stealth::loadPath( const string & filename )
{
  FILE * fp = fopen( filename.c_str(), "r" );
  if( fp == NULL )
    {
      cout << "Error opening file " << filename << ".  Did not load path.\n";
      return;
    }

  cout << "loading path\n";
  clearPath();

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
}

void
Stealth::savePath( const string & filename )
{
  FILE * fp = fopen( filename.c_str(), "w" );
  if( fp == NULL )
    {
      cout << "error opening file " << filename << ".  Did not save path.\n";
      return;
    }

  cout << "Saving path\n";

  fprintf( fp, "%d\n", (int)path_.size() );
  for( int cnt = 0; cnt < path_.size(); cnt++ )
    {
      fprintf( fp, "%2.2lf %2.2lf %2.2lf  %2.2lf %2.2lf %2.2lf\n",
	       path_[cnt].x(), path_[cnt].y(), path_[cnt].z(),
	       look_ats_[cnt].x(), look_ats_[cnt].y(), look_ats_[cnt].z() );
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

