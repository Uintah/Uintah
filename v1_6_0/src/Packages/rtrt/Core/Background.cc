
#include <Packages/rtrt/Core/Background.h>
#include <Packages/rtrt/Core/PPMImage.h>
#include <Core/Math/MiscMath.h>

#include <fstream>
#include <Core/Geometry/Vector.h>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

Persistent* cbackground_maker() {
  return new ConstantBackground();
}

// initialize the static member type_id
PersistentTypeID Background::type_id("Background", "Persistent", 0);
PersistentTypeID ConstantBackground::type_id("ConstantBackground", 
					     "Background", 
					     cbackground_maker);

Background::Background(const Color& avg)
  : avg(avg), origAvg_( avg )
{
}

Background::~Background() {}


// *****************************************************************
//     ConstantBackground members

ConstantBackground::ConstantBackground(const Color& C) : 
  Background(C), C(C), origC_(C) {}

ConstantBackground::~ConstantBackground() {}

void ConstantBackground::color_in_direction( const Vector&, Color& result) const
{
  result=C;
}

// *****************************************************************
//     LinearBackground members


LinearBackground::~LinearBackground() {}

LinearBackground::LinearBackground( const Color& C1, 
				    const Color& C2,
				    const Vector& direction_to_C1) :
  Background(C1), C1(C1), C2(C2), origC1_(C1), origC2_(C2),
  direction_to_C1(direction_to_C1)
{
}

void LinearBackground::color_in_direction(const Vector& v, 
					  Color& result) const 
{
    double t = 0.5* (1 + Dot(v, direction_to_C1 ) );
    result=t*C1 + (1-t)*C2;
}

inline  int IndexOfMinAbsComponent( const Vector& v ) 
{
    if ( fabs(v.x()) <= fabs(v.y()) && fabs(v.x()) <= fabs(v.z()))
            return 0;
        else if( fabs(v.y()) <= fabs(v.z()) )
            return 1;
        else
            return 2;
    }

inline Vector PerpendicularVector( const Vector& v ) 
{
   int axis = IndexOfMinAbsComponent( v );
   if( axis == 0 )
      return Vector( 0.0, v.z(), -v.y() );
   else if ( axis == 1 )
      return Vector( v.z(), 0.0, -v.x() );
   else
      return Vector( v.y(), -v.x(), 0.0 );
   }


EnvironmentMapBackground::EnvironmentMapBackground( char* filename,
						    const Vector& up ) :
    Background( Color( 0, 0, 0 ) ),
    _width( 0 ),
    _height( 0 ),
    _up( up ),
    ambientScale_( 1.0 )
{
  //
  // Built an orthonormal basis
  //
  _up.normalize();
  _u = PerpendicularVector( _up );
  _v = Cross( _up, _u );
  PPMImage ppm(filename);
  ppm.get_dimensions_and_data(_image, _width, _height);
  
  cout << "env_map width, height: " << _width << ", " << _height << endl;
}

EnvironmentMapBackground::~EnvironmentMapBackground( void )
{
}

void
EnvironmentMapBackground::color_in_direction( const Vector& DIR , Color& result) const
{
    //
    // Convert to local basis
    //
    Vector dir = ChangeToBasis( DIR );

    // cerr << "direction  = " << dir << endl;
    // Map direction vector dir to (u,v) coordinates
    // cerr << "Length = " << dir.length() << endl;
    //double r = atan2( sqrt( dir.x()*dir.x() + dir.y()*dir.y() ), dir.z() );
    //r /= M_PI;  /* -0.5 .. 0.5 */
    //double phi = atan2( dir.y(), dir.x() );
    double r =  sqrt( dir.x()*dir.x() + dir.y()*dir.y() );
    double v = atan2( dir.x(), dir.y() ) / ( 2.0*M_PI ) + 0.5;
    double u = atan2( r, dir.z() ) / M_PI;
    // double u = ( ( atan2( dir.y(), dir.x() ) + M_PI ) / ( 2.0*M_PI ) );
    // double v = ( ( asin( dir.z() ) + (0.5*M_PI) ) / M_PI );
    //double u = Clamp( r * cos( phi ) + 0.5, 0.0, 1.0 );
    //double v = Clamp( r * sin( phi ) + 0.5, 0.0, 1.0 );

    //double l1 = sqrt(dir.x()*dir.x()+dir.y()*dir.y());
    //double l2 = sqrt(dir.x()*dir.x()+dir.y()*dir.y()+dir.z()*dir.z());
    //double v = asin(dir.z()/l2);
    //double u = (dir.x()<0)?M_PI-asin(dir.y()/l1):asin(dir.y()/l1);      

    //dir.normalize();
    
    //double v = (DIR.x()+1)/2.;
    //double u = (DIR.z()+1)/2.;
    
    int i=int( v*( _width - 1 ) );
    int j=int( u*( _height - 1 ) );
    result = _image( i, j ) * ambientScale_;
}

const int BACKGROUND_VERSION = 1;
const int CBACKGROUND_VERSION = 1;

void 
Background::io(SCIRun::Piostream &str)
{
  str.begin_class("Background", BACKGROUND_VERSION);
  Background::io(str);
  SCIRun::Pio(str, avg);
  SCIRun::Pio(str, origAvg_);
  str.end_class();
}
void 
ConstantBackground::io(SCIRun::Piostream &str)
{
  str.begin_class("ConstantBackground", CBACKGROUND_VERSION);
  SCIRun::Pio(str, C);
  SCIRun::Pio(str, origC_);
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Background*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Background::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::Background*>(pobj);
    //ASSERT(obj != 0)
  }
}
} // end namespace SCIRun
