
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
Persistent* abackground_maker() {
  return new AmbientBackground();
}
Persistent* lbackground_maker() {
  return new LinearBackground();
}
Persistent* embackground_maker() {
  return new EnvironmentMapBackground();
}

// initialize the static member type_id
PersistentTypeID Background::type_id("Background", "Persistent", 0);
PersistentTypeID ConstantBackground::type_id("ConstantBackground", 
					     "Background", 
					     cbackground_maker);
PersistentTypeID AmbientBackground::type_id("AmbientBackground", 
					     "Background", 
					     cbackground_maker);
PersistentTypeID LinearBackground::type_id("LinearBackground", 
					   "Background", 
					   lbackground_maker);
PersistentTypeID EnvironmentMapBackground::type_id("EnvironmentMapBackground", 
						   "Background", 
						   embackground_maker);

Background::~Background() {}


// *****************************************************************
//     ConstantBackground members

ConstantBackground::ConstantBackground(const Color& C) : 
  C(C) {}

ConstantBackground::~ConstantBackground() {}

void ConstantBackground::color_in_direction(const Vector&, Color& result) const
{
  result=C;
}

// *****************************************************************
//     AmbientBackground members

AmbientBackground::AmbientBackground(const Color& C) : 
  C(C), origC_(C) {}

AmbientBackground::~AmbientBackground() {}

void AmbientBackground::color_in_direction(const Vector&, Color& result) const
{
  result=C;
}

// *****************************************************************
//     LinearBackground members


LinearBackground::~LinearBackground() {}

LinearBackground::LinearBackground( const Color& C1, 
				    const Color& C2,
				    const Vector& direction_to_C1) :
  C1(C1), C2(C2), direction_to_C1(direction_to_C1)
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
    ambientScale_( 1.0 ),
    _width( 0 ), _height( 0 ),
    _up( up )
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
  Vector dir = DIR /*ChangeToBasis( DIR )*/;
  dir=Vector(-Dot(DIR,_u), -Dot(DIR,_v), -Dot(DIR,_up));

    // cerr << "direction  = " << dir << endl;
    // Map direction vector dir to (u,v) coordinates
    // cerr << "Length = " << dir.length() << endl;
    //double r = atan2( sqrt( dir.x()*dir.x() + dir.y()*dir.y() ), dir.z() );
    //r /= M_PI;  /* -0.5 .. 0.5 */
    //double phi = atan2( dir.y(), dir.x() );
  //double r =  sqrt( dir.x()*dir.x() + dir.y()*dir.y() );
  //double v = atan2( dir.x(), dir.y() ) / ( 2.0*M_PI ) + 0.5;
  //double u = atan2( r, dir.z() ) / M_PI;
    double u = ( ( atan2( dir.y(), dir.x() ) + M_PI ) / ( 2.0*M_PI ) );
    double v = ( ( asin( dir.z() ) + (0.5*M_PI) ) / M_PI );
    //double u = Clamp( r * cos( phi ) + 0.5, 0.0, 1.0 );
    //double v = Clamp( r * sin( phi ) + 0.5, 0.0, 1.0 );

    //double l1 = sqrt(dir.x()*dir.x()+dir.y()*dir.y());
    //double l2 = sqrt(dir.x()*dir.x()+dir.y()*dir.y()+dir.z()*dir.z());
    //double v = asin(dir.z()/l2);
    //double u = (dir.x()<0)?M_PI-asin(dir.y()/l1):asin(dir.y()/l1);      

    //dir.normalize();
    
    //double v = (DIR.x()+1)/2.;
    //double u = (DIR.z()+1)/2.;
    
    //int i=int( v*( _width - 1 ) );
    //int j=int( u*( _height - 1 ) );
    //result = _image( i, j ) * ambientScale_;

    // image dimensions include interpolation slop
    u *= (_image.dim1()-3);
    v *= (_image.dim2()-3);
    
    int iu = (int)u;
    int iv = (int)v;

    double tu = u-iu;
    double tv = v-iv;

    result = _image(iu,iv)*(1-tu)*(1-tv)+
      _image(iu+1,iv)*tu*(1-tv)+
      _image(iu,iv+1)*(1-tu)*tv+
      _image(iu+1,iv+1)*tu*tv;
}

const int BACKGROUND_VERSION = 1;
const int CBACKGROUND_VERSION = 1;
const int LBACKGROUND_VERSION = 1;
const int EMBACKGROUND_VERSION = 1;

void 
Background::io(SCIRun::Piostream &str)
{
  str.begin_class("Background", BACKGROUND_VERSION);
  str.end_class();
}

void 
ConstantBackground::io(SCIRun::Piostream &str)
{
  str.begin_class("ConstantBackground", CBACKGROUND_VERSION);
  Background::io(str);
  SCIRun::Pio(str, C);
  str.end_class();
}

void 
AmbientBackground::io(SCIRun::Piostream &str)
{
  str.begin_class("AmbientBackground", CBACKGROUND_VERSION);
  Background::io(str);
  SCIRun::Pio(str, C);
  SCIRun::Pio(str, origC_);
  str.end_class();
}

void 
LinearBackground::io(SCIRun::Piostream &str)
{
  str.begin_class("LinearBackground", LBACKGROUND_VERSION);
  Background::io(str);
  SCIRun::Pio(str, C1);
  SCIRun::Pio(str, C2);
  SCIRun::Pio(str, direction_to_C1);
  str.end_class();
}

void 
EnvironmentMapBackground::io(SCIRun::Piostream &str)
{
  str.begin_class("EnvironmentMapBackground", EMBACKGROUND_VERSION);
  Background::io(str);
  SCIRun::Pio(str, ambientScale_);
  SCIRun::Pio(str, _width);
  SCIRun::Pio(str, _height);
  rtrt::Pio(str, _image);
  SCIRun::Pio(str, _up);
  SCIRun::Pio(str, _u);
  SCIRun::Pio(str, _v);
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

void Pio(SCIRun::Piostream& stream, rtrt::ConstantBackground*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::ConstantBackground::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::ConstantBackground*>(pobj);
    //ASSERT(obj != 0)
  }
}

void Pio(SCIRun::Piostream& stream, rtrt::AmbientBackground*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::AmbientBackground::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::AmbientBackground*>(pobj);
    //ASSERT(obj != 0)
  }
}

void Pio(SCIRun::Piostream& stream, rtrt::LinearBackground*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::LinearBackground::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::LinearBackground*>(pobj);
    //ASSERT(obj != 0)
  }
}

void Pio(SCIRun::Piostream& stream, rtrt::EnvironmentMapBackground*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::EnvironmentMapBackground::type_id);
  if(stream.reading()) {
    obj=dynamic_cast<rtrt::EnvironmentMapBackground*>(pobj);
    //ASSERT(obj != 0)
  }
}

} // end namespace SCIRun
