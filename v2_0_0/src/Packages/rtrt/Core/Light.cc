
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/MusilRNG.h>
#include <Packages/rtrt/Core/LightMaterial.h>
#include <Packages/rtrt/Core/HaloMaterial.h>
#include <math.h>

using namespace rtrt;
using namespace SCIRun;  

Persistent* light_maker() {
  return new Light;
}

// initialize the static member type_id
PersistentTypeID Light::type_id("Light", "Persistent", light_maker);


void make_ortho(const Vector& v, Vector& v1, Vector& v2) 
{
    Vector v0(Cross(v, Vector(1,0,0)));
    if(v0.length2() == 0){
        v0=Cross(v, Cross(v, Vector(0,1,0)));
    }
    v1=Cross(v, v0);
    v1.normalize();
    v2=Cross(v, v1);
    v2.normalize();
}

// returns n^2 unit vectors within a cone with interior angle 2*theta0
// oriented in direction axis (need not be unit).  Assumes incone is
// already allocated
void get_cone_vectors(const Vector& axis, double theta0, int n, Vector* incone)
{
     Vector what = axis;
     what.normalize();
     Vector uhat, vhat;
     make_ortho(what, uhat, vhat);
     double cosTheta0 = cos(theta0);
     int k = 0;
     for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
           double phi = 2 * M_PI * (i + drand48()) / n;
           double cosTheta = 1 - (1 - cosTheta0) * (j + drand48()) / n;
           double sinTheta = sqrt(1 - cosTheta*cosTheta);
           incone[k++] = uhat*cos(phi)*sinTheta +
                         vhat*sin(phi)*sinTheta +
                         what*cosTheta;
        }
     }
}

static Material * flat_yellow   = NULL;
static Material * flat_orange   = NULL;
static Material * flat_white    = NULL;
static HaloMaterial *white_halo = NULL;

void Light::init() {
  Vector d(pos-Point(0,0,.5));
  int n=3;
  beamdirs.resize(n*n);
  get_cone_vectors(d, .03, n, &beamdirs[0]);

  if ( !flat_yellow )
    flat_yellow = new LightMaterial(Color(.8,.8,.0));
  if ( !flat_orange )
    flat_orange = new LightMaterial(Color(1.0,.7,.0));
  if ( !flat_white )
    flat_white = new LightMaterial(Color(1,1,1));
  if ( !white_halo )
    white_halo = new HaloMaterial(flat_white,4);

  if( intensity_ == 0.0 )
    turnOff();

  // Create a yellow sphere that can be rendered in the location
  // of the light.
  sphere_ = new Sphere( white_halo /*flat_yellow*/, pos, 0.1 );
}

Light::Light(const Point& pos,
	     const Color& color,
	     double radius,
	     double intensity, /* = 1.0 */
	     bool   moodLight  /* = false */ ) :
  radius(radius), 
  fixed_to_eye(false),
  eye_offset_basis(0,0,0),
  last_offset(0,0,0),
  intensity_(intensity), 
  origIntensity_(intensity), 
  currentColor_(color*intensity), 
  origColor_(color), 
  pos(pos),
  isOn_(true),
  moodLight_(moodLight)
{
  init();
}

void
Light::modifyCurrentIntensity( double scale )
{
  // Not a perfect implementation, but will work for now.
  double inten = intensity_ * scale;
  updateIntensity( inten );
}

void
Light::updateIntensity( double toIntensity )
{
  if( toIntensity < 0 ) toIntensity = 0;
  else if( toIntensity > 1.0 ) toIntensity = 1.0;

  currentColor_ = origColor_ * toIntensity;
  intensity_ = toIntensity;
}

void
Light::updatePosition( const Point & newPos )
{ 
  pos = newPos+last_offset;
  sphere_->updatePosition( pos );
}

void
Light::updatePosition( const Point & newPos, const Vector &offset, const Vector &/*fwd*/ )
{ 
  last_offset = offset;
  pos = newPos+offset;
  sphere_->updatePosition( pos );
}

const int LIGHT_VERSION = 1;

void 
Light::io(SCIRun::Piostream &str)
{
  str.begin_class("Light", LIGHT_VERSION);
  SCIRun::Pio(str, radius);
  SCIRun::Pio(str, intensity_);
  SCIRun::Pio(str, currentColor_);
  SCIRun::Pio(str, origColor_);
  SCIRun::Pio(str, pos);
  SCIRun::Pio(str, moodLight_);
  SCIRun::Pio(str, isOn_);
  SCIRun::Pio(str, eye_offset_basis);
  SCIRun::Pio(str, last_offset);
  if (str.reading()) {
    init();
  }
  str.end_class();
}

namespace SCIRun {
void Pio(SCIRun::Piostream& stream, rtrt::Light*& obj)
{
  SCIRun::Persistent* pobj=obj;
  stream.io(pobj, rtrt::Light::type_id);
  if(stream.reading())
    obj=(rtrt::Light*)pobj;
}
} // end namespace SCIRun
