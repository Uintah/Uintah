#ifndef BACKGROUND_H
#define BACKGROUND_H 1

#include <Packages/rtrt/Core/Color.h>
#include <Core/Geometry/Vector.h>
#include <Core/Persistent/Persistent.h>
#include <Packages/rtrt/Core/Array2.h>
#include <Packages/rtrt/Core/ppm.h>

namespace rtrt {
class Background;
class ConstantBackground;
class LinearBackground;
class EnvironmentMapBackground;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::Background*&);
void Pio(Piostream&, rtrt::ConstantBackground*&);
void Pio(Piostream&, rtrt::LinearBackground*&);
void Pio(Piostream&, rtrt::EnvironmentMapBackground*&);
}

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Dot;

class Background : public SCIRun::Persistent {
  Color avg;
  Color origAvg_;
public:
  Background(const Color& avg);
  virtual ~Background();

  Background() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Background*&);

  // expects a unit vector
  virtual void color_in_direction( const Vector& v, Color& c) const = 0;

  // gives some approximate value
  inline const Color& average( ) const {
    return avg;
  }

  virtual void updateAmbient( double scale ) {
    avg = origAvg_ * scale;
  }

};

class ConstantBackground : public Background {
protected:
  Color C;
  Color origC_;
public:
  ConstantBackground(const Color& C);
  virtual ~ConstantBackground();

  ConstantBackground() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, ConstantBackground*&);

  virtual void color_in_direction(const Vector& v, Color& c) const; 
  virtual void updateAmbient( double scale ) {
    C = origC_ * scale;
  }
};


class LinearBackground : public Background {
protected:
  Color C1;
  Color C2;
  Color origC1_;
  Color origC2_;
  Vector direction_to_C1;
public:
  LinearBackground( const Color& C1, const Color& C2,  
		    const Vector& direction_to_C1);

  virtual ~LinearBackground();   
  LinearBackground() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, LinearBackground*&);

  virtual void color_in_direction(const Vector& v, Color& c) const ;

  virtual void updateAmbient( double scale ) {
    C1 = origC1_ * scale;
    C2 = origC2_ * scale;
  }
};

class EnvironmentMapBackground : public Background {
public:
  EnvironmentMapBackground( char* filename, 
			    const Vector& up = Vector( 0.0, 0.0, 1.0 ) );

  virtual ~EnvironmentMapBackground();

  EnvironmentMapBackground() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, EnvironmentMapBackground*&);

  virtual void color_in_direction(const Vector& v, Color& c) const ;

protected:
  inline Vector ChangeFromBasis( const Vector& a ) const 
  {
    return( a.x()*_u + a.y()*_v + a.z()*_up );
  }
  
  inline Vector ChangeToBasis( const Vector& a ) const 
  {
    return Vector( Dot( a, _u), Dot( a, _v ), Dot( a, _up ) );
  }
      
  virtual void updateAmbient( double scale ) {
    ambientScale_ = scale;
  }

  double ambientScale_;
  int _width, _height;
  Array2<Color> _image;
  Vector _up;
  Vector _u;
  Vector _v;
};

} // end namespace rtrt

#endif
