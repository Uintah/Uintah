#ifndef BACKGROUND_H
#define BACKGROUND_H 1

///////////////////////////////////////////////////////////////////
// Background.h
//
// Provides a common interface for the background color.  We would like
// the background to do more than just pass a color back, but we want a
// common interface for each.
//
// There used to be a notion of the background color changing with the
// ambient level, but I find this to be very annoying (the background color
// should be the background color, regardless of the ambient level).
// I've added a new class called AmbientBackground which takes into account
// the ambient levels.  Constant and Linear both don't take into account
// the current abmbient levels, however, Ambient and EnvironmentMap do.
//   -- James

#include <Packages/rtrt/Core/Color.h>
#include <Core/Geometry/Vector.h>
#include <Core/Persistent/Persistent.h>
#include <Packages/rtrt/Core/Array2.h>
#include <Packages/rtrt/Core/ppm.h>

namespace rtrt {
class Background;
class ConstantBackground;
class AmbientBackground;
class LinearBackground;
class EnvironmentMapBackground;
}

namespace SCIRun {
void Pio(Piostream&, rtrt::Background*&);
void Pio(Piostream&, rtrt::ConstantBackground*&);
void Pio(Piostream&, rtrt::AmbientBackground*&);
void Pio(Piostream&, rtrt::LinearBackground*&);
void Pio(Piostream&, rtrt::EnvironmentMapBackground*&);
}

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Dot;

class Background : public virtual SCIRun::Persistent {
public:
  Background() {}
  virtual ~Background();

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, Background*&);

  // expects a unit vector
  virtual void color_in_direction( const Vector& v, Color& c) const = 0;

  virtual void updateAmbient( double /*scale*/ ) = 0;
};

class ConstantBackground : public Background {
protected:
  Color C;
public:
  ConstantBackground(const Color& C);
  virtual ~ConstantBackground();

  ConstantBackground() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, ConstantBackground*&);

  virtual void color_in_direction(const Vector& v, Color& c) const; 
  virtual void updateAmbient( double ) {}
};


class AmbientBackground : public Background {
protected:
  Color C;
  Color origC_; // This multiplied by the ambient scale is used for C.
public:
  AmbientBackground(const Color& C);
  virtual ~AmbientBackground();

  AmbientBackground() {} // for Pio.

  //! Persistent I/O.
  static  SCIRun::PersistentTypeID type_id;
  virtual void io(SCIRun::Piostream &stream);
  friend void SCIRun::Pio(SCIRun::Piostream&, AmbientBackground*&);

  virtual void color_in_direction(const Vector& v, Color& c) const; 
  virtual void updateAmbient( double scale ) {
    C = origC_ * scale;
  }
};


class LinearBackground : public Background {
protected:
  Color C1;
  Color C2;
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

  virtual void updateAmbient( double ) {}
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
