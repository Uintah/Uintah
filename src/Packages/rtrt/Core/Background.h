#ifndef BACKGROUND_H
#define BACKGROUND_H 1

#include <Packages/rtrt/Core/Color.h>
#include <Core/Geometry/Vector.h>
#include <Packages/rtrt/Core/Array2.h>
#include <Packages/rtrt/Core/ppm.h>

namespace rtrt {

using SCIRun::Vector;
using SCIRun::Dot;

class Background {
    Color avg;
public:
    Background(const Color& avg);
    virtual ~Background();


    // expects a unit vector
    virtual void color_in_direction( const Vector& v, Color& c) const = 0;

    // gives some approximate value
  inline const Color& average( ) const {
    return avg;
  }

};

class ConstantBackground : public Background {
protected:
    Color C;
public:
    ConstantBackground(const Color& C);
    virtual ~ConstantBackground();

    virtual void color_in_direction(const Vector& v, Color& c) const; 
};


class LinearBackground : public Background {
protected:
    Color C1;
    Color C2;
    Vector direction_to_C1;
public:
    LinearBackground( const Color& C1, const Color& C2,  const Vector& direction_to_C1);

    virtual ~LinearBackground();   
    
    virtual void color_in_direction(const Vector& v, Color& c) const ;

};

  class EnvironmentMapBackground : public Background 
  {
    
  public:

    EnvironmentMapBackground( char* filename, const Vector& up = Vector( 0.0, 0.0, 1.0 ) );
      
      virtual ~EnvironmentMapBackground( void );
      
      virtual void color_in_direction(const Vector& v, Color& c) const ;
      
  protected:
      
      void read_image( char* filename );
      inline Vector ChangeFromBasis( const Vector& a ) const 
      {
	  return( a.x()*_u + a.y()*_v + a.z()*_up );
      }

      inline Vector ChangeToBasis( const Vector& a ) const 
      {
	  return Vector( Dot( a, _u), Dot( a, _v ), Dot( a, _up ) );
      }
      
      texture* _text;
      int _width, _height;
      Array2<Color> _image;
      double _aspectRatio;
      Vector _up;
      Vector _u;
      Vector _v;

};


} // end namespace rtrt

#endif

