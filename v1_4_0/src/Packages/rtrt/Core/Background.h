#ifndef BACKGROUND_H
#define BACKGROUND_H 1

#include <Packages/rtrt/Core/Color.h>
#include <Core/Geometry/Vector.h>

namespace rtrt {

  using namespace SCIRun;
  
class Background {
    Color avg;
public:
    Background(const Color& avg);
    virtual ~Background();


    // expects a unit vector
    virtual Color color_in_direction( const Vector& v) const = 0;

    // gives some approximate value
    inline const Color& average( ) const;

};

const Color& Background::average() const {
    return avg;
}


class ConstantBackground : public Background {
protected:
    Color C;
public:
    ConstantBackground(const Color& C);
    virtual ~ConstantBackground();

    virtual Color color_in_direction(const Vector& v) const; 
};


class LinearBackground : public Background {
protected:
    Color C1;
    Color C2;
    Vector direction_to_C1;
public:
    LinearBackground( const Color& C1, const Color& C2,  const Vector& direction_to_C1);

    virtual ~LinearBackground();   
    
    virtual Color color_in_direction(const Vector& v) const ;

};



} // end namespace rtrt

#endif
