#include "Background.h"

using namespace rtrt;

Background::Background(const Color& avg)
: avg(avg)
{
}

Background::~Background() {}


//*****************************************************************
//     ConstantBackground members

ConstantBackground::ConstantBackground(const Color& C) : Background(C), C(C) {}

ConstantBackground::~ConstantBackground() {}

Color ConstantBackground::color_in_direction( const Vector& ) const
{
    return C;
}

//*****************************************************************
//     LinearBackground members


LinearBackground::~LinearBackground() {}

LinearBackground::LinearBackground( const Color& C1, const Color& C2,  const Vector& direction_to_C1) :
    Background(C1),
    C1(C1), C2(C2),  direction_to_C1(direction_to_C1) {}

    
Color LinearBackground::color_in_direction(const Vector& v) const {
    double t = 0.5* (1 + v.dot( direction_to_C1 ) );
    return (t)*C1 + (1-t)*C2;
}


