#ifndef Uintah_MPM_Spline
#define Uintah_MPM_Spline

namespace SCICore {
namespace Geometry {
  class Vector;
}}

namespace Uintah {
namespace MPM {

  using SCICore::Geometry::Vector;

class Spline {
public:
          double             radius;

  virtual double             w(const Vector& r) const = 0;
  virtual double             dwdx(int i,const Vector& r) const = 0;
};

}} //namespace

#endif

// $Log$
// Revision 1.1  2000/07/06 06:30:34  tan
// Added Spline class.
//
