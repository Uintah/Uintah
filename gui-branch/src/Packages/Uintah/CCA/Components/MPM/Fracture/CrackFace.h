#ifndef Uintah_MPM_CrackFace
#define Uintah_MPM_CrackFace

#include <Core/Geometry/Vector.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>

namespace Uintah {

using SCIRun::Vector;
using SCIRun::Point;
class Lattice;

class CrackFace {
public:
                 CrackFace(const Vector& n,const Point& p,double halfGap) :
		   _normal(n),_tip(p),_halfGap(halfGap) {}

  //vector n is a normalized vector here
  void           setup(const Vector& n,const Point& p,double halfGap);
  
  const Point&   getTip() const;
  void           setTip(const Point& p);

  const Vector&  getNormal() const;
  void           setNormal(const Vector& n);

  double         getHalfGap() const;

  double         distance(const Point& p);

  bool           isTip(const ParticleVariable<Vector>& pCrackNormal,
                       const ParticleVariable<int>& pIsBroken,
                       const Lattice& lattice) const;

  bool           atTip(const Point& p) const;

  bool           closeToBoundary(const Point& p,
                      const Lattice& lattice) const;

private:
  Vector      _normal;
  Point       _tip;
  double      _halfGap;
};

} //namespace

#endif
