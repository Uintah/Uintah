#ifndef __Uintah_SurfaceCouples__
#define __Uintah_SurfaceCouples__

#include "SurfaceCouple.h"
#include "Lattice.h"
#include <vector>

namespace Uintah {
using namespace SCIRun;

class SurfaceCouples {
public:
  SurfaceCouples(const ParticleVariable<Vector>& pCrackNormal,
                 const Lattice& lattice);
  
  void  setup();
  void  build();
  
  int   size() const
    {
      return d_couples.size();
    }

  SurfaceCouple& operator[](int i)
    {
      return d_couples[i];
    }

  const SurfaceCouple& operator[](int i) const
    {
      return d_couples[i];
    }

  const ParticleVariable<Vector>& getpCrackNormal() const
    {
      return d_pCrackNormal;
    }
    
  const ParticleVariable<Point>& getpX() const
    {
      return d_lattice.getpX();
    }
		 
private:
  const ParticleVariable<Vector>&  d_pCrackNormal;
  const Lattice&                   d_lattice;
  std::vector<SurfaceCouple>       d_couples;
};

ostream& operator<<( ostream& os, const SurfaceCouples& couples );

} // End namespace Uintah

#endif //__Uintah_SurfaceCouples__
