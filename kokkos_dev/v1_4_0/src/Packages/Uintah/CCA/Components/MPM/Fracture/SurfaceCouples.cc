#include "SurfaceCouples.h"

#include <float.h>  // for DBL_MAX

namespace Uintah {
using namespace SCIRun;

SurfaceCouples::SurfaceCouples(const ParticleVariable<Vector>& pCrackNormal,
                               const Lattice& lattice)
: d_pCrackNormal(pCrackNormal),d_lattice(lattice)
{}

void SurfaceCouples::setup()
{
  double relate_cosine = 0.7;
  
  ParticleSubset* pset = getpX().getParticleSubset();

  for(ParticleSubset::iterator iterA = pset->begin();
          iterA != pset->end(); iterA++)
  {
    particleIndex pIdxA = *iterA;
    const Vector& crackNormalA = d_pCrackNormal[pIdxA];
    if(crackNormalA.length2()>0.5) {
      particleIndex match = -1;
      Vector normal = crackNormalA;
      double distance = DBL_MAX;
      for(ParticleSubset::iterator iterB = pset->begin();
            iterB != pset->end(); iterB++)
      {
        particleIndex pIdxB = *iterB;
        if(pIdxA != pIdxB) {
          const Vector& crackNormalB = d_pCrackNormal[pIdxB];
          if( Dot(crackNormalA,crackNormalB) < -relate_cosine ) {
	    normal -= crackNormalB;
	    normal.normalize();
            Vector dis = getpX()[pIdxB] - getpX()[pIdxA];
	    double AB = dis.length();
	    double vAB = Dot(normal,dis);
            if(vAB/AB>relate_cosine) {
	      if(AB<distance) {
	        distance = AB;
		match = pIdxB;
	      }
	    }
	  }
	}
      }
      if(match >= 0) {
        bool newMatch = true;
        int coupleNum = d_couples.size();
        for(int i=0;i<coupleNum;++i) {
	  if( d_couples[i].getIdxA() == match && 
	      d_couples[i].getIdxB() == pIdxA )
	  {
	    newMatch = false;
	    break;
	  }
	}
        if(newMatch) {
	  SurfaceCouple couple(pIdxA,match,normal);
          d_couples.push_back(couple);
	}
      }
    } //if A
  }
}

void SurfaceCouples::build()
{
  double relate_cosine = 0.7;
  
  ParticleSubset* pset = getpX().getParticleSubset();

  for(ParticleSubset::iterator iterA = pset->begin();
          iterA != pset->end(); iterA++)
  {
    particleIndex pIdxA = *iterA;
    const Vector& crackNormalA = d_pCrackNormal[pIdxA];
    if(crackNormalA.length2()>0.5) {
      particleIndex match = -1;
      double distance = DBL_MAX;
      for(ParticleSubset::iterator iterB = pset->begin();
            iterB != pset->end(); iterB++)
      {
        particleIndex pIdxB = *iterB;
        if(pIdxA != pIdxB) {
          Vector dis = getpX()[pIdxB] - getpX()[pIdxA];
	  double AB = dis.length();
	  double vAB = Dot(crackNormalA,dis);
          if(vAB/AB>relate_cosine) {
	    if(AB<distance) {
	      distance = AB;
	      match = pIdxB;
	    }
	  }
	}
      }
      if(match >= 0) {
        bool newMatch = true;
        int coupleNum = d_couples.size();
        for(int i=0;i<coupleNum;++i) {
	  if( d_couples[i].getIdxA() == match && 
	      d_couples[i].getIdxB() == pIdxA )
	  {
	    newMatch = false;
	    break;
	  }
	}
        if(newMatch) {
	  SurfaceCouple couple(pIdxA,match,crackNormalA);
          d_couples.push_back(couple);
	}
      }
    } //if A
  }
}

ostream& operator<<( ostream& os, const SurfaceCouples& couples )
{
  int coupleNum = couples.size();
  for(int i=0;i<coupleNum;++i) {
    os << "surface couple " << i << " : ("
       <<couples.getpX()[couples[i].getIdxA()] <<","
       <<couples.getpX()[couples[i].getIdxB()] <<";"
       <<couples.getpCrackNormal()[couples[i].getIdxA()] <<")"<<endl;
  }
  return os;
}


} // End namespace Uintah
