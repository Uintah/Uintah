#ifndef Uintah_Core_Grid_ParticleVariable_special_cc
#define Uintah_Core_Grid_ParticleVariable_special_cc

#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

  template<>
  void
  ParticleVariable<Point>::packMPI(void* buf, int bufsize, int* bufpos,
				   const ProcessorGroup* pg,
				   ParticleSubset* pset, const Patch* forPatch)
  {
    // This should be fixed for variable sized types!
    if (!forPatch->isVirtual()) {
      packMPI(buf, bufsize, bufpos, pg, pset);
    }
    else {
      Vector offset = forPatch->getVirtualOffsetVector();
      const TypeDescription* td = getTypeDescription()->getSubType();
      if(td->isFlat()){
	for(ParticleSubset::iterator iter = pset->begin();
	    iter != pset->end(); iter++){
	  Point p = d_pdata->data[*iter] - offset;
	  MPI_Pack(&p, 1, td->getMPIType(), buf, bufsize, bufpos, pg->getComm());
	}
      } else {
	SCI_THROW(InternalError("packMPI not finished\n"));
      }
    }
  }

  // specialization for T=Point
  template <>
  void ParticleVariable<Point>::gather(ParticleSubset* pset,
				       vector<ParticleSubset*> subsets,
				       vector<ParticleVariableBase*> srcs,
				       const vector<const Patch*>& srcPatches,
				       particleIndex extra)
  {
    if(d_pdata && d_pdata->removeReference())
      delete d_pdata;
    if(d_pset && d_pset->removeReference())
      delete d_pset;

#if SCI_ASSERTION_LEVEL >= 2
    IntVector lowIndex, highIndex;
    pset->getPatch()->
      computeVariableExtents(Patch::CellBased, IntVector(0,0,0),
			     pset->getGhostType(), pset->numGhostCells(),
			     lowIndex, highIndex);
    Box box = pset->getPatch()->getLevel()->getBox(lowIndex, highIndex);
#endif
    
    d_pset = pset;
    pset->addReference();
    d_pdata=scinew ParticleData<Point>(pset->getParticleSet()->numParticles());
    d_pdata->addReference();
    ASSERTEQ(subsets.size(), srcs.size());
    ParticleSubset::iterator dstiter = pset->begin();
    for(int i=0;i<(int)subsets.size();i++){
      ParticleVariable<Point>* srcptr =
	dynamic_cast<ParticleVariable<Point>*>(srcs[i]);
      if(!srcptr)
	SCI_THROW(TypeMismatchException("Type mismatch in ParticleVariable::gather"));
      ParticleVariable<Point>& src = *srcptr;
      ParticleSubset* subset = subsets[i];
      const Patch* srcPatch = srcPatches[i];
      if (srcPatch == 0 || !srcPatch->isVirtual()) {
	for(ParticleSubset::iterator srciter = subset->begin();
	    srciter != subset->end(); srciter++){
	  const Point& p = (*this)[*dstiter] = src[*srciter];
	  ASSERT(box.contains(p));	  
	  dstiter++;
	}
      }
      else if (subset->numParticles() != 0) {
	Vector offset = srcPatch->getVirtualOffsetVector();
	for(ParticleSubset::iterator srciter = subset->begin();
	    srciter != subset->end(); srciter++){
	  (*this)[*dstiter] = src[*srciter] + offset;
	  ASSERT(box.contains((*this)[*dstiter]));
	  dstiter++;
	}
      }
    }
    ASSERT(dstiter+extra == pset->end());    
  }

#endif // this file does need to be included to satisfy template instantiations
       // for some compilers
