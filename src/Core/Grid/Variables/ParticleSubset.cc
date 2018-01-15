/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <Core/Grid/Variables/ParticleSubset.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>

using namespace Uintah;
using namespace std;

ParticleSubset::ParticleSubset() : d_numParticles(0)
{
  init();
}

//______________________________________________________________________
//
ParticleSubset::~ParticleSubset()
{
  for( unsigned int i = 0; i < neighbor_subsets.size(); i++ ) {
    if( neighbor_subsets[i]->removeReference() ) {
      delete neighbor_subsets[i];
    }
  }

  if( d_particles ) {
    delete [] d_particles;
  }
}


//______________________________________________________________________
//
ParticleSubset::ParticleSubset( const unsigned int   num_particles,
                                const int            matlIndex, 
                                const Patch        * patch)
    : d_numParticles(num_particles), d_matlIndex(matlIndex), d_patch(patch)
{
  init();

  if (patch) {
    d_low  = patch->getExtraCellLowIndex();
    d_high = patch->getExtraCellHighIndex();
  }
  else {
    // don't matter...
    d_low  = IntVector(0,0,0);
    d_high = IntVector(0,0,0);
  }
  fillset();
}

//______________________________________________________________________
//
ParticleSubset::ParticleSubset( const unsigned int              num_particles,
                                const int                       matlIndex,
                                const Patch                   * patch,
                                const IntVector               & low,
                                const IntVector               & high,
                                const vector<const Patch*>    & neighbors,
                                const vector<ParticleSubset*> & neighbor_subsets )
  : d_numParticles(num_particles), d_matlIndex(matlIndex), d_patch(patch),
    d_low(low), d_high(high),
    neighbors(neighbors), neighbor_subsets(neighbor_subsets)
{
  init();
  
  for(int i=0;i<(int)neighbor_subsets.size();i++){
    neighbor_subsets[i]->addReference();
  }
  
  fillset();
}

//______________________________________________________________________
//
ParticleSubset::ParticleSubset(       unsigned int   num_particles,
                                      int            matlIndex, 
                                const Patch        * patch,
                                const IntVector    & low,
                                const IntVector    & high )
  : d_numParticles(num_particles), d_matlIndex(matlIndex), d_patch(patch),
    d_low(low), d_high(high)
{
  init();
  
  for(int i=0;i<(int)neighbor_subsets.size();i++){
    neighbor_subsets[i]->addReference();
  }
  
  fillset();
}

//______________________________________________________________________
//
void
ParticleSubset::fillset()
{
  if (d_numParticles > 0) {
    d_particles = scinew particleIndex[d_numParticles];
    
    for( unsigned int i = 0; i < d_numParticles; i++ ) {
      d_particles[i] = i;
    }
    
    d_allocatedSize = d_numParticles;
  }
}

//______________________________________________________________________
//
class compareIDFunctor
{
public:
  compareIDFunctor(ParticleVariable<long64>* particleIDs)
    : particleIDs_(particleIDs) {}
  
  bool operator()(particleIndex x, particleIndex y)
  {
    return (*particleIDs_)[x] < (*particleIDs_)[y];
  }

private:
  ParticleVariable<long64>* particleIDs_;
};

//______________________________________________________________________
//
void
ParticleSubset::sort(ParticleVariableBase* particleIDs)
{
  ParticleVariable<long64>* pIDs = dynamic_cast<ParticleVariable<long64>*>(particleIDs);
  
  if (pIDs == 0){
    SCI_THROW(InternalError("particleID variable must be ParticleVariable<long64>", __FILE__, __LINE__));
  }

  compareIDFunctor comp(pIDs);
  std::sort(d_particles, d_particles+d_numParticles, comp);
}

//______________________________________________________________________
//
void
ParticleSubset::init()
{
  d_particles = 0;
  d_allocatedSize = 0;
  d_numExpansions = 0;
}

//______________________________________________________________________
//
void
ParticleSubset::resize(particleIndex newSize)
{
  // Check for spurious resizes
  if(d_particles) {
    SCI_THROW(InternalError("ParticleSubsets should not be resized after creation", __FILE__, __LINE__));
  }

  d_allocatedSize = newSize;
  d_numParticles  = newSize;

  d_particles = scinew particleIndex[ newSize ];
}

//______________________________________________________________________
//
void
ParticleSubset::expand( unsigned int minSizeIncrement )
{
  unsigned int minAmount = d_numParticles >> 2;
  if( minAmount < 10 ) {
    minAmount = 10;
  }
  
  if( minSizeIncrement < minAmount ) {
    minSizeIncrement = minAmount;
  }
  d_allocatedSize += minSizeIncrement;
#if 0
  if(d_numExpansions++ > 18){
    static ProgressiveWarning warn("Performance warning in ParticleSubset",10);
    warn.changeMessage("Performance warning in ParticleSubset, more than 18 expansions occured");
    warn.invoke();
  }
#endif
  particleIndex* newparticles = scinew particleIndex[d_allocatedSize];
  if(d_particles){
    for(unsigned int i = 0; i < d_numParticles; i++ ) {
      newparticles[i] = d_particles[i];
    }
    delete [] d_particles;
  }
  d_particles = newparticles;
}

//______________________________________________________________________
//
particleIndex
ParticleSubset::addParticles( unsigned int count )
{
  if( d_numParticles + count > d_allocatedSize ) {
    expand( count );
  }

  unsigned int oldsize = d_numParticles;
  d_numParticles += count;

  for( unsigned int idx = oldsize; idx < d_numParticles; idx++) {
    d_particles[idx] = idx;
  }
  return oldsize;  // The beginning of the new index range
}

//______________________________________________________________________
//
namespace Uintah {

ostream &
operator<<(ostream& out, ParticleSubset& pset)
{
  const Patch* patch = pset.getPatch();
  out << "pset L-" << patch->getLevel()->getIndex() << ", "
      << *(patch) 
      << " (" << (patch?patch->getID():0)
      << "), matl: "
      << pset.getMatlIndex() << ", pset range [" << pset.getLow() << ", " << pset.getHigh() << "]" 
      << ", particles: " << pset.numParticles()
      << ", neighboring patches: "<< pset.getNeighbors().size() ;
  return out;
}

} // end namespace Uintah
