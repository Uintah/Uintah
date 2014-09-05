/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef PARTICLESUBSET_H
#define PARTICLESUBSET_H

#include <Core/Util/RefCounted.h>
#include <Core/Grid/Ghost.h>
#include <Core/Geometry/IntVector.h>

#include <vector>
#include <iostream>


using std::ostream;
using SCIRun::IntVector;

#include <Core/Grid/uintahshare.h>
namespace Uintah {
  typedef int particleIndex;
  typedef int particleId;
  class Patch;
/**************************************

CLASS
   ParticleSubset
   
   Short description...

GENERAL INFORMATION

   ParticleSubset.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Particle, ParticleSubset

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class ParticleVariableBase;

  class UINTAHSHARE ParticleSubset : public RefCounted {
  public:
    ParticleSubset(int num_particles, int matlIndex, const Patch*);
    ParticleSubset(int num_particles, int matlIndex, const Patch*,
                   IntVector low, IntVector high);
    ParticleSubset(int num_particles, int matlIndex, const Patch*,
		   IntVector low, IntVector high,
                   const std::vector<const Patch*>& neighbors,
		   const std::vector<ParticleSubset*>& subsets);
    ParticleSubset();
    ~ParticleSubset();
    
    //////////
    // Insert Documentation Here:
    bool operator==(const ParticleSubset& ps) const {
      return d_numParticles == ps.d_numParticles && 
        // a null patch means that there is no patch center for the pset
        // (probably on an AMR copy data timestep)
        (!d_patch || !ps.d_patch || d_patch == ps.d_patch) && 
        d_matlIndex == ps.d_matlIndex && d_low == ps.d_low && d_high == ps.d_high;
    }
      
    //////////
    // Insert Documentation Here:
    void addParticle(particleIndex idx) {
      if(d_numParticles >= d_allocatedSize)
	expand(1);
      d_particles[d_numParticles++] = idx;
    }
    particleIndex addParticles(particleIndex count);

    void resize(particleIndex idx);

    typedef particleIndex* iterator;
      
    //////////
    // Insert Documentation Here:
    iterator begin() {
      return d_particles;
    }
      
    //////////
    // Insert Documentation Here:

    iterator end() {
      return d_particles+d_numParticles;
    }
      
    //////////
    // Insert Documentation Here:
    //    const particleIndex* getPointer() const
    //    {
    //      return d_particles;
    //    }
      
    particleIndex* getPointer()
    {
      return d_particles;
    }
      
    //////////
    // Insert Documentation Here:
    particleIndex numParticles() {
      return d_numParticles;
    }
      
    //////////
    // Insert Documentation Here:
    void set(particleIndex idx, particleIndex value) {
      d_particles[idx] = value;
    }

    IntVector getLow() const {
      return d_low;
    }
    IntVector getHigh() const {
      return d_high;
    }
    const Patch* getPatch() const {
      return d_patch;
    }
    int getMatlIndex() const {
      return d_matlIndex;
    }

    void expand(particleIndex minSizeIncrement);

    // sort the set by particle IDs
    void sort(ParticleVariableBase* particleIDs);

    const std::vector<const Patch*>& getNeighbors() const {
      return neighbors;
    }
    const std::vector<ParticleSubset*>& getNeighborSubsets() const {
      return neighbor_subsets;
    }
    
    UINTAHSHARE friend ostream& operator<<(ostream& out, Uintah::ParticleSubset& pset);

   private:
    //////////
    // Insert Documentation Here:
    particleIndex* d_particles;
    particleIndex d_numParticles;
    particleIndex d_allocatedSize;
    int d_numExpansions;

    int d_matlIndex;
    const Patch* d_patch;
    IntVector d_low, d_high;

    std::vector<const Patch*> neighbors;
    std::vector<ParticleSubset*> neighbor_subsets;

    void fillset();

    void init();
    ParticleSubset(const ParticleSubset& copy);
    ParticleSubset& operator=(const ParticleSubset&);
  };
} // End namespace Uintah

#endif
