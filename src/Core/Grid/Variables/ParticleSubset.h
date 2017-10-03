/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#ifndef PARTICLESUBSET_H
#define PARTICLESUBSET_H

#include <Core/Util/RefCounted.h>
#include <Core/Geometry/IntVector.h>

#include <vector>
#include <iostream>

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
  

KEYWORDS
   Particle, ParticleSubset

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class ParticleVariableBase;

  class ParticleSubset : public RefCounted {
  public:
    ParticleSubset( const unsigned int   num_particles,
                    const int            matlIndex,
                    const Patch        * patch );

    ParticleSubset( const unsigned int        num_particles,
                    const int                 matlIndex,
                    const Patch             * patch,
                    const Uintah::IntVector & low,
                    const Uintah::IntVector & high);

    ParticleSubset( const unsigned int                   num_particles,
                    const int                            matlIndex,
                    const Patch                        * patch,
                    const Uintah::IntVector            & low,
                    const Uintah::IntVector            & high,
                    const std::vector<const Patch*>    & neighbors,
                    const std::vector<ParticleSubset*> & subsets);
    ParticleSubset();
    ~ParticleSubset();
    

    bool operator==(const ParticleSubset& ps) const {
    
    
      // a null patch means that there is no patch center for the pset
      // (probably on an AMR copy data timestep)
      bool A = ( !d_patch || !ps.d_patch || d_patch == ps.d_patch );
      bool B = ( d_numParticles == ps.d_numParticles );
      bool C = ( d_matlIndex == ps.d_matlIndex );
      bool D = ( d_low == ps.d_low && d_high == ps.d_high );
      
      return A && B && C && D;
    }
      
    void addParticle( particleIndex idx ) {
      if( d_numParticles >= d_allocatedSize ){
        expand( 1 );
      }
      
      d_particles[d_numParticles++] = idx;
    }
    
    particleIndex addParticles( unsigned int count );

    void resize(particleIndex idx);

    typedef particleIndex* iterator;

    iterator begin() {
      return d_particles;
    }

    iterator end() {
      return d_particles+d_numParticles;
    }
      
    particleIndex* getPointer()
    {
      return d_particles;
    }
      
    unsigned int numParticles() const {
      return d_numParticles;
    }

    void set(particleIndex idx, particleIndex value) {
      d_particles[idx] = value;
    }

    void setLow(const Uintah::IntVector low) {
      d_low=low;
    }
    void setHigh(const Uintah::IntVector high) {
      d_high=high;
    }

    Uintah::IntVector getLow() const {
      return d_low;
    }
    Uintah::IntVector getHigh() const {
      return d_high;
    }
    const Patch* getPatch() const {
      return d_patch;
    }
    int getMatlIndex() const {
      return d_matlIndex;
    }

    void expand( unsigned int minSizeIncrement );

    // sort the set by particle IDs
    void sort(ParticleVariableBase* particleIDs);

    const std::vector<const Patch*>& getNeighbors() const {
      return neighbors;
    }
    const std::vector<ParticleSubset*>& getNeighborSubsets() const {
      return neighbor_subsets;
    }
    
    friend std::ostream& operator<<(std::ostream& out, Uintah::ParticleSubset& pset);
  //__________________________________
  //
   private:
    particleIndex * d_particles;
    unsigned int    d_numParticles;
    unsigned int    d_allocatedSize;
    int             d_numExpansions;

    int                 d_matlIndex;
    const Patch       * d_patch;
    Uintah::IntVector   d_low, d_high;

    std::vector<const Patch*>    neighbors;
    std::vector<ParticleSubset*> neighbor_subsets;

    void fillset();

    void init();

    ParticleSubset( const ParticleSubset & copy );
    ParticleSubset& operator=( const ParticleSubset & );
  };
} // End namespace Uintah

#endif
