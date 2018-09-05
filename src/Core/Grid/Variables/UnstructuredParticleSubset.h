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

#ifndef UNSTRUCTURED_PARTICLESUBSET_H
#define UNSTRUCTURED_PARTICLESUBSET_H

#include <Core/Util/RefCounted.h>
#include <Core/Geometry/IntVector.h>

#include <vector>
#include <iostream>

namespace Uintah {
  typedef int particleIndex;
  typedef int particleId;
  class UnstructuredPatch;
/**************************************




CLASS
   UnstructuredParticleSubset
   
   Short description...

GENERAL INFORMATION

   UnstructuredParticleSubset.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   UnstructuredParticle, UnstructuredParticleSubset

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class UnstructuredParticleVariableBase;

  class UnstructuredParticleSubset : public RefCounted {
  public:
    UnstructuredParticleSubset( const unsigned int   num_particles,
                    const int            matlIndex,
                    const UnstructuredPatch        * patch );

    UnstructuredParticleSubset( const unsigned int        num_particles,
                    const int                 matlIndex,
                    const UnstructuredPatch             * patch,
                    const Uintah::IntVector & low,
                    const Uintah::IntVector & high);

    UnstructuredParticleSubset( const unsigned int                   num_particles,
                    const int                            matlIndex,
                    const UnstructuredPatch                        * patch,
                    const Uintah::IntVector            & low,
                    const Uintah::IntVector            & high,
                    const std::vector<const UnstructuredPatch*>    & neighbors,
                    const std::vector<UnstructuredParticleSubset*> & subsets);
    UnstructuredParticleSubset();
    ~UnstructuredParticleSubset();
    

    bool operator==(const UnstructuredParticleSubset& ps) const {
    
    
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
    const UnstructuredPatch* getPatch() const {
      return d_patch;
    }
    int getMatlIndex() const {
      return d_matlIndex;
    }

    void expand( unsigned int minSizeIncrement );

    // sort the set by particle IDs
    void sort(UnstructuredParticleVariableBase* particleIDs);

    const std::vector<const UnstructuredPatch*>& getNeighbors() const {
      return neighbors;
    }
    const std::vector<UnstructuredParticleSubset*>& getNeighborSubsets() const {
      return neighbor_subsets;
    }
    
    friend std::ostream& operator<<(std::ostream& out, Uintah::UnstructuredParticleSubset& pset);
  //__________________________________
  //
   private:
    particleIndex * d_particles;
    unsigned int    d_numParticles;
    unsigned int    d_allocatedSize;
    int             d_numExpansions;

    int                 d_matlIndex;
    const UnstructuredPatch       * d_patch;
    Uintah::IntVector   d_low, d_high;

    std::vector<const UnstructuredPatch*>    neighbors;
    std::vector<UnstructuredParticleSubset*> neighbor_subsets;

    void fillset();

    void init();

    UnstructuredParticleSubset( const UnstructuredParticleSubset & copy );
    UnstructuredParticleSubset& operator=( const UnstructuredParticleSubset & );
  };
} // End namespace Uintah

#endif
