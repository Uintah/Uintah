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

#ifndef CCA_COMPONENTS_SCHEDULERS_SENDSTATE_H
#define CCA_COMPONENTS_SCHEDULERS_SENDSTATE_H

#include <Core/Grid/Ghost.h>
#include <Core/Grid/Variables/PSPatchMatlGhostRange.h>

#include <map>

namespace Uintah {

class Patch;
class ParticleSubset;

/**************************************

 CLASS
 SendState

 GENERAL INFORMATION

 SendState.h

 Steven G. Parker
 Department of Computer Science
 University of Utah

 Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


 KEYWORDS
 SendState

 DESCRIPTION

 ****************************************/

class SendState {

public:

  SendState()   = default;

  ~SendState();

  ParticleSubset* find_sendset(       int         dest
                              , const Patch     * patch
                              ,       int         matl
                              ,       IntVector   low
                              ,       IntVector   high
                              ,       int         dwid = 0
                              ) const;

  void add_sendset(       ParticleSubset * pset
                  ,       int              dest
                  , const Patch          * patch
                  ,       int              matl
                  ,       IntVector        low
                  ,       IntVector        high
                  ,       int              dwid = 0
                  );

  // Clears out all sendsets..
  void reset();

  void print();


private:

  // disable copy, assignment, and move
  SendState( const SendState & )            = delete;
  SendState& operator=( const SendState & ) = delete;
  SendState( SendState && )                 = delete;
  SendState& operator=( SendState && )      = delete;

  using map_type = std::map<std::pair<PSPatchMatlGhostRange, int>, ParticleSubset*>;
  map_type sendSubsets;

};

}  // End namespace Uintah

#endif // CCA_COMPONENTS_SCHEDULERS_SENDSTATE_H

