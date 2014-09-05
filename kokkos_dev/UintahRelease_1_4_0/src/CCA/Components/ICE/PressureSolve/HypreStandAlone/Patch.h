/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


#ifndef __PATCH_H__
#define __PATCH_H__

#include "Box.h"
#include <map>

/* Forward declarations */
class Level;

class Patch {
  /*_____________________________________________________________________
    class Patch:
    A box of data at a certain level. A processor may own more than one
    patch. A patch can be owned by one proc only.
    _____________________________________________________________________*/

public:

  enum BoundaryType {
    Domain = 0, 
    CoarseFine, 
    Neighbor
  };
  
  enum BoundaryCondition {
    NA = -1,  // Not applicable
    Dirichlet = 0,
    Neumann
  };
  
  Patch(const int procID, 
        const Counter levelID,
        const Box& box);
  Patch(const Patch& other);
  Patch& operator = (const Patch& other);

  /* Boundary conditions get & set */
  BoundaryType getBoundaryType(const Counter d,
                               const Side s) const;
  BoundaryCondition getBC(const Counter d,
                          const Side s) const;
  void setAllBoundaries(const Vector<BoundaryType>& boundaries);
  void setBoundaryType(const Counter d,
                       const Side s,
                       const BoundaryType& bt);
  void setAllBC(const Vector<BoundaryCondition>& bc);
  void setBC(const Counter d,
             const Side s,
             const BoundaryCondition& bc);
  void setDomainBoundaries(const Level& lev);

  int         _procID;    // Owning processor's ID
  Counter     _levelID;   // Which level this Patch belongs to
  Box         _box;       // Patch box extents
  Counter     _numCells;  // Total # cells
  Counter     _patchID;   // Patch ID, unique across all levels & procs
 protected:
  Vector<BoundaryType>      _boundaries;
  Vector<BoundaryCondition> _bc;
  
 private:
};

std::ostream&
operator << (std::ostream& os, const Patch& patch);
std::ostream&
operator << (std::ostream& os, const Patch::BoundaryCondition& c);
std::ostream&
operator << (std::ostream& os, const Patch::BoundaryType& b);

#endif // __PATCH_H__
