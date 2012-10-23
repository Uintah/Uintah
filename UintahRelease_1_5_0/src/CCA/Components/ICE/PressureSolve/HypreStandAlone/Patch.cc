/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include "Patch.h"
#include "Level.h"
#include "util.h"
#include <string>
#include <map>

using namespace std;

Patch::Patch(const int procID, 
             const Counter levelID,
             const Box& box) :
  _box(box)
{
  funcPrint("Patch::Patch()",FBegin);

  const Counter numDims = box.getNumDims();
  _procID = procID;
  _levelID = levelID; 
  _boundaries.resize(0,2*numDims);
  _bc.resize(0,2*numDims);
  Vector<int> sz(0,numDims);
  _numCells = box.volume();
  _patchID = 987654; // Dummy value to indicate something is wrong
  dbg.setLevel(10);
  dbg << "_numDims  = " << numDims << "\n"
      << "_numCells = " << _numCells << "\n"
      << "_box      = " << _box << "\n"
      << "box       = " << box << "\n";
  funcPrint("Patch::Patch()",FEnd);
}

Patch::Patch(const Patch& other) :
  _procID(other._procID),
  _levelID(other._levelID),
  _box(other._box),
  _numCells(other._numCells),
  _patchID(other._patchID),
  _boundaries(other._boundaries),
  _bc(other._bc)
{
}

Patch&
Patch::operator = (const Patch& other)
{
  _procID = other._procID;
  _levelID = other._levelID;
  _box = other._box;
  _numCells = other._numCells;
  _patchID = other._patchID;
  _boundaries = other._boundaries;
  _bc = other._bc;
  return *this;
}

Patch::BoundaryType
Patch::getBoundaryType(const Counter d,
                       const Side s) const
{
  return _boundaries[2*d + ((s+1)/2)];
}

Patch::BoundaryCondition
Patch::getBC(const Counter d,
             const Side s) const
{
  return _bc[2*d + ((s+1)/2)];
}

void
Patch::setAllBoundaries(const Vector<BoundaryType>& boundaries)
{
  _boundaries = boundaries;
}

void
Patch::setBoundaryType(const Counter d,
                       const Side s,
                       const BoundaryType& bt) 
{
  _boundaries[2*d+((s+1)/2)] = bt;
}

void
Patch::setAllBC(const Vector<BoundaryCondition>& bc)
{
  _bc = bc;
}

void
Patch::setBC(const Counter d,
             const Side s,
             const BoundaryCondition& bc) 
{
  _bc[2*d+((s+1)/2)] = bc;
}

void
Patch::setDomainBoundaries(const Level& lev)
{
  // Figure out whether you are next to the domain boundary and set
  // boundary condition there.
  // Hardcoded to one domain box of size [0,0] to [resolution].
  // TODO: L-shaped domain with its own makeHierarchy() function of
  // patches. Right now hard-coded to two levels and 2^d processors.
  const Counter numDims = _box.getNumDims();
  Box domainBox(Vector<int>(0,numDims,0,"",0),lev._resolution - 1);
  for (Side s = Left; s <= Right; ++s) {
    const Vector<int>& corner = _box.get(s);
    const Vector<int>& domainCorner = domainBox.get(s);
    for (Counter d = 0; d < numDims; d++) {
      if (corner[d] == domainCorner[d]) {
        setBoundaryType(d,s,Patch::Domain);
        setBC(d,s,Patch::Dirichlet); // Hard coded to Dirichlet B.C.
      } else {
        setBoundaryType(d,s,Patch::CoarseFine);
        setBC(d,s,Patch::NA);
      }
    } // end for d
  } // end for s
}

std::ostream&
operator << (std::ostream& os, const Patch& patch)
  // Write the Patch to the output stream os.
{
  os << "Patch ID = " << patch._patchID
     << ", owned by proc " << patch._procID << "\n";
  os << patch._box;
  os << "\n";
  const Counter numDims = patch._box.getNumDims();
  for (Counter d = 0; d < numDims; d++) {
    for (Side s = Left; s <= Right; ++s) {
      os << "BOUNDARY(d = " << d
         << " , s = " << s << "): "
         << "boundary type = "
         << patch.getBoundaryType(d,s)
         << "  bc = "
         << patch.getBC(d,s)
         << "\n";
    }
  }
  return os;
}

std::ostream&
operator << (std::ostream& os, const Patch::BoundaryType& b)
{
  if      (b == Patch::Domain    ) os << "Domain    ";
  else if (b == Patch::CoarseFine) os << "CoarseFine";
  else if (b == Patch::Neighbor  ) os << "Neighbor  ";
  else os << "BT WRONG!!!";
  return os;
}

std::ostream&
operator << (std::ostream& os, const Patch::BoundaryCondition& c)
{
  if      (c == Patch::NA       ) os << "NA         ";
  else if (c == Patch::Dirichlet) os << "Dirichlet  ";
  else if (c == Patch::Neumann  ) os << "Neumann    ";
  else os << "BC WRONG!!!";
  return os;
}


