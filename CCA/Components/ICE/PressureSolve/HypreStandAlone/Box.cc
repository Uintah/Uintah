/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "Box.h"
//#include "Level.h"
#include "util.h"

using namespace std;

Box::Box(const Counter numDims)
{
  _lower.resize(0,numDims);
  _upper.resize(0,numDims);
}


Box::Box(const Point& lower,
         const Point& upper) :
  _lower(lower), _upper(upper)
{
  // Allow degenerate boxes to be constructed
  //  assert(_lower.getLen() <= _upper.getLen());
}

Box::Box(const Box& other) :
  _lower(other._lower), _upper(other._upper)
{
}

Box&
Box::operator = (const Box& b)
{
  _lower = b._lower;
  _upper = b._upper;
  return *this;
}

const Point&
Box::get(const Side& s) const
{
  if      (s == Left ) return _lower;
  else if (s == Right) return _upper;
  else {
    cerr << "\n\nError: Box::get() with s == NA" << "\n";
    clean();
    exit(1);
  }
}

Point&
Box::get(const Side& s)
{
  if      (s == Left ) return _lower;
  else if (s == Right) return _upper;
  else {
    cerr << "\n\nError: Box::get() with s == NA" << "\n";
    clean();
    exit(1);
  }
}

void
Box::set(const Side& s,
         const Point& value)
{
  if      (s == Left ) _lower = value;
  else if (s == Right) _upper = value;
  else {
    cerr << "\n\nError: Box::set() with s == NA" << "\n";
    clean();
    exit(1);
  }
}

void
Box::set(const Counter d,
         const Side& s,
         const int& value)
{
  if      (s == Left ) _lower[d] = value;
  else if (s == Right) _upper[d] = value;
  else {
    cerr << "\n\nError: Box::set() with s == NA" << "\n";
    clean();
    exit(1);
  }
}

Point
Box::size(void) const
{
  const Counter numDims = _lower.getLen();
  Point sz(0,numDims);
  for (Counter d = 0; d < numDims; d++) {
    sz[d] = _upper[d] - _lower[d] + 1;
  }
  return sz;
}

Counter
Box::volume(void) const
{
  Counter numCells = 1;
  const Counter numDims = _lower.getLen();
  for (Counter d = 0; d < numDims; d++) {
    int temp = _upper[d] - _lower[d] + 1;
    if (temp <= 0) {
      return 0;
    } else {
      numCells *= Counter(temp);
    }
  }
  return numCells;
}

std::ostream&
operator << (std::ostream& os,
             const Box& a)
  // Print the box to output stream os.
{
  os << "Box extents: " 
     << a.get(Left)
     << " .. "
     << a.get(Right);
  return os;
}

Box
Box::faceExtents(const Counter d,
                 const Side& s) const
  /*_____________________________________________________________________
    Function faceExtents:
    Compute face box extents of the numDims-dimensional box *this.
    are lower,upper. This is the face in the d-dimension; s = Left
    means the left face, s = Right the right face (so d=1, s=Left is the
    x-left face). Face extents are returned as a box.
    _____________________________________________________________________*/
{
  Box face = *this;
  face.set(d,Side(-s),face.get(s)[d]);
  return face;
}

Box 
Box::coarseNbhrExtents(const Vector<Counter>& refRat,
                       const Counter d,
                       const Side& s) const
  /*_____________________________________________________________________
    Function coarseNbhrExtents: return the box of the next-coarse-
    level cells neighbouring *this on dimension d and side s. *this
    is normally a face on level k, the returned box on level k-1.
    refRat is the refinement ratio at level k.
    _____________________________________________________________________*/
{
  funcPrint("Box::coarseNbhrExtents",FBegin);
  const Counter numDims = getNumDims();
  Box coarseNbhr(numDims);
  for (Counter dim = 0; dim < numDims; dim++) {
    for (Side s = Left; s <= Right; ++s) {
      coarseNbhr.set(dim,s,get(s)[dim]/refRat[dim]);
    }
  }
  dbg << "# fine   cell faces = " << volume() << "\n";
  dbg << "# coarse cell faces = " << coarseNbhr.volume() << "\n";

  coarseNbhr.get(Left)[d] += s;
  coarseNbhr.get(Right)[d] += s;

  funcPrint("Box::coarseNbhrExtents",FEnd);
  return coarseNbhr;
}

void
Box::iterator::operator ++ (void)
  /*_____________________________________________________________________
    Function Box::iterator::operator++()
    Increment the d-dimensional subscript sub. This is useful when looping
    over a volume or an area. active is a d- boolean array. Indices with
    active=false are kept fixed, and those with active=true are updated.
    lower,upper specify the extents of the hypercube we loop over.
    eof is returned as 1 if we're at the end of the cube (the value of sub
    is set to lower for active=1 indices; i.e. we do a periodic looping)
    and 0 otherwise.
    E.g., incrementing sub=(2,0,1) with active=(0,1,0), lower=(0,0,0)
    and upper=(2,2,2) results in sub=(0,0,2) and eof=1. If sub were
    (2,0,2) then sub=(0,0,0) and eof=1.
    _____________________________________________________________________*/
{
  Counter numDims = _box.getNumDims();
  const Point& lower = _box.get(Left);
  const Point& upper = _box.get(Right);
  //  bool eof = false;

  Counter d = 0;
  _sub[d]++;
  if (_sub[d] > upper[d]) {
    while (_sub[d] > upper[d]) {
      _sub[d] = lower[d];
      d++;
      if (d == numDims) { // end of box reached
        _sub[0] -= 1; // So that iter is now Box::end() that is outside the box
        break;
      }
      _sub[d]++;
    }
  }
}

bool 
Box::overlaps(const Box& otherbox, double epsilon) const
{
  for (Counter d = 0; d < getNumDims(); d++) {
    if((_lower[d]+epsilon > otherbox._upper[d]) ||
       (_upper[d] < otherbox._lower[d]+epsilon)) {
      return false;
    }
  }
  return true;
}

bool 
Box::contains(const Point& p) const {
  return ((_lower <= p) && (p <= _upper));
}

Box 
Box::intersect(const Box& b) const {
  return Box(max(_lower, b._lower),
             min(_upper, b._upper));
}
 
bool
Box::degenerate() const {
  return (!(_lower <= _upper));
}

bool
Box::degenerate(const Counter d) const {
  return (!(_lower[d] <= _upper[d]));
}

#if 0
static void instantiate(void)
{
  Point lower(0,2,0,"",0);
  Point upper(0,2,0,"",3);
  Box box(lower,upper);
  Box box2(box);
  Box::iterator iter = box.begin();
  ++iter;
  instantiate();
}
#endif
