#include "Box.h"
//#include "Level.h"
#include "util.h"

using namespace std;

Box::Box(const Counter numDims)
{
  _lower.resize(0,numDims);
  _upper.resize(0,numDims);
}


Box::Box(const Vector<int>& lower,
         const Vector<int>& upper) :
  _lower(lower), _upper(upper)
{
  assert(_lower.getLen() == _upper.getLen());
  for (Counter d = 0; d < getNumDims(); d++) {
    assert(_lower[d] <= _upper[d]);
  }
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

const Vector<int>&
Box::get(const Side& s) const
{
  if      (s == Left ) return _lower;
  else if (s == Right) return _upper;
  else {
    fprintf(stderr,"\n\nError: Box::get() with s == NA\n");
    clean();
    exit(1);
  }
}

Vector<int>&
Box::get(const Side& s)
{
  if      (s == Left ) return _lower;
  else if (s == Right) return _upper;
  else {
    fprintf(stderr,"\n\nError: Box::get() with s == NA\n");
    clean();
    exit(1);
  }
}

void
Box::set(const Side& s,
         const Vector<int>& value)
{
  if      (s == Left ) _lower = value;
  else if (s == Right) _upper = value;
  else {
    fprintf(stderr,"\n\nError: Box::set() with s == NA\n");
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
    fprintf(stderr,"\n\nError: Box::set() with s == NA\n");
    clean();
    exit(1);
  }
}

Vector<int>
Box::size(void) const
{
  const Counter numDims = _lower.getLen();
  Vector<int> sz(0,numDims);
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
  /* Print the box to output stream os. */
{
  os << "Box extents: from " 
     << a.get(Left)
     << " to "
     << a.get(Right);
  return os;
}

Box
Box::faceExtents(const Counter d,
                 const Side& s)
  /*_____________________________________________________________________
    Function faceExtents:
    Compute face box extents of the numDims-dimensional box *this.
    are ilower,iupper. This is the face in the d-dimension; s = Left
    means the left face, s = Right the right face (so d=1, s=Left is the
    x-left face). Face extents are returned as a box.
    _____________________________________________________________________*/
{
  Box face = *this;
  face.set(Side(-s),face.get(s));
#if DRIVER_DEBUG
  Print("Face(d = %c, s = %s) ",
        d+'x',(s == Left) ? "Left" : "Right");
  cout << face << "\n";
#endif
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
  const Counter numDims = getNumDims();
  Box coarseNbhr(numDims);
  for (Counter dim = 0; dim < numDims; dim++) {
    Print("dim = %d\n",dim);
    Print("refRat = %d\n",refRat[dim]);
    for (Side s = Left; s <= Right; ++s) {
      coarseNbhr.set(dim,s,get(s)[dim]/refRat[dim]);
    }
  }
  Print("# fine   cell faces = %d\n",volume());
  Print("# coarse cell faces = %d\n",coarseNbhr.volume());

  coarseNbhr.get(Left)[d] += s;
  coarseNbhr.get(Right)[d] += s;

  return coarseNbhr;
}

void
Box::iterator::operator ++ (void)
  /*_____________________________________________________________________
    Function Box::iterator::operator++()
    Increment the d-dimensional subscript sub. This is useful when looping
    over a volume or an area. active is a d- boolean array. Indices with
    active=false are kept fixed, and those with active=true are updated.
    ilower,iupper specify the extents of the hypercube we loop over.
    eof is returned as 1 if we're at the end of the cube (the value of sub
    is set to ilower for active=1 indices; i.e. we do a periodic looping)
    and 0 otherwise.
    E.g., incrementing sub=(2,0,1) with active=(0,1,0), ilower=(0,0,0)
    and iupper=(2,2,2) results in sub=(0,0,2) and eof=1. If sub were
    (2,0,2) then sub=(0,0,0) and eof=1.
    _____________________________________________________________________*/
{
  Counter numDims = _box.getNumDims();
  const Vector<int>& lower = _box.get(Left);
  const Vector<int>& upper = _box.get(Right);
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

static void instantiate(void)
{
  Vector<int> lower(0,2,0,"",0);
  Vector<int> upper(0,2,0,"",3);
  Box box(lower,upper);
  Box box2(box);
  Box::iterator iter = box.begin();
  ++iter;
  instantiate();
}
