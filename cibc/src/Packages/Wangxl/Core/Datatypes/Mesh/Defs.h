#ifndef SCI_Wangxl_Datatypes_Mesh_Defs_h
#define SCI_Wangxl_Datatypes_Mesh_Defs_h

namespace Wangxl {

  //using namespace SCIRun;

  //enum {VX, VY, VZ, VW};
typedef double (*V_FCT_PTR)(double);
//#define V_ERROR(E) { cerr << E; exit(EXIT_FAILURE); }

enum Locate_type { VERTEX = 0, EDGE, FACET, CELL, OUTSIDE_CONVEX_HULL, OUTSIDE_AFFINE_HULL };

enum Sign { NEGATIVE=-1, ZERO, POSITIVE };
typedef Sign Orientation;

const Orientation LEFTTURN = POSITIVE;
const Orientation RIGHTTURN = NEGATIVE;
const Orientation COUNTERCLOCKWISE = POSITIVE;
const Orientation CLOCKWISE =NEGATIVE;
const Orientation COPLANAR = ZERO;
const Orientation COLLINEAR = ZERO;
const Orientation DEGENERATE = ZERO;

enum Comparison_result { SMALLER=-1, EQUAL, LARGER };

enum Oriented_side { ON_NEGATIVE_SIDE=-1, ON_ORIENTED_BOUNDARY, ON_POSITIVE_SIDE };

enum Bounded_side { ON_UNBOUNDED_SIDE=-1, ON_BOUNDARY, ON_BOUNDED_SIDE };

template <class T>
inline T opposite(const T& t)
{ return static_cast<T>( - static_cast<int>(t)); }

}

#endif



