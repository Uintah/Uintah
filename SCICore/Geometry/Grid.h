
/*
 *  Grid.h: Uniform grid containing triangular elements
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Grid_h
#define SCI_project_Grid_h 1

#include <share/share.h>

#include <Containers/Array1.h>
#include <Containers/Array3.h>
#include <Containers/LockingHandle.h>
#include <Geometry/Point.h>

// I have used the term element for a single cube in the grid and the 
// term member for the id of a triangle that passes through an element

namespace SCICore {
namespace Geometry {

using SCICore::Containers::Array1;
using SCICore::Containers::Array3;

class SHARE Grid {
    double spacing;
    Point min;
    Array3<Array1<int> *> *e;
public:
    inline int dim1() const { return e->dim1(); }
    inline int dim2() const { return e->dim2(); }
    inline int dim3() const { return e->dim3(); }
    inline double get_spacing() const { return spacing; }
    inline Point get_min() const { return min; }
    Grid(int x, int y, int z, const Point &m, double sp);
    ~Grid();
    inline Array1<int> *get_members(int x, int y, int z) const {
	return (*e)(x,y,z); }
    inline Array1<int> *get_members(const Point &p) const {
	    return (*e)((int) ((p.x()-min.x())/spacing),
		(int) ((p.y()-min.y())/spacing),
		(int) ((p.z()-min.z())/spacing)); }
    void add_member(int id, int x, int y, int z);
    void size(int *i, int *j, int *k);
    int remove_member(int id, int x, int y, int z);
    void get_cubes_at_distance(int dist, int i, int j, int k, Array1<int>&);
    void get_cubes_within_distance(int dist, int i, int j, int k, Array1<int>&);
    void get_element(const Point &p, int *i, int *j, int *k);
    void get_element(const Point &p, int *i, int *j, int *k, double *dist);
    void get_intersected_elements(Array1<int> *inter, const Point &p1,
				   const Point &p2, const Point &p3);
    void add_triangle(int id, const Point &p1, const Point &p2, 
		      const Point &p3);
    void remove_triangle(int id, const Point &p1, const Point &p2, 
			 const Point &p3);
    int element_triangle_intersect(int i, int j, int k, const Point &p1,
				   const Point &p2, const Point &p3);
};

} // End namespace Geometry
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:55  mcq
// Initial commit
//
// Revision 1.4  1999/07/09 00:27:39  moulding
// added SHARE support for win32 shared libraries (.dll's)
//
// Revision 1.3  1999/05/06 19:56:16  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:17  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:27  dav
// Import sources
//
//

#endif
