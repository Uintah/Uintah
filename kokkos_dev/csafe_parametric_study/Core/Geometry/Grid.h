/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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

#include <Core/Containers/Array1.h>
#include <Core/Containers/Array3.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/Point.h>

// I have used the term element for a single cube in the grid and the 
// term member for the id of a triangle that passes through an element

namespace SCIRun {


class Grid {
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

} // End namespace SCIRun


#endif
