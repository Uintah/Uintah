/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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

#include <Core/share/share.h>

#include <Core/share/share.h>

#include <Core/Containers/Array1.h>
#include <Core/Containers/Array3.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/Point.h>

// I have used the term element for a single cube in the grid and the 
// term member for the id of a triangle that passes through an element

namespace SCIRun {


class SCICORESHARE Grid {
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
