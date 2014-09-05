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
 *  Grid.cc: Uniform grid containing triangular elements
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Geometry/Grid.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array3.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>

namespace SCIRun {


Grid::Grid(int x, int y, int z, const Point &m, double sp) 
: spacing(sp), min(m) {
    e=scinew Array3<Array1<int> *>(x,y,z);
    e->initialize((Array1<int>*)0);
}

Grid::~Grid() {
    for (int i=0; i<e->dim1(); i++) {
	for (int j=0;j<e->dim2(); j++) {
	    for (int k=0; k<e->dim3(); k++) {
		if (e->get_dataptr()[i][j][k]) {
		    delete e->get_dataptr()[i][j][k];
		}
	    }
	}
    }
    delete e;
}

void Grid::get_element(const Point &p, int *i, int *j, int *k) {
    Vector loc((p-min)/spacing);
    *i = (int) loc.x();
    *j = (int) loc.y();
    *k = (int) loc.z();
}

void Grid::get_element(const Point &p, int *i, int *j, int *k, double *dist) {


    Vector loc((p-min)/spacing);
    *i = (int) loc.x();
    *j = (int) loc.y();
    *k = (int) loc.z();
    double rx, ry, rz;
    rx=Abs(loc.x() - *i - .5);
    ry=Abs(loc.y() - *j - .5);
    rz=Abs(loc.z() - *k - .5);
    *dist=(.5 - Max(rx,ry,rz)) * spacing;
}

void Grid::add_member(int id, int x, int y, int z) {
    if (!(*e)(x,y,z)) {
	(e->get_dataptr())[x][y][z] = scinew Array1<int>(0,4,4);
    }
    (e->get_dataptr())[x][y][z]->add(id);
}

int Grid::remove_member(int id, int x, int y, int z) {
    if (!e->get_dataptr()[x][y][z]) return 0;
    for (int i=0; i<(e->get_dataptr())[x][y][z]->size(); i++) {
	if ((*(e->get_dataptr()[x][y][z]))[i] == id) {
	    e->get_dataptr()[x][y][z]->remove(i);
	    return 1;
	}
    }
    return 0;
}

void Grid::size(int *i, int *j, int *k) {
    *i=e->dim1();
    *j=e->dim2();
    *k=e->dim3();
}

void Grid::get_intersected_elements(Array1<int> *inter, const Point &p1,
				    const Point &p2, const Point &p3) {
    int i[3],j[3],k[3];
    get_element(p1, &i[0], &j[0], &k[0]);
    get_element(p2, &i[1], &j[1], &k[1]);
    get_element(p3, &i[2], &j[2], &k[2]);
    int xmin, xmax, ymin, ymax, zmin, zmax;
    xmin=Min(i[0], i[1], i[2]);
    xmax=Max(i[0], i[1], i[2]);
    ymin=Min(j[0], j[1], j[2]);
    ymax=Max(j[0], j[1], j[2]);
    zmin=Min(k[0], k[1], k[2]);
    zmax=Max(k[0], k[1], k[2]);
    for (int X=xmin; X<=xmax; X++) {
	for (int Y=ymin; Y<=ymax; Y++) {
	    for (int Z=zmin; Z<=zmax; Z++) {
		if (element_triangle_intersect(X,Y,Z,p1,p2,p3)) {
		    inter->add(X);
		    inter->add(Y);
		    inter->add(Z);
		}
	    }
	}
    }
}

void Grid::add_triangle(int id, const Point &p1, 
		      const Point &p2, const Point &p3) {
    Array1<int> intersect;
    if (id==19200) {
	id=19200;
    }
    get_intersected_elements(&intersect, p1, p2, p3);
    for (int i=0; i<intersect.size(); i+=3) {
	add_member(id, intersect[i], intersect[i+1], intersect[i+2]);
    }
}

void Grid::get_cubes_at_distance(int dist, int i, int j, int k, Array1<int>& set) {
    int imax=e->dim1();
    int jmax=e->dim1();
    int kmax=e->dim1();
    set.remove_all();
    if (dist==0) {
	set.add(i); set.add(j); set.add(k);
	return;
    }
    int curri=i-dist, currj, currk;
    for (curri=i-dist; curri<=i+dist; curri+=2*dist)
	if (curri>=0 && curri<imax)
	    for (currj=j-dist; currj<=j+dist; currj++)
		if (currj>=0 && currj<jmax)
		    for (currk=k-dist; currk<=k+dist; currk++)
			if (currk>=0 && currk<kmax) {
			    set.add(curri); set.add(currj); set.add(currk);
			}
    for (currj=j-dist; currj<=j+dist; currj+=2*dist)
	if (currj>=0 && currj<jmax)
	    for (currk=k-dist; currk<=k+dist; currk++)
		if (currk>=0 && currk<kmax)
		    for (curri=i-dist+1; curri<=i+dist-1; curri++)
			if (curri>=0 && curri<imax)  {
			    set.add(curri); set.add(currj); set.add(currk);
			}
    for (currk=k-dist; currk<=k+dist; currk+=2*dist)
	if (currk>=0 && currk<kmax)
	    for (curri=i-dist+1; curri<=i+dist-1; curri++)
		if (curri>=0 && curri<imax)
		    for (currj=j-dist+1; currj<=j+dist-1; currj++)
			if (currj>=0 && currj<jmax) {
			    set.add(curri); set.add(currj); set.add(currk);
			}
}

void Grid::get_cubes_within_distance(int dist, int i, int j, int k, Array1<int>& set) {
    int imax=e->dim1();
    int jmax=e->dim1();
    int kmax=e->dim1();
    set.remove_all();
    int curri, currj, currk;
    for (curri=i-dist; curri<=i+dist; curri++)
	if (curri>=0 && curri<imax)
	    for (currj=j-dist; currj<=j+dist; currj++)
		if (currj>=0 && currj<jmax)
		    for (currk=k-dist; currk<=k+dist; currk++)
			if (currk>=0 && currk<kmax) {
			    set.add(curri); set.add(currj); set.add(currk);
			}
}

void Grid::remove_triangle(int id, const Point &p1,
			   const Point &p2, const Point &p3) {
    Array1<int> intersect;
    get_intersected_elements(&intersect, p1, p2, p3);
    for (int i=0; i<intersect.size(); i+=3) {
	remove_member(id, intersect[i], intersect[i+1], intersect[i+2]);
    }
}

// we'll clip the triangle to the cube, and see if we have anything left
int Grid::element_triangle_intersect(int i, int j, int k, const Point &p1,
				      const Point &p2, const Point &p3) {
    double xmin, xmax, ymin, ymax, zmin, zmax;

    xmin=i*spacing+min.x(); ymin=j*spacing+min.y(); zmin=k*spacing+min.z();
    xmax=xmin+spacing;      ymax=ymin+spacing;      zmax=zmin+spacing;

    Array1<Point> cu;
    Array1<Point> ne;

    Array1<Point> *curr=&cu;
    Array1<Point> *next=&ne;
    Array1<Point> *temp;
    
    // there aren't any trivial rejects, so let's forge ahead...
    // put our three vertices in our point list

    curr->add(p1); curr->add(p2); curr->add(p3); curr->add(p1);
    
    // now we'll go through the points in our list, and intersect each
    // edge with the cube, storing the new points in next

    // first we'll intersect with xmin
    int a;
    for (a=0; a<curr->size()-1; a++) {
	if ((*curr)[a].x() >= xmin) {		// from in...
	    if ((*curr)[a+1].x() >= xmin) {		// ...to in
		next->add((*curr)[a+1]);
	    } else {					// ...to out
		next->add(Interpolate((*curr)[a], (*curr)[a+1], 
				      ((xmin-(*curr)[a].x())/
				       ((*curr)[a+1].x()-(*curr)[a].x()))));
	    }
	} else {				// from out...
	    if ((*curr)[a+1].x() >= xmin) {			// ...to in
		next->add(Interpolate((*curr)[a], (*curr)[a+1], 
				      ((xmin-(*curr)[a].x())/
				       ((*curr)[a+1].x()-(*curr)[a].x()))));
		next->add((*curr)[a+1]);
	    }
	}
    }
    if (next->size()) {next->add((*next)[0]);} else {return 0;}
    temp=curr; curr=next; next=temp;
    next->remove_all();

    // now ymin
    for (a=0; a<curr->size()-1; a++) {
	if ((*curr)[a].y() >= ymin) {		// from in...
	    if ((*curr)[a+1].y() >= ymin) {		// ...to in
		next->add((*curr)[a+1]);
	    } else {					// ...to out
		next->add(Interpolate((*curr)[a], (*curr)[a+1], 
				      ((ymin-(*curr)[a].y())/
				       ((*curr)[a+1].y()-(*curr)[a].y()))));
	    }
	} else {				// from out...
	    if ((*curr)[a+1].y() >= ymin) {			// ...to in
		next->add(Interpolate((*curr)[a], (*curr)[a+1], 
				      ((ymin-(*curr)[a].y())/
				       ((*curr)[a+1].y()-(*curr)[a].y()))));
		next->add((*curr)[a+1]);
	    }
	}
    }

    if (next->size()) {next->add((*next)[0]);} else {return 0;}
    temp=curr; curr=next; next=temp;
    next->remove_all();

    // now zmin
    for (a=0; a<curr->size()-1; a++) {
	if ((*curr)[a].z() >= zmin) {		// from in...
	    if ((*curr)[a+1].z() >= zmin) {		// ...to in
		next->add((*curr)[a+1]);
	    } else {					// ...to out
		next->add(Interpolate((*curr)[a], (*curr)[a+1], 
				      ((zmin-(*curr)[a].z())/
				       ((*curr)[a+1].z()-(*curr)[a].z()))));
	    }
	} else {				// from out...
	    if ((*curr)[a+1].z() >= zmin) {			// ...to in
		next->add(Interpolate((*curr)[a], (*curr)[a+1], 
				      ((zmin-(*curr)[a].z())/
				       ((*curr)[a+1].z()-(*curr)[a].z()))));
		next->add((*curr)[a+1]);
	    }
	}
    }

    if (next->size()) {next->add((*next)[0]);} else {return 0;}
    temp=curr; curr=next; next=temp;
    next->remove_all();

    // now xmax
    for (a=0; a<curr->size()-1; a++) {
	if ((*curr)[a].x() <= xmax) {		// from in...
	    if ((*curr)[a+1].x() <= xmax) {		// ...to in
		next->add((*curr)[a+1]);
	    } else {					// ...to out
		next->add(Interpolate((*curr)[a], (*curr)[a+1], 
				      ((xmax-(*curr)[a].x())/
				       ((*curr)[a+1].x()-(*curr)[a].x()))));
	    }
	} else {				// from out...
	    if ((*curr)[a+1].x() <= xmax) {			// ...to in
		next->add(Interpolate((*curr)[a], (*curr)[a+1], 
				      ((xmax-(*curr)[a].x())/
				       ((*curr)[a+1].x()-(*curr)[a].x()))));
		next->add((*curr)[a+1]);
	    }
	}
    }
    if (next->size()) {next->add((*next)[0]);} else {return 0;}
    temp=curr; curr=next; next=temp;
    next->remove_all();

    // now ymax
    for (a=0; a<curr->size()-1; a++) {
	if ((*curr)[a].y() <= ymax) {		// from in...
	    if ((*curr)[a+1].y() <= ymax) {		// ...to in
		next->add((*curr)[a+1]);
	    } else {					// ...to out
		next->add(Interpolate((*curr)[a], (*curr)[a+1], 
				      ((ymax-(*curr)[a].y())/
				       ((*curr)[a+1].y()-(*curr)[a].y()))));
	    }
	} else {				// from out...
	    if ((*curr)[a+1].y() <= ymax) {			// ...to in
		next->add(Interpolate((*curr)[a], (*curr)[a+1], 
				      ((ymax-(*curr)[a].y())/
				       ((*curr)[a+1].y()-(*curr)[a].y()))));
		next->add((*curr)[a+1]);
	    }
	}
    }

    if (next->size()) {next->add((*next)[0]);} else {return 0;}
    temp=curr; curr=next; next=temp;
    next->remove_all();

    // now zmax
    for (a=0; a<curr->size()-1; a++) {
	if ((*curr)[a].z() <= zmax) {		// from in...
	    if ((*curr)[a+1].z() <= zmax) {		// ...to in
		next->add((*curr)[a+1]);
	    } else {					// ...to out
		next->add(Interpolate((*curr)[a], (*curr)[a+1], 
				      ((zmax-(*curr)[a].z())/
				       ((*curr)[a+1].z()-(*curr)[a].z()))));
	    }
	} else {				// from out...
	    if ((*curr)[a+1].z() <= zmax) {			// ...to in
		next->add(Interpolate((*curr)[a], (*curr)[a+1], 
				      ((zmax-(*curr)[a].z())/
				       ((*curr)[a+1].z()-(*curr)[a].z()))));
		next->add((*curr)[a+1]);
	    }
	}
    }
    return (next->size());
}

} // End namespace SCIRun


