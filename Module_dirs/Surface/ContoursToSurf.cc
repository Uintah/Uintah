
/*
 *  ContoursToSurf.cc:  Merge multiple contour sets into a Surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Modules/Surface/ContoursToSurf.h>
#include <Classlib/Array1.h>
#include <Classlib/Assert.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/ContourSet.h>
#include <Datatypes/ContourSetPort.h>
#include <Datatypes/Surface.h>
#include <Datatypes/SurfacePort.h>
#include <Geometry/Grid.h>
#include <Math/MiscMath.h>
#include <Math/MinMax.h>
#include <Math/Expon.h>

#include <iostream.h>
#include <fstream.h>

#define Sqr(x) ((x)*(x))
static Module* make_ContoursToSurf(const clString& id)
{
    return new ContoursToSurf(id);
}

static RegisterModule db1("Contours", "Contours To Surface", make_ContoursToSurf);
static RegisterModule db2("Visualization", "Contours To Surface", make_ContoursToSurf);
static RegisterModule db3("Surfaces", "Contours To Surface", make_ContoursToSurf);
static RegisterModule db4("Dave", "Contours To Surface", make_ContoursToSurf);

ContoursToSurf::ContoursToSurf(const clString& id)
: Module("ContoursToSurf", id, Filter)
{
    // Create the input port
    incontours.add(new ContourSetIPort(this, "ContourSet", 
				       ContourSetIPort::Atomic));

    add_iport(incontours[0]);
    osurface=new SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurface);
}

void ContoursToSurf::connection(ConnectionMode mode, int which_port,
			       int output)
{
    if (output) return;
    if (mode==Disconnected) {
	remove_iport(which_port);
	delete incontours[which_port];
	incontours.remove(which_port);
    } else {
	ContourSetIPort* ci=new ContourSetIPort(this, "ContourSet", 
						ContourSetIPort::Atomic);
	add_iport(ci);
	incontours.add(ci);
    }
}
	
ContoursToSurf::ContoursToSurf(const ContoursToSurf&copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("ContoursToSurf::ContoursToSurf");
}

ContoursToSurf::~ContoursToSurf()
{
}

Module* ContoursToSurf::clone(int deep)
{
    return new ContoursToSurf(*this, deep);
}

void ContoursToSurf::execute()
{
    Array1<ContourSetHandle> contours(incontours.size()-1);
    for (int flag=0, i=0; i<incontours.size()-1; i++)
	if (!incontours[i]->get(contours[i]))
	    flag=1;
    if (flag) return;
    TriSurface* surf=new TriSurface;
    contours_to_surf(contours, surf);
/*    for (i=0; i<surf->points.size(); i++) {
	ASSERT((surf->points[i]-Point(1,1,1)).length()<1.1);
    }
    for (i=0; i<surf->elements.size(); i++) {
	ASSERT((surf->points[surf->elements[i]->i1]-Point(1,1,1)).length()<1.1);
	ASSERT((surf->points[surf->elements[i]->i2]-Point(1,1,1)).length()<1.1);
	ASSERT((surf->points[surf->elements[i]->i3]-Point(1,1,1)).length()<1.1);
    }
*/
    osurface->send(surf);
}

void ContoursToSurf::lace_contours(const ContourSetHandle& contour, 
				   TriSurface* surf) {
    Array1<int> row;	
    for (int i=0, curr=0; i<contour->contours.size(); i++) {
	row.add(curr);	
	curr+=contour->contours[i].size();
	for (int j=0; j<contour->contours[i].size(); j++) {
	    Point p(contour->contours[i][j]-contour->origin);
	    p=Point(0,0,0)+contour->basis[0]*p.x()+contour->basis[1]*p.y()+
		contour->basis[2]*(p.z()*contour->space);
	    surf->add_point(p);
	}
    }
   // i will be the index of the top contour being laced, i-1 being the other
   for (i=1; i<contour->contours.size(); i++) {
       int top=0;
       double dtemp;
       int sz_top=contour->contours[i].size();
       int sz_bot=contour->contours[i-1].size();
       if ((sz_top < 2) && (sz_bot < 2)) {
	   cerr << "Not enough points to lace!\n";
	   return;
       }
       // 0 will be the index of our first bottom point, set top to be the 
       // index of the closest top point to it
       double dist=Sqr(contour->contours[i][0].x()-
		       contour->contours[i-1][0].x())+
		   Sqr(contour->contours[i][0].y()-
		       contour->contours[i-1][0].y());
       for (int start=1; start<sz_top; start++) {
	   if ((dtemp=(Sqr(contour->contours[i][start].x()-
			   contour->contours[i-1][0].x())+
		       Sqr(contour->contours[i][start].y()-
			   contour->contours[i-1][0].y())))<dist) {
	       top=start;
	       dist=dtemp;
	   }
       }
       int bot=0;
       // lets start lacing...  top and bottom will always store the indices
       // of the first matched points so we know when to stop
       int jdone=(sz_top==1); // does this val have to change for us to
       int kdone=(sz_bot==1); // be done lacing
       for (int j=top,k=bot; !jdone || !kdone;) {
	   double d1=Sqr(contour->contours[i][j].x()-
			 contour->contours[i-1][(k+1)%sz_bot].x())+
	       Sqr(contour->contours[i][j].y()-
		   contour->contours[i-1][(k+1)%sz_bot].y());
	   double d2=Sqr(contour->contours[i][(j+1)%sz_top].x()-
			 contour->contours[i-1][k].x())+
	             Sqr(contour->contours[i][(j+1)%sz_top].y()-
			 contour->contours[i-1][k].y());
	   if ((d1<d2 || jdone) && !kdone){ 	// bottom moves
	       grid->add_triangle(surf->add_triangle(row[i]+j, row[i-1]+k,
						     row[i-1]+((k+1)%sz_bot)),
				  surf->points[row[i]+j],
				  surf->points[row[i-1]+k],
				  surf->points[row[i-1]+((k+1)%sz_bot)]);
	       k=(k+1)%sz_bot;
	       if (k==bot) kdone=1;
	   } else {     			// top moves
	       grid->add_triangle(surf->add_triangle(row[i]+j,row[i-1]+k,
						     row[i]+((j+1)%sz_top)),
				  surf->points[row[i]+j], 
				  surf->points[row[i-1]+k],
				  surf->points[row[i]+((j+1)%sz_top)]);
	       j=(j+1)%sz_top;
	       if (j==top) jdone=1;
	   }
       }
   }
}

void ContoursToSurf::break_triangle(int tri_id, int pt_id, const Point& p,
				    TriSurface* surf) {
    int v0=surf->elements[tri_id]->i1;
    int v1=surf->elements[tri_id]->i2;
    int v2=surf->elements[tri_id]->i3;
    surf->remove_triangle(tri_id);
    Point p0(surf->points[v0]);
    Point p1(surf->points[v1]);
    Point p2(surf->points[v2]);
    grid->remove_triangle(tri_id, p0, p1, p2);
    static Point pp;
    pp=surf->points[v0];
    pp=surf->points[v1];
    pp=surf->points[v2];
    pp=surf->points[pt_id];
    grid->add_triangle(surf->add_triangle(v0, v1, pt_id), p0, p1, p);
    grid->add_triangle(surf->add_triangle(v1, v2, pt_id), p1, p2, p);
    grid->add_triangle(surf->add_triangle(v0, v2, pt_id), p0, p2, p);
}

void ContoursToSurf::break_edge2(int tri1, int e1, int pt_id,
				const Point &p, TriSurface *surf) {
    ASSERT(e1<3 && e1>=0);
    int v[3];
    v[0]=surf->elements[tri1]->i1;
    v[1]=surf->elements[tri1]->i2;
    v[2]=surf->elements[tri1]->i3;
    Point pts[3];
    pts[0]=surf->points[v[0]];
    pts[1]=surf->points[v[1]];
    pts[2]=surf->points[v[2]];
    static Point pp;
    pp=surf->points[v[0]];
    pp=surf->points[v[1]];
    pp=surf->points[v[2]];
    pp=surf->points[pt_id];
    grid->add_triangle(surf->add_triangle(pt_id, v[(e1+2)%3], v[(e1+1)%3]),
		       p, pts[(e1+2)%3], pts[(e1+1)%3]);
}

void ContoursToSurf::break_edge(int tri1, int tri2, int e1, int e2, int pt_id,
				const Point &p, TriSurface *surf) {
    ASSERT(e1<3 && e1>=0);
    ASSERT(e2<3 && e2>=0);

    // first break tri1
    if (pt_id==11218)
	pt_id=11218;
    int v[3];
    v[0]=surf->elements[tri1]->i1;
    v[1]=surf->elements[tri1]->i2;
    v[2]=surf->elements[tri1]->i3;
    static Point pp;
    pp=surf->points[v[0]];
    pp=surf->points[v[1]];
    pp=surf->points[v[2]];
    pp=surf->points[pt_id];
    surf->remove_triangle(tri1);
    Point pts[3];
    pts[0]=surf->points[v[0]];
    pts[1]=surf->points[v[1]];
    pts[2]=surf->points[v[2]];
    grid->add_triangle(surf->add_triangle(pt_id, v[e1], v[(e1+1)%3]),
		       p, pts[e1], pts[(e1+1)%3]);
    grid->add_triangle(surf->add_triangle(pt_id, v[e1], v[(e1+2)%3]),
		       p, pts[e1], pts[(e1+2)%3]);

    // now break tri2
    v[0]=surf->elements[tri2]->i1;
    v[1]=surf->elements[tri2]->i2;
    v[2]=surf->elements[tri2]->i3;
    surf->remove_triangle(tri2);
    pts[0]=surf->points[v[0]];
    pts[1]=surf->points[v[1]];
    pts[2]=surf->points[v[2]];
    pp=surf->points[v[0]];
    pp=surf->points[v[1]];
    pp=surf->points[v[2]];
    grid->add_triangle(surf->add_triangle(pt_id, v[e2], v[(e2+1)%3]),
		       p, pts[e2], pts[(e2+1)%3]);
    grid->add_triangle(surf->add_triangle(pt_id, v[e2], v[(e2+2)%3]),
		       p, pts[e2], pts[(e2+2)%3]);
}

void ContoursToSurf::add_point(const Point& p, TriSurface* surf) {
    Array1<int> res;

    double dist=distance(p, res, surf);
    if (res.size() > 4) {	// we were closest to a vertex
	return;
    } 
    surf->add_point(p);
    if (res.size() == 4) {	// we were closest to an edge 
	    break_edge(res[0], res[2], res[1], res[3], surf->points.size()-1,
		       p, surf);
	    return;
    }
    if (res.size() == 2) {	// we were to a boundary
	    break_edge2(res[0], res[1], surf->points.size()-1, p, surf);
	    return;
    }
    break_triangle(res[0], surf->points.size()-1, p, surf);
}

Array1<int>* ContoursToSurf::get_cubes_at_distance(int dist, int i, int j,
				   int k, int imax, int jmax, int kmax) {
    Array1<int>* set=new Array1<int>(0,10,12*(2*dist+1)*(2*dist+1));
    if (dist==0) {
	set->add(i); set->add(j); set->add(k);
	return set;
    }
    int curri=i-dist, currj, currk;
    for (curri=i-dist; curri<=i+dist; curri+=2*dist)
	if (curri>=0 && curri<=imax)
	    for (currj=j-dist; currj<=j+dist; currj++)
		if (currj>=0 && currj<=jmax)
		    for (currk=k-dist; currk<=k+dist; currk++)
			if (currk>=0 && currk<=kmax) {
			    set->add(curri); set->add(currj); set->add(currk);
			}
    for (currj=j-dist; currj<=j+dist; currj+=2*dist)
	if (currj>=0 && currj<=jmax)
	    for (currk=k-dist; currk<=k+dist; currk++)
		if (currk>=0 && currk<=kmax)
		    for (curri=i-dist+1; curri<=i+dist-1; curri++)
			if (curri>=0 && curri<=imax)  {
			    set->add(curri); set->add(currj); set->add(currk);
			}
    for (currk=k-dist; currk<=k+dist; currk+=2*dist)
	if (currk>=0 && currk<=kmax)
	    for (curri=i-dist+1; curri<=i+dist-1; curri++)
		if (curri>=0 && curri<=imax)
		    for (currj=j-dist+1; currj<=j+dist-1; currj++)
			if (currj>=0 && currj<=jmax) {
			    set->add(curri); set->add(currj); set->add(currk);
			}
    return set;
}

// find which "cube" of the mesh it's in.  look for nearest neighbors.
// determine if it's closest to a vertex, edge, or triangle, and store
// the type in "type" and the info (i.e. edge #, vertex #, triangle #, etc)
// in "res" (result).  finally, return the distance.
// the information in res will be stored thus...
// if the closest thing is a:
// 	triangle -- [0]=triangle index
//	edge -- [0]=triangle[1] index
//		[1]=triangle[1] edge #
//		[2]=triangle[2] index
//		[3]=triangle[2] edge #
//	vertex -- [0]=triangle[1] index
//		  [1]=triangle[1] vertex #
//		   ...

double ContoursToSurf::distance(const Point &p, Array1<int> &res, 
				TriSurface *surf) {
    Array1<int>* candid;
    Array1<int>* elem;
    Array1<int> tri;
    int i, j, k, imax, jmax, kmax;
    
    double dmin;
    double sp=grid->get_spacing();
    grid->get_element(p, &i, &j, &k, &dmin);
    grid->size(&imax, &jmax, &kmax);
    imax--; jmax--; kmax--;
    int dist=0;
    int done=0;
    double Dist=1000000;
    Array1<int> info;
    int type;
    while (!done) {
	while (!tri.size()) {
	    candid=get_cubes_at_distance(dist, i, j, k, imax, jmax, kmax);
	    for(int index=0; index<candid->size(); index+=3) {
		elem=grid->get_members((*candid)[index], (*candid)[index+1], 
				       (*candid)[index+2]);
		if (elem) {
		    for (int a=0; a<elem->size(); a++) {
			for (int duplicate=0, b=0; b<tri.size(); b++)
			    if (tri[b]==(*elem)[a]) duplicate=1;
			for (b=0; b<info.size(); b+=2)
			    if (info[b]==(*elem)[a]) duplicate=1;
			if (!duplicate) tri.add((*elem)[a]);
		    }
		}
	    }
	    dist++;
	    delete candid;
	}
	// now tri holds the indices of the triangles we're closest to


//    for (int ind=0; ind<surf->elements.size(); ind++) {
//	tri.add(ind);
//    }
	for (int index=0; index<tri.size(); index++) {
	    double d=surf->distance(p, tri[index], &type);
	    if (Abs(d-Dist)<.00001) {
		if (type==0) {
		    info.remove_all();
		    Dist=d;
		    info.add(tri[index]);
		} else {
		    if (res.size() != 1) {
			info.add(tri[index]);
			info.add((type-1)%3);
		    }
		}
	    } else if (d<Dist) {
		info.remove_all();
		Dist=d;
		info.add(tri[index]);
		if (type>0) info.add((type-1)%3);
	    }
	}

	tri.remove_all();

	// if our closest point is INSIDE of the squares we looked at...
	if (Dist<(dmin+(dist-1)*sp)) 
	    done=1;		// ... we're done
    }
	
    res=info;
    return Dist;
}


void ContoursToSurf::contours_to_surf(const Array1<ContourSetHandle> &contours,
				   TriSurface *surf) {
    // have to make sure all of the contour-sets have valid bboxes...
    // ...the build_bbox() method does just that.

    BBox bb;
    for (int i=0; i<contours.size(); i++) {
	contours[i]->build_bbox();
	BBox bb0(contours[i]->bbox);
	for (int ti=0; ti<=1; ti++) 
	    for (int tj=0; tj<=1; tj++) 
		for (int tk=0; tk<=1; tk++) {
		    Point p1(ti?bb0.min().x():bb0.max().x(),
			     tj?bb0.min().y():bb0.max().y(),
			     tk?bb0.min().z():bb0.max().z());
		    p1-=contours[i]->origin;
		    bb.extend(Point(0,0,0)+contours[i]->basis[0]*p1.x()+
			      contours[i]->basis[1]*p1.y()+
			      contours[i]->basis[2]*
			                   (p1.z()*contours[i]->space));
		}
    }

    // the first contour has to be the most complete, so we'll use it for
    // the number of points, and we're assuming that the middle slice of
    // that set has an average number of points
    double num_pts=contours[0]->contours.size()*
	contours[0]->contours[contours[0]->contours.size()/2].size();
    Vector sides=bb.max()-bb.min();
    double spacing=Cbrt(sides.x()*sides.y()*sides.z()/(num_pts*2.5)); 
    
    Vector parts(sides/spacing);
    grid = new Grid((int)parts.x()+1, (int)parts.y()+1, 
		    (int)parts.z()+1, bb.min(), spacing);
    lace_contours(contours[0], surf);

    Array1<int> cube_sizes(100);
    for (i=0; i<100; i++)
	cube_sizes[i]=0;
    for (i=0; i<grid->dim1(); i++) {
	for (int j=0; j<grid->dim2(); j++) {
	    for (int k=0; k<grid->dim3(); k++) {
		Array1<int> *qw=grid->get_members(i,j,k);
		if (qw && qw->size()<100) {
		    cube_sizes[qw->size()]++;
		} else {
		    cube_sizes[0]++;
		}
	    }
	}
    }
    for (i=0; i<100; i++) {
	if (cube_sizes[i]) {
	    cerr << i << ": " << cube_sizes[i] << "\n";
	}
    }
//    static int count=0;
//    int bob=0;
    for (i=1;i<contours.size();i++) {
	for (int j=0; j<contours[i]->contours.size(); j++) {
	    for (int k=0; k<contours[i]->contours[j].size(); k++) {
		Point p(contours[i]->contours[j][k]-contours[i]->origin);
		p=Point(0,0,0)+contours[i]->basis[0]*p.x()+
		    contours[i]->basis[1]*p.y()+
		    contours[i]->basis[2]*(p.z()*contours[i]->space);
//		if (bob++ < count) 
		    add_point(p, surf);
	    }	
	}
    }
//    count++;
    delete grid;
}
