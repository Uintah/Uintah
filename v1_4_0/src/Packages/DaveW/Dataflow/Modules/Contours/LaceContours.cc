
/*
 *  LaceContours.cc:  Merge multiple contour sets into a Surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   August 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Containers/Array1.h>
#include <Core/Containers/Assert.h>
#include <Core/Containers/NotFinished.h>
#include <Dataflow/Dataflow/Module.h>
#include <Dataflow/Datatypes/ContourSet.h>
#include <Dataflow/Datatypes/ContourSetPort.h>
#include <Dataflow/Datatypes/Surface.h>
#include <Dataflow/Datatypes/SurfacePort.h>
#include <Dataflow/Datatypes/TriSurfFieldace.h>
#include <Core/Geometry/Grid.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Expon.h>

#include <iostream>

using namespace SCIRun;

class LaceContours : public Module {
    ContourSetIPort* incontour;
    SurfaceOPort* osurface;
    void lace_contours(const ContourSetHandle& contour, TriSurfFieldace* surf);
public:
    LaceContours(const clString&);
    LaceContours(const LaceContours&, int deep);
    virtual ~LaceContours();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_LaceContours(const clString& id)
{
    return new LaceContours(id);
}
}

LaceContours::LaceContours(const clString& id)
: Module("LaceContours", id, Filter)
{
    // Create the input port
    incontour=new ContourSetIPort(this, "ContourSet", 
				  ContourSetIPort::Atomic);

    add_iport(incontour);
    osurface=new SurfaceOPort(this, "Surface", SurfaceIPort::Atomic);
    add_oport(osurface);
}

LaceContours::LaceContours(const LaceContours&copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("LaceContours::LaceContours");
}

LaceContours::~LaceContours()
{
}

Module* LaceContours::clone(int deep)
{
    return new LaceContours(*this, deep);
}

void LaceContours::execute()
{
    ContourSetHandle contours;
    if (!incontour->get(contours)) return;
    TriSurfFieldace* surf=new TriSurfFieldace;
    lace_contours(contours, surf);
    osurface->send(SurfaceHandle(surf));
}

void LaceContours::lace_contours(const ContourSetHandle& contour, 
				   TriSurfFieldace* surf) {
    surf->name=contour->name;
    Array1<int> row;	
    int i;
    int curr;
    for (i=curr=0; i<contour->contours.size(); i++) {
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
	       surf->add_triangle(row[i]+j, row[i-1]+k,
				  row[i-1]+((k+1)%sz_bot));
	       k=(k+1)%sz_bot;
	       if (k==bot) kdone=1;
	   } else {     			// top moves
	       surf->add_triangle(row[i]+j,row[i-1]+k,
				  row[i]+((j+1)%sz_top));
	       j=(j+1)%sz_top;
	       if (j==top) jdone=1;
	   }
       }
   }
}
