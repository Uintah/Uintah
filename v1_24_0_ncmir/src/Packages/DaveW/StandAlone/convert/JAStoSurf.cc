
/*
 *  JAStoSurf.cc: Read in John's .pts and .tri files and output a TriSurfField
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Containers/String.h>
#include <Core/Containers/Array1.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/TriSurfFieldace.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;
using std::ifstream;
using std::endl;

using namespace SCIRun;

main(int argc, char **argv) {
    TriSurfFieldace *surf=new TriSurfFieldace;
    char ptsname[100];
    char triname[100];
//    char nrmname[100];
    sprintf(ptsname, "%s.pts", argv[1]);
    sprintf(triname, "%s.tri", argv[1]);
//    sprintf(nrmname, "%s.nrm", argv[1]);

    ifstream ptsstream(ptsname);
    ifstream tristream(triname);
//    ifstream nrmstream(nrmname);

    char *name=argv[1];
    surf->name=name;

    int i;

    ptsstream >> i;

    cerr << "number of point ="<<i<<"\n";

    for (; i>0; i--) {
	double x, y, z;
	ptsstream >> x >> y >> z;
	surf->points.add(Point(x,y,z));
    }

    cerr << "done adding points.\n";

    while (tristream) {
	int n1, n2, n3;
	tristream >> n1 >> n2 >> n3;
	surf->elements.add(new TSElement(n1-1,n2-1,n3-1));
    }

//    while (nrmstream) {
    surf->normType=TriSurfFieldace::PointType;
    for (i=0; i<surf->points.size(); i++) {
//	double x, y, z;
//	nrmstream >> x >> y >> z;
	if (surf->points[i].x() == 0 || surf->points[i].x() == 1)
	    surf->normals.add(Vector(1,0,0));
	else if (surf->points[i].y() == 0 || surf->points[i].y() == 1)
	    surf->normals.add(Vector(0,1,0));
	else surf->normals.add(Vector(0,0,1));
    }
			  
    for (i=0; i<surf->points.size(); i++) {
	surf->bcIdx.add(i);
	surf->bcVal.add((surf->points[i].vector()).length2());
    }
    
    cerr << "done adding bc's\n";
    char tsname[100];
    sprintf(tsname, "%s.surf", name);
    TextPiostream stream(tsname, Piostream::Write);
    SurfaceHandle sh=surf;
    Pio(stream, sh);
}
