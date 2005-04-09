
/*
 *  VPtoTriSurf.cc: Read in ViewPoint .pts and .tri files and output a TriSurf
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>
#include <Classlib/String.h>
#include <Classlib/Pstreams.h>
#include <Datatypes/TriSurface.h>
#include <Geometry/Point.h>
#include <Classlib/Array1.h>
#include <stdio.h>

main(int argc, char **argv) {

    double tx, ty, tz, sc;
    if (argc < 2) {
	cerr << "Need name of set.\n";
	exit(0);
    }
    tx=ty=tz=0;
    sc=1;
    if (argc == 6) {
	sc=atol(argv[2]);
	tx=atol(argv[3]);
	ty=atol(argv[4]);
	tz=atol(argv[5]);
    }
    TriSurface *surf=new TriSurface;
    char ptsname[100];
    char triname[100];
    sprintf(ptsname, "%s.pts", argv[1]);
    sprintf(triname, "%s.tri", argv[1]);
    ifstream ptsstream(ptsname);
    ifstream tristream(triname);
#if 0
    surf->conductivity.add(1);
    surf->conductivity.add(0);
    surf->conductivity.add(0);
    surf->conductivity.add(1);
    surf->conductivity.add(0);
    surf->conductivity.add(1);
    surf->bdry_type=TriSurface::Interior;
#endif
    char *name=&(argv[1][0]);
    for (int i=0; argv[1][i]!='\0'; i++)
	if (argv[1][i]=='/') name=&(argv[1][i+1]);
    surf->name=name;

    ptsstream >> i;
    for (; i>0; i--) {
	double x, y, z;
	ptsstream >> x >> y >> z;
	surf->points.add(Point(tx+(sc*x),ty+(sc*y),tz+(sc*z)));
    }
    while (tristream) {
	int n1, n2, n3;
	tristream >> n1 >> n2 >> n3;
	TSElement* elem=new TSElement(n1-1,n2-1,n3-1);
	surf->elements.add(elem);
    }
    
    char tsname[100];
    sprintf(tsname, "/home/grad/dweinste/mydata/%s.surf", name);
    TextPiostream stream(tsname, Piostream::Write);
    SurfaceHandle sh=surf;
    Pio(stream, sh);
}
