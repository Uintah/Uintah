
/*
 *  JAStoSurf.cc: Read in John's .pts and .tri files and output a TriSurf
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
#include <Geometry/Vector.h>

main(int argc, char **argv) {

    double tx, ty, tz, sc;
    int cautious = 0;
    if (argc < 2) {
	cerr << "Usage: "<<argv[0]<<" surfName [-cautious | sc tx ty tz]\n";
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
    if (argc == 3) {
	if (clString(argv[1]) == clString("-cautious"))
	    cautious = 1;
    }
    TriSurface *surf=new TriSurface;
    char ptsname[100];
    char triname[100];
    char nrmname[100];
    char outname[100];
    char natename[100];
    sprintf(ptsname, "%s.pts", argv[1]);
    sprintf(triname, "%s.tri", argv[1]);
    sprintf(nrmname, "%s.nrm", argv[1]);
    sprintf(natename, "%s.nate", argv[1]);
    sprintf(outname, "%s.out", argv[1]);

    ifstream ptsstream(ptsname);
    ifstream tristream(triname);

    char *name=&(argv[1][0]);
    for (int i=0; argv[1][i]!='\0'; i++)
	if (argv[1][i]=='/') name=&(argv[1][i+1]);
    surf->name=name;

    ptsstream >> i;
    Point mid(0,0,0);
    for (; i>0; i--) {
	double x, y, z;
	ptsstream >> x >> y >> z;
	surf->points.add(Point(tx+(sc*x),ty+(sc*y),tz+(sc*z)));
	mid+=Vector(x,y,z);
    }
    mid.x(mid.x()/surf->points.size());
    mid.y(mid.y()/surf->points.size());
    mid.z(mid.z()/surf->points.size());
    cerr << "Midpoint = "<<mid<<"\n";
    while (tristream) {
	int n1, n2, n3;
	tristream >> n1 >> n2 >> n3;
	if (!tristream) continue;
	Vector v1(surf->points[n1-1]-surf->points[n2-1]);
	Vector v2(surf->points[n1-1]-surf->points[n3-1]);
	Vector v3(surf->points[n2-1]-surf->points[n3-1]);
	if (!cautious || (v1.length2() > .000001 && v2.length2() > .000001 && 
	    v3.length2() > .000001)) {
	    TSElement* elem=new TSElement(n1-1,n2-1,n3-1);
	    surf->elements.add(elem);
	} else {
	    cerr << "Removed degenerate triangle: "<<n1<<", "<<n2<<", "<<n3<<"\n";
	}
    }

    FILE *fin=fopen(nrmname, "rt");
    if (fin) {
	int nnrmls;
	double nx, ny, nz;
	fscanf(fin, "%d", &nnrmls);
	surf->normals.resize(nnrmls);
	if (nnrmls == surf->points.size()) {
	    surf->normType=TriSurface::PointType;
	} else if (nnrmls == surf->elements.size()) {
	    surf->normType=TriSurface::ElementType;
	} else if (nnrmls == surf->elements.size()*3) {
	    surf->normType=TriSurface::VertexType;
	} else {
	    cerr << "Error -- I have "<<surf->points.size()<<" points and "<<surf->elements.size()<<" elements. \nDon't know what to do with "<<nnrmls<<" normals!\n";
	    surf->normals.resize(0);
	    surf->normType=TriSurface::None;
	}
	for (int ii=0; ii<surf->normals.size(); ii++) {
	    fscanf(fin, "%lf %lf %lf", &nx, &ny, &nz);
	    surf->normals[ii]=Vector(nx,ny,nz);
	    surf->normals[ii].normalize();
	}
	cerr << "Using normals.  Found "<<surf->normals.size()<<" of them.\n";
    }
    fclose(fin);

    int havemap=0;
    Array1<int> ptmap;

    fin=fopen(natename, "rt");
    if (fin) {
	// this needs to be fixed to also adjust normals!
	havemap=1;
	int lastvalid=0;
	for (int i=0; i<surf->elements.size(); i++) {
	    int ok;
	    fscanf(fin, "%d", &ok);
	    if (ok) {
		surf->elements[lastvalid]=surf->elements[i];
		lastvalid++;
	    }
	}
	cerr << "Eliminated: "<<surf->elements.size()-lastvalid<<" of "<<surf->elements.size()<<" elements ("<<(surf->elements.size()-lastvalid)*100./(surf->elements.size())<<"% reduction)\n";
	surf->elements.resize(lastvalid);
	Array1<int> usedpts(surf->points.size());
	usedpts.initialize(0);
	for (i=0; i<surf->elements.size(); i++) {
	    usedpts[surf->elements[i]->i1]=1;
	    usedpts[surf->elements[i]->i2]=1;
	    usedpts[surf->elements[i]->i3]=1;
	}
	ptmap.resize(surf->points.size());
	ptmap.initialize(-1);
	lastvalid=0;
	for (i=0; i<surf->points.size(); i++) {
	    if (usedpts[i]) {
		surf->points[lastvalid]=surf->points[i];
		ptmap[i]=lastvalid;
		lastvalid++;
	    }
	}
	surf->points.resize(lastvalid);
	for (i=0; i<surf->elements.size(); i++) {
	    surf->elements[i]->i1 = ptmap[surf->elements[i]->i1];
	    surf->elements[i]->i2 = ptmap[surf->elements[i]->i2];
	    surf->elements[i]->i3 = ptmap[surf->elements[i]->i3];
	}
    }

    fin=fopen(outname, "rt");
    if (fin) {
	int nv;
	int idx;
	double val;
	fscanf(fin, "%d", &nv);
	surf->bcIdx.resize(nv);
	surf->bcVal.resize(nv);
	for (int ii=0; ii<nv; ii++) {
	    fscanf(fin, "%d %lf", &idx, &val);
	    if (havemap) {
		if (ptmap[ii] != -1) {
		    surf->bcIdx[ptmap[ii]]=idx-1;
		    surf->bcVal[ptmap[ii]]=val;
		}
	    } else {
		surf->bcIdx[ii]=idx-1;
		surf->bcVal[ii]=val;
	    }
	}
	cerr << "Using values at nodes.  Found "<<nv<<" of them.\n";
    }
    fclose(fin);

    char tsname[100];
    sprintf(tsname, "%s.surf", name);
    TextPiostream stream(tsname, Piostream::Write);
    SurfaceHandle sh=surf;
    Pio(stream, sh);
}
