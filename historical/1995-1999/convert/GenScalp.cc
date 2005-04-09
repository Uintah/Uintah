
/*
 *  GenScalp.cc: Nudge a set of contours in towards the center to create
 *	the inside surface of the scalp.
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <iostream.h>
#include <fstream.h>
#include <Classlib/String.h>
#include <Classlib/Pstreams.h>
#include <Math/Expon.h>
#include <stdlib.h>
#include <stdio.h>

#define NUDGE_DIST 	6

// Given the old point and the center, nudge the point towards
// the center by a distance of NUDGE_DIST
void getnewpoint(double x, double y, double cx, double cy, 
		 double *nx, double *ny) {
    double dx=cx-x;
    double dy=cy-y;
    double tot=Sqrt(dx*dx+dy*dy);
    dx/=tot;	// x-direction towards center
    dy/=tot;	// y-direction towards center
    dx*=NUDGE_DIST;
    dy*=NUDGE_DIST;
    *nx=x+dx;
    *ny=y+dy;
} 

main(int argc, char **argv) {

    if (argc != 2) {
	cerr << "Need name of set (ie patrick/axial)\n";
	exit(0);
    }
    char hdrname[100];
    char temp[100];
    sprintf(hdrname, "/home/sci/data1/MRI/brain/%s/scalp1/header", argv[1]);
    ifstream hdrstream(hdrname);
    hdrstream >> temp;
    int num;
    hdrstream >> num;
    double cx, cy;
    cx=268;
    cy=271;
    for (int i=1; i<=num; i++, cy--) {
	char fullname[100];
	char outname[100];
	sprintf(fullname, "/home/sci/data1/MRI/brain/%s/scalp1/%03d", 
		argv[1], i);
	sprintf(outname, "/home/sci/data1/MRI/brain/%s/scalp2/%03d", 
		argv[1], i);
	ifstream instream(fullname);
	ofstream outstream(outname);
	while (instream) {
	    double x, y, z, nx, ny;
	    instream >> x >> y >> z;
	    getnewpoint(x,y,cx,cy,&nx,&ny); 
	    outstream << nx << "  " << ny << "  " << z << "\n";
	}
    }
}
