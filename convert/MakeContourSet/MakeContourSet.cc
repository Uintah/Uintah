
/*
 *  MakeContourSet.cc: Read in Han-Wei's ouptput and dump a ContourSet
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
#include <ContourSet.h>
#include <Geometry/Point.h>
#include <Classlib/Array1.h>
#include <stdio.h>

main(int argc, char **argv) {

    if (argc != 2) {
	cerr << "Need name of set (ie axial/scalp)\n";
	exit(0);
    }
    ContourSetHandle set=new ContourSet;
    char hdrname[100];
    sprintf(hdrname, "/home/sci/data1/MRI/brain/contours/%s/header", argv[1]);
    ifstream hdrstream(hdrname);
    double b0x, b0y, b0z, b1x, b1y, b1z, b2x, b2y, b2z, ox, oy, oz;
    hdrstream >> set->name;
    int num;
    hdrstream >> num;
    hdrstream >> b0x >> b0y >> b0z >> b1x >> b1y >> b1z >> b2x >> b2y >> b2z;
    hdrstream >> ox >> oy >> oz;
    hdrstream >> set->space;
    set->basis[0]=Vector(b0x, b0y, b0z);
    set->basis[1]=Vector(b1x, b1y, b1z);
    set->basis[2]=Vector(b2x, b2y, b2z);
    set->origin=Vector(ox, oy, oz);
    for (int i=1; i<=num; i++) {
	char fullname[100];
	sprintf(fullname, "/home/sci/data1/MRI/brain/contours/%s/%03d", 
		argv[1], i);
	ifstream instream(fullname);
        Array1<Point> temp;
	while (instream) {
	    double x, y, z;
	    instream >> x >> y >> z;
	    temp.add(Point(x,y,i));
	}
	temp.remove(temp.size()-1);
	temp.remove(temp.size()-1);
	set->contours.add(temp);
    }
    set->build_bbox();
    TextPiostream stream("contourSet.out", Piostream::Write);
    Pio(stream, set);
}
