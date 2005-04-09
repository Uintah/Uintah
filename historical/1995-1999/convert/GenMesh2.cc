
/*
 *   ConvMesh.cc: Read pts/tetra files and dump a Mesh
 *
 *  Written by:
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/String.h>
#include <Classlib/Pstreams.h>
#include <Datatypes/Mesh.h>
#include <Geometry/Point.h>
#include <Classlib/Array1.h>
#include <Math/MusilRNG.h>
#include <iostream.h>
#include <fstream.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 4) {
	cerr << "usage: " << argv[0] << " pts_file nr nz output_file\n";
	exit(0);
    }
    MeshHandle mesh=new Mesh;
    int nr=atoi(argv[1]);
    int nz=atoi(argv[2]);
    for(int i=0;i<nz;i++){
	double z=double(i)/double(nz-1);
	for(int j=0;j<nr;j++){
	    double r=2*M_PI*double(j)/double(nr);
	    double xx=cos(r);
	    double yy=sin(r);
	    double zz=z*10;
	    mesh->nodes.add(NodeHandle(new Node(Point(xx,yy,zz))));
	}
    }
    TextPiostream stream(argv[argc-1], Piostream::Write);
    Pio(stream, mesh);
    return 0;
}
