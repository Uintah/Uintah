
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
    if (argc != 3) {
	cerr << "usage: " << argv[0] << " pts_file n output_file\n";
	exit(0);
    }
    MeshHandle mesh=new Mesh;
    int n=atoi(argv[1]);
    MusilRNG rng;
    for(int i=0;i<n;i++){
	double x=double(i)/double(n-1);
	for(int j=0;j<n;j++){
	    double y=double(j)/double(n-1);
	    for(int k=0;k<n;k++){
		double z=double(k)/double(n-1);
		double xx=x;
		double yy=y;
		double zz=z;
		if(i>0 && i<n-1)
		    xx+=rng()*0.3/double(n-1);
		if(j>0 && j<n-1)
		    yy+=rng()*0.3/double(n-1);
		if(k>0 && k<n-1)
		    zz+=rng()*0.3/double(n-1);
		mesh->nodes.add(NodeHandle(new Node(Point(xx,yy,zz))));
	    }
	}
    }
    TextPiostream stream(argv[argc-1], Piostream::Write);
    Pio(stream, mesh);
    return 0;
}
