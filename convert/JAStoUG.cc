
/*
 *  JAStoUG.cc: Convert .pts, .tetra and .out into Scalar and Vector UGs
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/String.h>
#include <Classlib/Pstreams.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Datatypes/VectorFieldUG.h>
#include <Datatypes/Mesh.h>
#include <Geometry/Point.h>
#include <Classlib/Array1.h>
#include <iostream.h>
#include <fstream.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 5 && argc != 6) {
	cerr << "usage: " << argv[0] << " .pts_file .tetra_file .out_file output_file\n";
	exit(0);
    }
    MeshHandle mesh=new Mesh;

    ifstream ptsfile(argv[1]);
    int n;
    ptsfile >> n;
    cerr << "nnodes=" << n << endl;
    while(ptsfile){
	double x,y,z;
	ptsfile >> x >> y >> z;
	if(ptsfile){
	    mesh->nodes.add(NodeHandle(new Node(Point(x,y,z))));
	}
    }
    cerr << "nnodes=" << mesh->nodes.size() << endl;

    ifstream tetrafile(argv[2]);
    int t;
    tetrafile >> t;
    cerr << "nelems=" << t << endl;
    while(tetrafile){
	int i1, i2, i3, i4, cond;
	tetrafile >> i1 >> i2 >> i3 >> i4 >> cond;
	if(tetrafile){
	    mesh->elems.add(new Element(mesh.get_rep(),
					i1-1, i2-1, i3-1, i4-1));
	}
    }
    cerr << "nelems=" << mesh->elems.size() << endl;

    ScalarFieldUG* sf=new ScalarFieldUG(mesh);
    VectorFieldUG* vf=new VectorFieldUG(mesh);

    ifstream outfile(argv[3]);
    for (int i=0; i<n; i++) {
	double v;
	outfile >> v;
	sf->data[i]=v;
    }
    double scale=1;
    sscanf(argv[5], "%lf", &scale);
    for (i=0; i<n; i++) {
	double x,y,z;
	outfile >> x >> y >> z;
	vf->data[i]=Vector(x,y,z)*scale;
    }

    char sfname[100], vfname[100];
    sprintf(sfname, "%s.sug", argv[4]);
    sprintf(vfname, "%s.vug", argv[4]);

    BinaryPiostream stream(sfname, Piostream::Write);
    Pio(stream, ScalarFieldHandle(sf));

    BinaryPiostream stream2(vfname, Piostream::Write);
    Pio(stream2, VectorFieldHandle(vf));

    return 0;
}
