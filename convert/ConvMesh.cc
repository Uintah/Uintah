
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
#include <iostream.h>
#include <fstream.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 4 && argc != 3) {
	cerr << "usage: " << argv[0] << " pts_file [tetra_file] output_file\n";
	exit(0);
    }
    MeshHandle mesh=new Mesh;
    ifstream ptsfile(argv[1]);
    while(ptsfile){
	double x,y,z;
	ptsfile >> x >> y >> z;
	if(ptsfile){
	    mesh->nodes.add(new Node(Point(x,y,z)));
	}
    }
    if(argc == 4){
	// Have a tetra file, read it...
	ifstream tetrafile(argv[2]);
	while(tetrafile){
	    int i1, i2, i3, i4;
	    tetrafile >> i1 >> i2 >> i3 >> i4;
	    if(tetrafile){
		mesh->elems.add(new Element(mesh.get_rep(),
					    i1-1, i2-1, i3-1, i4-1));
	    }
	}
    }
    TextPiostream stream(argv[argc-1], Piostream::Write);
    Pio(stream, mesh);
    return 0;
}
