/*
 *  CVRTItoMesh.cc: Convert .grad, .tetra and .pot into Dataflow fields
 *
 *  Written by:
 *   Chris Packages/Moulding
 *   Department of Computer Science
 *   University of Utah
 *   May 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Core/Containers/String.h>
#include <Core/Containers/Array1.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Datatypes/TriSurface.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/Mesh.h>
#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>
#include <stdio.h>

using namespace SCIRun;

#ifdef _WIN32
PersistentTypeID VectorField::type_id("VectorField", "Datatype", 0);
#endif

int main(int argc, char **argv)
{
	Mesh* mesh = new Mesh();

	NodeHandle node=0;
	Element* e=0;
	double x,y,z;          // coordinates for 3D points
	int v1,v2,v3,v4;       // vertices of the tetras

	clString basename(argv[1]);
	clString ptsfilename(basename+".pts");
	clString tetrafilename(basename+".tetra");

	ifstream tetrafile(tetrafilename());
	ifstream ptsfile(ptsfilename());
	
	int n =0;
       
	while(!ptsfile.eof()) {
	     // read the points
	   ptsfile >> x;
	   ptsfile >> y;
	   ptsfile >> z;

	   Point p = Point(x,y,z);
	   
	   // cerr << "(" << p.x() << "," << p.y() << ","<< p.z() <<")\n";
	   cerr << n << "\n";
	   n++;
	   
	   node = new Node(p);
	   mesh->nodes.add(node);  // ERROR here!! - if remove line -> no problem
	}
	cerr << "HERE\n";

	int nnodes=mesh->nodes.size();
	if (mesh->nodes[nnodes-1]->p == mesh->nodes[nnodes-2]->p) {
	    mesh->nodes.resize(nnodes-1);
	    nnodes--;
	}
	cerr << "# points = "<<nnodes<<"\n";

	// read the tetras from the .tetra file into the VectorFieldUG
	while(!tetrafile.eof())
	{
		tetrafile >> v1;
		tetrafile >> v2;
		tetrafile >> v3;
		tetrafile >> v4;
		e = new Element(mesh,v1-1,v2-1,v3-1,v4-1);
		mesh->elems.add(e);
	}
	int nelems=mesh->elems.size();
	Element *e1=mesh->elems[nelems-1];
	Element *e2=mesh->elems[nelems-2];

	if (e1->n[0] == e2->n[0] && e1->n[1] == e2->n[1] && 
	    e1->n[2] == e2->n[2] && e1->n[3] == e2->n[3]) {
	    delete mesh->elems[nelems-1];
	    mesh->elems.resize(nelems-1);
	    nelems--;
	}
	cerr << "# elems = "<<nelems<<"\n";
	
	clString meshfilename(basename+".mesh");
	TextPiostream stream(meshfilename, Piostream::Write);

	MeshHandle m = mesh;
	Pio(stream,m);

    return 0;
}


