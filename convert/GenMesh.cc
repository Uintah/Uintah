
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
#include <Geometry/BBox.h>
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
	cerr << "usage: " << argv[0] << " n output_file\n";
	exit(0);
    }
    MeshHandle meshh=new Mesh;
    Mesh* mesh=meshh.get_rep();
    int n=atoi(argv[1]);
    MusilRNG rng;
    BBox bbox;
    bbox.extend(Point(0,0,0));
    bbox.extend(Point(1,1,1));
        double epsilon=.1*bbox.longest_edge();

    // Extend by max-(eps, eps, eps) and min+(eps, eps, eps) to
    // avoid thin/degenerate bounds
    Point max(bbox.max()+Vector(epsilon, epsilon, epsilon));
    Point min(bbox.min()-Vector(epsilon, epsilon, epsilon));

    mesh->nodes.add(new Node(Point(min.x(), min.y(), min.z())));
    mesh->nodes.add(new Node(Point(max.x(), min.y(), min.z())));
    mesh->nodes.add(new Node(Point(max.x(), min.y(), max.z())));
    mesh->nodes.add(new Node(Point(min.x(), min.y(), max.z())));
    mesh->nodes.add(new Node(Point(min.x(), max.y(), min.z())));
    mesh->nodes.add(new Node(Point(max.x(), max.y(), min.z())));
    mesh->nodes.add(new Node(Point(max.x(), max.y(), max.z())));
    mesh->nodes.add(new Node(Point(min.x(), max.y(), max.z())));

    int nn=0;
    Element* el1=new Element(mesh, nn+0, nn+1, nn+4, nn+3);
    Element* el2=new Element(mesh, nn+2, nn+1, nn+3, nn+6);
    Element* el3=new Element(mesh, nn+7, nn+3, nn+6, nn+4);
    Element* el4=new Element(mesh, nn+5, nn+6, nn+4, nn+1);
    Element* el5=new Element(mesh, nn+1, nn+3, nn+4, nn+6);
    el1->faces[0]=4; el1->faces[1]=-1; el1->faces[2]=-1; el1->faces[3]=-1;
    el2->faces[0]=4; el2->faces[1]=-1; el2->faces[2]=-1; el2->faces[3]=-1;
    el3->faces[0]=4; el3->faces[1]=-1; el3->faces[2]=-1; el3->faces[3]=-1;
    el4->faces[0]=4; el4->faces[1]=-1; el4->faces[2]=-1; el4->faces[3]=-1;
    el5->faces[0]=2; el5->faces[1]=3; el5->faces[2]=1; el5->faces[3]=0;
    el1->orient();
    el2->orient();
    el3->orient();
    el4->orient();
    el5->orient();
    mesh->elems.add(el1);
    mesh->elems.add(el2);
    mesh->elems.add(el3);
    mesh->elems.add(el4);
    mesh->elems.add(el5);

    MusilRNG rng2;
    Array1<int> map(n);
    for(int ii=0;ii<n;ii++){
	map[ii]=ii;
    }
#if 0
    for(ii=0;ii<n;ii++){
	int n1=rng2()*n;
	int tmp=map[n1];
	map[n1]=map[ii];
	map[ii]=tmp;
    }
#endif
    for(ii=0;ii<n;ii++){
	cerr << "map[" << ii << "]=" << map[ii] << endl;
    }
    for(int i=0;i<n;i++){
        cerr << i << "/" << n << endl;
	double x=double(map[i])/double(n-1);
	cerr << "x=" << x << endl;
	for(int j=0;j<n;j++){
	    double y=double(map[j])/double(n-1);
	    for(int k=0;k<n;k++){
		double z=double(map[k])/double(n-1);
		double xx=x;
		double yy=y;
		double zz=z;
		if(i>0 && i<n-1)
		    //xx+=rng()*0.3/double(n-1);
		    //xx+=rng()*0.0/double(n-1);
		if(j>0 && j<n-1)
		    //yy+=rng()*0.3/double(n-1);
		    yy+=rng()*0.0/double(n-1);
		if(k>0 && k<n-1)
		    //zz+=rng()*0.3/double(n-1);
		    zz+=rng()*0.0/double(n-1);
		//mesh->nodes.add(NodeHandle(new Node(Point(xx,yy,zz))));
		if(!mesh->insert_delaunay(Point(xx,yy,zz), 0)){
		    cerr << "Mesher upset - point outside of domain...";
		    return;
		}
		// Every 1000 nodes, cleanup the elems array...
	    }
	}
    }
    mesh->compute_neighbors();
    mesh->remove_delaunay(nn, 0);
    mesh->remove_delaunay(nn+1, 0);
    mesh->remove_delaunay(nn+2, 0);
    mesh->remove_delaunay(nn+3, 0);
    mesh->remove_delaunay(nn+4, 0);
    mesh->remove_delaunay(nn+5, 0);
    mesh->remove_delaunay(nn+6, 0);
    mesh->remove_delaunay(nn+7, 0);
    mesh->pack_all();
    double vol=0;
    cerr << "There are " << mesh->elems.size() << " elements" << endl;
    for(i=0;i<mesh->elems.size();i++){
	vol+=mesh->elems[i]->volume();
    }
    cerr << "Total volume: " << vol << endl;

    TextPiostream stream(argv[argc-1], Piostream::Write);
    Pio(stream, meshh);
    return 0;
}
