
/*
 *   Mesher.cc: Read pts/tetra files and dump a tesselated Mesh
 *
 *  Written by:
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Classlib/String.h>
#include <Datatypes/Mesh.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>
#include <Classlib/Array1.h>
#include <iostream.h>
#include <fstream.h>
#include <stdio.h>
#include <stdlib.h>

void read_pts(const MeshHandle& mesh, char* infile);
void write_tetras(const MeshHandle& mesh, char* outfile);
void tess(const MeshHandle& mesh);

int main(int argc, char **argv)
{
    if (argc != 3) {
	cerr << "usage: " << argv[0] << " pts_file tetra_file\n";
	exit(0);
    }
    MeshHandle mesh=new Mesh;
    read_pts(mesh, argv[1]);
    tess(mesh);
    write_tetras(mesh, argv[2]);
    return 0;
}

void read_pts(const MeshHandle& mesh, char* infile)
{
    ifstream ptsfile(infile);
    if(!ptsfile){
	cerr << "Cannot open pts file: " << infile << endl;
	exit(-1);
    }
#if 0
    int n;
    ptsfile >> n;
    cerr << "nnodes=" << n << endl;
#endif
    while(ptsfile){
	double x,y,z;
	ptsfile >> x >> y >> z;
	if(ptsfile){
	    mesh->nodes.add(NodeHandle(new Node(Point(x,y,z))));
	}
	if(mesh->nodes.size()%500 == 0){
	    cerr << mesh->nodes.size() << " nodes read\r";
	}
    }
    cerr << endl;
    cerr << "Read " << mesh->nodes.size() << " nodes from " << infile << endl;
}

void write_tetras(const MeshHandle& mesh, char* outfile)
{
    ofstream tfile(outfile);
    if(!tfile){
	cerr << "Cannot open tetra file: " << outfile << endl;
	exit(-1);
    }
    int n=mesh->elems.size();
    tfile << n << endl;
    for(int i=0;i<n;i++){
	for(int j=0;j<4;j++){
	    tfile << mesh->elems[i]->n[j] << " ";
	}
	tfile << endl;
    }
}

void tess(const MeshHandle& mesh)
{
    BBox bbox;
    int nn=mesh->nodes.size();
    for(int ii=0;ii<nn;ii++){
	Point p(mesh->nodes[ii]->p);
	bbox.extend(p);
    }
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

    Element* el1=new Element(mesh.get_rep(), nn+0, nn+1, nn+4, nn+3);
    Element* el2=new Element(mesh.get_rep(), nn+2, nn+1, nn+3, nn+6);
    Element* el3=new Element(mesh.get_rep(), nn+7, nn+3, nn+6, nn+4);
    Element* el4=new Element(mesh.get_rep(), nn+5, nn+6, nn+4, nn+1);
    Element* el5=new Element(mesh.get_rep(), nn+1, nn+3, nn+4, nn+6);
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

    for(int node=0;node<nn;node++){
//	cerr << "Adding node " << node << " " << mesh->nodes[node]->p << endl;
	if(!mesh->insert_delaunay(node, 0)){
	    cerr << "Mesher failed!\n";
	    exit(-1);
	}
	if((node+1)%500 == 0){
	    mesh->pack_elems();
	    cerr << node+1 << " nodes meshed (" << mesh->elems.size() << " elements)\r";
	}
    }
    cerr << endl;
    cerr << "Performing cleanup...\n";
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
    for(int i=0;i<mesh->elems.size();i++){
	vol+=mesh->elems[i]->volume();
    }
    cerr << "Total volume: " << vol << endl;
}
