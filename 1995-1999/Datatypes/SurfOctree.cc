/*
 *  SurfOctree.cc: Unstructured SurfOctreees
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Classlib/TrivialAllocator.h>
#include <Datatypes/ScalarFieldRGint.h>
#include <Datatypes/SurfOctree.h>
#include <Malloc/Allocator.h>
#include <iostream.h>
#include <fstream.h>

static TrivialAllocator SurfOctree_alloc(sizeof(SurfOctree));

static Persistent* make_SurfOctreeTop()
{
    return scinew SurfOctreeTop;
}

PersistentTypeID SurfOctreeTop::type_id("SurfOctreeTop", "Datatype", make_SurfOctreeTop);

SurfOctreeTop::SurfOctreeTop() 
: nx(0), ny(0), nz(0)
{
    tree=0;
}

SurfOctreeTop::SurfOctreeTop(ScalarFieldRGint* sf)
{
    nx=sf->nx;
    ny=sf->ny;
    nz=sf->nz;
    Point min, max;
    sf->get_bounds(min,max);
    dv=max-min;
    dv.x(dv.x()/(nx-1));
    dv.y(dv.y()/(ny-1));
    dv.z(dv.z()/(nz-1));
    Vector half_dv=dv/2.0;
    tree = new SurfOctree(nx, ny, nz, min-half_dv, max+half_dv, sf, 0, 0, 0);
}

SurfOctreeTop::SurfOctreeTop(const SurfOctreeTop& copy)
: tree(copy.tree), nx(copy.nx), ny(copy.ny), nz(copy.nz), min(copy.min),
  max(copy.max), dv(copy.dv)
{
}

SurfOctreeTop* SurfOctreeTop::clone() {
    return scinew SurfOctreeTop(*this);
}

SurfOctreeTop::~SurfOctreeTop() {
    if (tree) delete tree;
}

void SurfOctreeTop::print() {
    tree->print(0,0,0);
}

#define SurfOctreeTop_VERSION 1

void SurfOctreeTop::io(Piostream& stream)
{
    stream.begin_class("SurfOctree", SurfOctreeTop_VERSION);
    Pio(stream, nx);
    Pio(stream, ny);
    Pio(stream, nz);
    Pio(stream, min);
    Pio(stream, max);
    Pio(stream, dv);
    Pio(stream, tree);
    stream.end_class();
}

void* SurfOctree::operator new(size_t)
{
    return SurfOctree_alloc.alloc();
}

void SurfOctree::operator delete(void* rp, size_t)
{
    SurfOctree_alloc.free(rp);
}

// Keep the majority on the Front/Down/Left side
//   (rather than the Back/Up/Right side)

SurfOctree::SurfOctree() {
    for (int i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int k=0; k<2; k++) {
		child[i][j][k]=0;
	    }
	}
    }
}

void SurfOctree::print(int x, int y, int z) {
    cerr << "Tree ["<<x<<", "<<y<<", "<<z<<"] - [";
    cerr << x+nx << ", "<<y+ny<<", "<<z+nz<<"]  Edges: "<< (int)(edges);
    cerr << " Materials: ";
    for (int m=0; m<matl.size(); m++)
	cerr << matl[m] << " ";
    cerr << endl;
    for (int i=0; i<2; i++)
	for (int j=0; j<2; j++)
	    for (int k=0; k<2; k++)
		child[i][j][k]->print(x+(nx+1)*i/2,y+(ny+1)*j/2,z+(nz+1)*k/2);
}

Array1<int>* SurfOctree::propagate_up_materials() {
    for (int i=0; i<2; i++) 
	for (int j=0; j<2; j++) 
	    for (int k=0; k<2; k++) {
		if (child[i][j][k]) {
		    Array1<int>* a = child[i][j][k]->propagate_up_materials();
		    for (int l=0; l<a->size(); l++) {
			int found=0;
			for (int m=0; !found && m<matl.size(); m++) {
			    if (matl[m] == (*a)[l]) found = 1;
			}
			if (!found) matl.add((*a)[l]);
		    }
		}
	    }
    return &(matl);
}

SurfOctree::SurfOctree(int nx, int ny, int nz, const Point &min, 
		       const Point &max, 
		       ScalarFieldRGint* sf, int minx, int miny, int minz)
: nx(nx), ny(ny), nz(nz)
{
    for (int i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int k=0; k<2; k++) {
		child[i][j][k]=0;
	    }
	}
    }

    if (nx<1 || ny<1 || nz<1) {
	cerr << "Shouldn't have 0 for x, y or z dimension!\n";
	return;
    }

    Point mid(min);
    if (nx>1)
	mid.x(min.x() + (max.x()-min.x())*((nx/2)/(nx-1.)));
    if (ny>1)
	mid.y(min.y() + (max.y()-min.y())*((ny/2)/(ny-1.)));
    if (nz>1)
	mid.z(min.z() + (max.z()-min.z())*((nz/2)/(nz-1.)));

    int bits=((nz>1)<<2)+((ny>1)<<1)+(nx>1);

    int x1=nx-nx/2;
    int x2=nx/2;
    int y1=ny-ny/2;
    int y2=ny/2;
    int z1=nz-nz/2;
    int z2=nz/2;

    switch (bits) {
    case 0:     // NADA
	if (sf) {
	    matl.add(sf->grid(minx, miny, minz));
	}
	break;
    case 1:     // Just X subdivides
        child[0][0][0] = new SurfOctree(x1, 1, 1,
					min, 
					Point(mid.x(), max.y(), max.z()), 
					sf, minx, miny, minz);
        child[0][0][1] = new SurfOctree(x2, 1, 1,
					Point(mid.x(), min.y(), min.z()),
					max, 
					sf, minx+x1, miny, minz);
        break;
    case 2:     // Just Y subdivides
        child[0][0][0] = new SurfOctree(1, y1, 1,
					min,
					Point(max.x(), mid.y(), max.z()), 
					sf, minx, miny, minz);
        child[0][1][0] = new SurfOctree(1, y2, 1,
					Point(min.x(), mid.y(), min.z()),
					max, 
					sf, minx, miny+y1, minz);
        break;
    case 4:     // Just Z subdivides    
        child[0][0][0] = new SurfOctree(1, 1, z1,
					min,
					Point(max.x(), max.y(), mid.z()), 
					sf, minx, miny, minz);
        child[1][0][0] = new SurfOctree(1, 1, z2,
					Point(min.x(), min.y(), mid.z()),
					max, 
					sf, minx, miny, minz+z1);
        break;
    case 3:     // X and Y subdivide    
        child[0][0][0] = new SurfOctree(x1, y1, 1,
					min,
					Point(mid.x(), mid.y(), max.z()), 
					sf, minx, miny, minz);
        child[0][0][1] = new SurfOctree(x2, y1, 1,
					Point(mid.x(), min.y(), min.z()),
					Point(max.x(), mid.y(), max.z()), 
					sf, minx+x1, miny, minz);
        child[0][1][0] = new SurfOctree(x1, y2, 1,
					Point(min.x(), mid.y(), min.z()),
					Point(mid.x(), max.y(), max.z()), 
					sf, minx, miny+y1, minz);
        child[0][1][1] = new SurfOctree(x2, y2, 1,
					Point(mid.x(), mid.y(), min.z()),
					Point(max.x(), max.y(), max.z()), 
					sf, minx+x1, miny+y1, minz);
        break;
    case 5:     // X and Z subdivide
        child[0][0][0] = new SurfOctree(x1, 1, z1,
					min,
					Point(mid.x(), max.y(), mid.z()), 
					sf, minx, miny, minz);
        child[0][0][1] = new SurfOctree(x2, 1, z1,
					Point(mid.x(), min.y(), min.z()),
					Point(max.x(), max.y(), mid.z()), 
					sf, minx+x1, miny, minz);
        child[1][0][0] = new SurfOctree(x1, 1, z2,
					Point(min.x(), min.y(), mid.z()),
					Point(mid.x(), max.y(), max.z()), 
					sf, minx, miny, minz+z1);
        child[1][0][1] = new SurfOctree(x2, 1, z2,
					Point(mid.x(), min.y(), mid.z()),
					Point(max.x(), max.y(), max.z()), 
					sf, minx+x1, miny, minz+z1);
        break;
    case 6:     // Y and Z subdivide
        child[0][0][0] = new SurfOctree(1, y1, z1,
					min,
					Point(max.x(), mid.y(), mid.z()), 
					sf, minx, miny, minz);
        child[0][1][0] = new SurfOctree(1, y2, z1,
					Point(min.x(), mid.y(), min.z()),
					Point(max.x(), max.y(), mid.z()), 
					sf, minx, miny+y1, minz);
        child[1][0][0] = new SurfOctree(1, y1, z2,
					Point(min.x(), min.y(), mid.z()),
					Point(max.x(), mid.y(), max.z()), 
					sf, minx, miny, minz+z1);
        child[1][1][0] = new SurfOctree(1, y2, z2,
					Point(min.x(), mid.y(), mid.z()),
					Point(max.x(), max.y(), max.z()), 
					sf, minx, miny+y1, minz+z1);
        break;
    case 7:     // ALL subdivide
        child[0][0][0] = new SurfOctree(x1, y1, z1,
					min,
					mid, 
					sf, minx, miny, minz);
        child[0][0][1] = new SurfOctree(x2, y1, z1,
					Point(mid.x(), min.y(), min.z()),
					Point(max.x(), mid.y(), mid.z()), 
					sf, minx+x1, miny, minz);
        child[0][1][0] = new SurfOctree(x1, y2, z1,
					Point(min.x(), mid.y(), min.z()),
					Point(mid.x(), max.y(), mid.z()), 
					sf, minx, miny+y1, minz);
        child[0][1][1] = new SurfOctree(x2, y2, z1,
					Point(mid.x(), mid.y(), min.z()),
					Point(max.x(), max.y(), mid.z()), 
					sf, minx+x1, miny+y1, minz);
        child[1][0][0] = new SurfOctree(x1, y1, z2,
					Point(min.x(), min.y(), mid.z()),
					Point(mid.x(), mid.y(), max.z()), 
					sf, minx, miny, minz+z1);
        child[1][0][1] = new SurfOctree(x2, y1, z2,
					Point(mid.x(), min.y(), mid.z()),
					Point(max.x(), mid.y(), max.z()), 
					sf, minx+x1, miny, minz+z1);
        child[1][1][0] = new SurfOctree(x1, y2, z2,
					Point(min.x(), mid.y(), mid.z()),
					Point(mid.x(), max.y(), max.z()), 
					sf, minx, miny+y1, minz+z1);
        child[1][1][1] = new SurfOctree(x2, y2, z2,
					mid,
					max, 
					sf, minx+x1, miny+y1, minz+z1);
        break;
    }
} 

SurfOctree::SurfOctree(const SurfOctree& copy)
: nx(copy.nx), ny(copy.ny), nz(copy.nz),
  edges(copy.edges), matl(copy.matl)
{
    for (int i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int k=0; k<2; k++) {
		child[i][j][k]=new SurfOctree(*(copy.child[i][j][k]));
	    }
	}
    }
}

SurfOctree::~SurfOctree()
{
    for (int i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int k=0; k<2; k++) {
		if (child[i][j][k]) delete child[i][j][k];
	    }
	}
    }
}

#define SurfOctree_VERSION 1
void Pio(Piostream& stream, SurfOctree*& o)
{
    if (stream.reading()) o=new SurfOctree;
    stream.begin_cheap_delim();
    Pio(stream, o->nx);
    Pio(stream, o->ny);
    Pio(stream, o->nz);
    Pio(stream, o->edges);
    Pio(stream, o->matl);
    for (int i=0; i<2; i++)
	for (int j=0; j<2; j++)
	    for (int k=0; k<2; k++)
    Pio(stream, o->child[i][j][k]);
    stream.end_cheap_delim();
}

