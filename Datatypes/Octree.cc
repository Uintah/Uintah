/*
 *  Octree.cc: Unstructured Octreees
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Classlib/TrivialAllocator.h>
#include <Datatypes/Octree.h>
#include <Malloc/Allocator.h>
#include <iostream.h>
#include <fstream.h>

static int flag=0;

static TrivialAllocator Octree_alloc(sizeof(Octree));

static Persistent* make_OctreeTop()
{
    return scinew OctreeTop;
}

PersistentTypeID OctreeTop::type_id("Octree", "Datatype", make_OctreeTop);

OctreeTop::OctreeTop() 
: nx(0), ny(0), nz(0)
{
    tree=0;
}

OctreeTop::OctreeTop(int nx, int ny, int nz, const BBox& b)
: nx(nx), ny(ny), nz(nz), vectors(0), scalars(0), tensors(0)
{
    tree = new Octree(nx, ny, nz, b.min(), b.max());
    tree->trunk=1;
    tree->leaf=1;
}

OctreeTop::OctreeTop(const OctreeTop& copy)
: tree(copy.tree), nx(copy.nx), ny(copy.ny), nz(copy.nz), 
  vectors(copy.vectors), scalars(copy.scalars), tensors(copy.tensors)
{
}

OctreeTop* OctreeTop::clone() {
    return scinew OctreeTop(*this);
}

OctreeTop::~OctreeTop() {
    if (tree) delete tree;
}

#define OctreeTop_VERSION 1

void OctreeTop::io(Piostream& stream)
{
    int version=stream.begin_class("Octree", OctreeTop_VERSION);
    Pio(stream, nx);
    Pio(stream, ny);
    Pio(stream, nz);
    Pio(stream, scalars);
    Pio(stream, vectors);
    Pio(stream, tensors);
    Pio(stream, tree);
    stream.end_class();
}

void* Octree::operator new(size_t)
{
    return Octree_alloc.alloc();
}

void Octree::operator delete(void* rp, size_t)
{
    Octree_alloc.free(rp);
}

// Keep the majority on the Front/Down/Left side
//   (rather than the Back/Up/Right side)

Octree::Octree() {
    for (int i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int k=0; k<2; k++) {
		child[i][j][k]=0;
		corner_s[i][j][k]=-321;
		corner_p[i][j][k]=Point(0,0,0);
	    }
	}
    }
}

Octree::Octree(int nx, int ny, int nz, const Point &min, const Point &max)
: nx(nx), ny(ny), nz(nz), last_leaf(0), trunk(0), leaf(0)
{
//    cerr << "Inserting tree (x=" << nx <<", y=" << ny << ", z="<< nz << ")\n";
    double x[2], y[2], z[2];
    x[0]=min.x(); x[1]=max.x(); 
    y[0]=min.y(); y[1]=max.y();
    z[0]=min.z(); z[1]=max.z();
    for (int i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int k=0; k<2; k++) {
		child[i][j][k]=0;
		corner_s[i][j][k]=-321;
		corner_p[k][j][i]=Point(x[i],y[j],z[k]);
	    }
	}
    }

    if (nx<2 || ny<2 || nz<2) {
	cerr << "Shouldn't have 0 or 1 for x, y or z dimension!\n";
	return;
    }

    if (nx==2 && ny==2 && nz==2) {
	last_leaf=1;
    }

    mid=min;
    if (nx != 2) {
	mid.x(min.x() + (max.x()-min.x())*((nx/2)/(nx-1.)));
    } else {
	mid.x((min.x()+max.x())/2.);
    }
    if (ny != 2) {
	mid.y(min.y() + (max.y()-min.y())*((ny/2)/(ny-1.)));
    } else {
	mid.y((min.y()+max.y())/2.);
    }
    if (nz != 2) {
	mid.z(min.z() + (max.z()-min.z())*((nz/2)/(nz-1.)));
    } else {
	mid.z((min.z()+max.z())/2.);
    }


    bits=((nz>2)<<2)+((ny>2)<<1)+(nx>2);

    nx++; ny++; nz++;

    switch (bits) {
    case 0:	// NADA
	break;
    case 1:	// Just X subdivides
	child[0][0][0] = new Octree(nx-nx/2, 2, 2,
				    min, 
				    Point(mid.x(), max.y(), max.z()));
	child[0][0][1] = new Octree(nx/2, 2, 2,
				    Point(mid.x(), min.y(), min.z()),
				    max);
	break;
    case 2:	// Just Y subdivides
	child[0][0][0] = new Octree(2, ny-ny/2, 2,
				    min,
				    Point(max.x(), mid.y(), max.z()));
	child[0][1][0] = new Octree(2, ny/2, 2,
				    Point(min.x(), mid.y(), min.z()),
				    max);
	break;
    case 4:	// Just Z subdivides	
	child[0][0][0] = new Octree(2, 2, nz-nz/2,
				    min,
				    Point(max.x(), max.y(), mid.z()));	
	child[1][0][0] = new Octree(2, 2, nz/2,
				    Point(min.x(), min.y(), mid.z()),
				    max);
	break;
    case 3:	// X and Y subdivide	
	child[0][0][0] = new Octree(nx-nx/2, ny-ny/2, 2,
				    min,
				    Point(mid.x(), mid.y(), max.z()));	
	child[0][0][1] = new Octree(nx/2, ny-ny/2, 2,
				    Point(mid.x(), min.y(), min.z()),
				    Point(max.x(), mid.y(), max.z()));
	child[0][1][0] = new Octree(nx-nx/2, ny/2, 2,
				    Point(min.x(), mid.y(), min.z()),
				    Point(mid.x(), max.y(), max.z()));
	child[0][1][1] = new Octree(nx/2, ny/2, 2,
				    Point(mid.x(), mid.y(), min.z()),
				    Point(max.x(), max.y(), max.z()));
	break;
    case 5:	// X and Z subdivide
	child[0][0][0] = new Octree(nx-nx/2, 2, nz-nz/2,
				    min,
				    Point(mid.x(), max.y(), mid.z()));
	child[0][0][1] = new Octree(nx/2, 2, nz-nz/2,
				    Point(mid.x(), min.y(), min.z()),
				    Point(max.x(), max.y(), mid.z()));
	child[1][0][0] = new Octree(nx-nx/2, 2, nz/2,
				    Point(min.x(), min.y(), mid.z()),
				    Point(mid.x(), max.y(), max.z()));
	child[1][0][1] = new Octree(nx/2, 2, nz/2,
				    Point(mid.x(), min.y(), mid.z()),
				    Point(max.x(), max.y(), max.z()));
	break;
    case 6:	// Y and Z subdivide
	child[0][0][0] = new Octree(2, ny-ny/2, nz-nz/2,
				    min,
				    Point(max.x(), mid.y(), mid.z()));
	child[0][1][0] = new Octree(2, ny/2, nz-nz/2,
				    Point(min.x(), mid.y(), min.z()),
				    Point(max.x(), max.y(), mid.z()));
	child[1][0][0] = new Octree(2, ny-ny/2, nz/2,
				    Point(min.x(), min.y(), mid.z()),
				    Point(max.x(), mid.y(), max.z()));
	child[1][1][0] = new Octree(2, ny/2, nz/2,
				    Point(min.x(), mid.y(), mid.z()),
				    Point(max.x(), max.y(), max.z()));
	break;
    case 7:	// ALL subdivide
	child[0][0][0] = new Octree(nx-nx/2, ny-ny/2, nz-nz/2,
				    min,
				    mid);
	child[0][0][1] = new Octree(nx/2, ny-ny/2, nz-nz/2,
				    Point(mid.x(), min.y(), min.z()),
				    Point(max.x(), mid.y(), mid.z()));
	child[0][1][0] = new Octree(nx-nx/2, ny/2, nz-nz/2,
				    Point(min.x(), mid.y(), min.z()),
				    Point(mid.x(), max.y(), mid.z()));
	child[0][1][1] = new Octree(nx/2, ny/2, nz-nz/2,
				    Point(mid.x(), mid.y(), min.z()),
				    Point(max.x(), max.y(), mid.z()));
	child[1][0][0] = new Octree(nx-nx/2, ny-ny/2, nz/2,
				    Point(min.x(), min.y(), mid.z()),
				    Point(mid.x(), mid.y(), max.z()));
	child[1][0][1] = new Octree(nx/2, ny-ny/2, nz/2,
				    Point(mid.x(), min.y(), mid.z()),
				    Point(max.x(), mid.y(), max.z()));
	child[1][1][0] = new Octree(nx-nx/2, ny/2, nz/2,
				    Point(min.x(), mid.y(), mid.z()),
				    Point(mid.x(), max.y(), max.z()));
	child[1][1][1] = new Octree(nx/2, ny/2, nz/2,
				    mid,
				    max);
	break;
    }

    nx--; ny--; nz--;

}

Octree::Octree(const Octree& copy)
: nx(copy.nx), ny(copy.ny), nz(copy.nz), mid(copy.mid),
  bits(copy.bits), avg_vec(copy.avg_vec), min_sc(copy.min_sc), 
  max_sc(copy.max_sc), avg_sc(copy.avg_sc)
{
    for (int i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int k=0; k<2; k++) {
		child[i][j][k]=(Octree*)0;
	    }
	}
    }
}

Octree::~Octree()
{
    for (int i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int k=0; k<2; k++) {
		if (child[i][j][k]) delete child[i][j][k];
	    }
	}
    }
}

void Octree::prune() {
    if (leaf) leaf=0;
    else {
	for (int i=0; i<2; i++) {
	    for (int j=0; j<2; j++) {
		for (int k=0; k<2; k++) {
		    if (child[i][j][k]) child[i][j][k]->prune();
		}
	    }
	}
    }
}

Octree *Octree::which_child(const Point&p) {
    Point max(corner_p[1][1][1]);
    Point min(corner_p[0][0][0]);

    if (!(p.x()<=max.x() && p.y()<=max.y() && p.z()<=max.z() &&
	  p.x()>=min.x() && p.y()>=min.y() && p.z()>=min.z())) return 0;
    if (nx==ny==nz==1) return 0;
    int x,y,z;
    x=y=z=0;
    if (nx && (p.x()>mid.x())) x=1;
    if (ny && (p.y()>mid.y())) y=1;
    if (nz && (p.z()>mid.z())) z=1;
    int bb=(z << 2) + (y << 1) + x;
    switch (bb) {
    case 0: return child[0][0][0];
	case 1: return child[0][0][1];
	case 2: return child[0][1][0];
	case 3: return child[0][1][1];
	case 4: return child[1][0][0];
	case 5: return child[1][0][1];
	case 6: return child[1][1][0];
	case 7: return child[1][1][1];
	}
    return (Octree*)0;
}

Octree *Octree::index_child(int bb) {
    switch (bb) {
    case 0: return child[0][0][0];
    case 1: return child[0][0][1];
    case 2: return child[0][1][0];
    case 3: return child[0][1][1];
    case 4: return child[1][0][0];
    case 5: return child[1][0][1];
    case 6: return child[1][1][0];
    case 7: return child[1][1][1];
    }
    return (Octree*)0;
}

void Octree::insert_scalar_field(ScalarFieldRG* sf) {
    if (sf->nx != nx || sf->ny != ny || sf->nz != nz) {
	cerr << "Octree and ScalarField dimensions don't match!\n";
	return;
    }
    for (int i=0; i<sf->nx; i++) {
	for (int j=0; j<sf->ny; j++) {
	    for (int k=0; k<sf->nz; k++) {
		set_scalar(i,j,k,sf->grid(i,j,k));
	    }
	}
    }
}

void Octree::insert_vector_field(VectorFieldRG* vf) {
    if (vf->nx != nx || vf->ny != ny || vf->nz != nz) {
	cerr << "Octree and ScalarField dimensions don't match!\n";
	return;
    }
    for (int i=0; i<vf->nx; i++) {
	for (int j=0; j<vf->ny; j++) {
	    for (int k=0; k<vf->nz; k++) {
		set_vector(i,j,k,vf->grid(i,j,k));
	    }
	}
    }
}

double Octree::set_and_return_max_scalar() {
    if (last_leaf) {
	max_sc=corner_s[0][0][0];
	for (int i=0; i<2; i++) {
	    for (int j=0; j<2; j++) {
		for (int k=0; k<2; k++) {
		    if (corner_s[i][j][k]>max_sc) max_sc=corner_s[i][j][k];
		}
	    }
	}
	return max_sc;
    } else {
	double max_value;
	int have_value=0;
	for (int i=0; i<2; i++) {
	    for (int j=0; j<2; j++) {
		for (int k=0; k<2; k++) {
		    Octree *kid = child[i][j][k];
		    if (kid) {
			double temp=kid->set_and_return_max_scalar();
			if (!have_value || temp>max_value) {
			    have_value=1;
			    max_value=temp;
			}
		    }
		}
	    }
	}
	if (have_value) {
	    max_sc=max_value;
	    return max_sc;
	} else {
	    cerr << "Lost max value coming up!\b";
	    return 0;
	}
    }
}

double Octree::set_and_return_min_scalar() {
    if (last_leaf) {
	min_sc=corner_s[0][0][0];
	for (int i=0; i<2; i++) {
	    for (int j=0; j<2; j++) {
		for (int k=0; k<2; k++) {
		    if (corner_s[i][j][k]<min_sc) min_sc=corner_s[i][j][k];
		}
	    }
	}
	return min_sc;
    } else {
	double min_value;
	int have_value=0;
	for (int i=0; i<2; i++) {
	    for (int j=0; j<2; j++) {
		for (int k=0; k<2; k++) {
		    Octree *kid = child[i][j][k];
		    if (kid) {
			double temp=kid->set_and_return_min_scalar();
			if (!have_value || temp<min_value) {
			    have_value=1;
			    min_value=temp;
			}
		    }
		}
	    }
	}
	if (have_value) {
	    min_sc=min_value;
	    return min_sc;
	} else {
	    cerr << "Lost min value coming up!\b";
	    return 0;
	}
    }
}

double Octree::set_and_return_avg_scalar() {
    if (last_leaf) {
        avg_sc=0;
	for (int i=0; i<2; i++) {
	    for (int j=0; j<2; j++) {
		for (int k=0; k<2; k++) {
		    avg_sc+=corner_s[i][j][k];
		}
	    }
	}
	avg_sc/=8.;
	return avg_sc;
    } else {
	double avg_value=0;
	int hits=0;
	for (int i=0; i<2; i++) {
	    for (int j=0; j<2; j++) {
		for (int k=0; k<2; k++) {
		    Octree *kid = child[i][j][k];
		    if (kid) {
			avg_value += kid->set_and_return_avg_scalar();
			hits += 1;
		    }
		}
	    }
	}
	if (hits) {
	    avg_sc = avg_value/hits;
	    return avg_sc;
	} else {
	    cerr << "Lost avg value coming up!\b";
	    return 0;
	}
    }
}

Vector Octree::set_and_return_avg_vector() {
    if (last_leaf) {
        avg_vec=Vector(0,0,0);
	for (int i=0; i<2; i++) {
	    for (int j=0; j<2; j++) {
		for (int k=0; k<2; k++) {
		    avg_vec+=corner_v[i][j][k];
		}
	    }
	}
	avg_vec.x(avg_vec.x()/8.);
	avg_vec.y(avg_vec.y()/8.);
	avg_vec.z(avg_vec.z()/8.);
	return avg_vec;
    } else {
	Vector avrg(0,0,0);
	int hits=0;
	for (int i=0; i<2; i++) {
	    for (int j=0; j<2; j++) {
		for (int k=0; k<2; k++) {
		    Octree *kid = child[i][j][k];
		    if (kid) {
			avrg = avrg + kid->set_and_return_avg_vector();
			hits += 1;
		    }
		}
	    }
	}
	if (hits) {
	    avg_vec = Vector(avrg.x()/hits, avrg.y()/hits, avrg.z()/hits);
	    return avg_vec;
	} else {
	    cerr << "Lost avg vecor coming up!\b";
	    return Vector(0,0,0);
	}
    }
}

double Octree::set_and_return_corner_scalar(int i, int j, int k) {
    if (last_leaf)
	return corner_s[i][j][k];
    int got_it=0;
    for (int ii=0; ii<2; ii++) {
	for (int jj=0; jj<2; jj++) {
	    for (int kk=0; kk<2; kk++) {
		Octree *kid = child[ii][jj][kk];
		if (kid) {
		    if (i==ii && j==jj && k==kk) {
			got_it=1;
			corner_s[i][j][k] = 
			    kid->set_and_return_corner_scalar(i,j,k);
		    } else {
			kid->set_and_return_corner_scalar(i,j,k);
		    }
		}
	    }
	}
    }
    if (!got_it) {
	corner_s[i][j][k]=child[(bits/4)&&i][((bits&2)/2)&&j][(bits&1)&&k]->
	    corner_s[i][j][k];
    }	
    return corner_s[i][j][k];
}

Vector Octree::set_and_return_corner_vector(int i, int j, int k) {
    if (last_leaf) {
	return corner_v[i][j][k];
    } else {
	int got_it=0;
	for (int ii=0; ii<2; ii++) {
	    for (int jj=0; jj<2; jj++) {
		for (int kk=0; kk<2; kk++) {
		    Octree *kid = child[ii][jj][kk];
		    if (kid) {
			if (i==ii && j==jj && k==kk) {
			    got_it=1;
			    corner_v[i][j][k] = 
				kid->set_and_return_corner_vector(i,j,k);
			} else {
			    kid->set_and_return_corner_vector(i,j,k);
			}
		    }
		}
	    }
	}
    }
    return corner_v[i][j][k];
}

void Octree::print_tree(int level) {
    if (level==0) {
	cerr <<"Printing Octree...\n\n";
    }
    int i;
    for (i=0; i<level; i++) {
	cerr << "  ";
    }
    cerr << "bits:"<<bits<<"  nx:"<<nx<<"  ny:"<<ny<<"  nz:"<<nz;
    cerr << "  leaf:"<<leaf<< "  last_leaf:"<<last_leaf;
    cerr << "  trunk:"<<trunk <<"\n";
    for (i=0; i<level; i++) {
	cerr << "  ";
    }
    cerr << "min_sc:"<<min_sc<<"  max_sc:"<<max_sc<<"  avg_sc:"<<avg_sc;
    cerr << "  avg_vec:"<<avg_vec<<"\n";
    for (i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int ii=0; ii<level; ii++) {
		cerr << "  ";
	    }
	    for (int k=0; k<2; k++) {
		cerr << i*4+j*2+k <<": " << corner_s[i][j][k] <<"  ";
	    }
	    cerr << "\n";
	}
    }
    for (i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int ii=0; ii<level; ii++) {
		cerr << "  ";
	    }
	    for (int k=0; k<2; k++) {
		cerr << i*4+j*2+k <<": " << corner_p[i][j][k] <<"  ";
	    }
	    cerr << "\n";
	}
    }
    for (i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int k=0; k<2; k++) {
		for (int ii=0; ii<level; ii++) {
		    cerr << "  ";
		}
		if (child[i][j][k]) {
		    cerr << "Child "<< i*4+j*2+k << "\n";
		    child[i][j][k]->print_tree(level+1);
		} else {
		    cerr << "Child "<< i*4+j*2+k << " doesn't exist\n";
		}
	    }
	}
    }
    if (level==0) {
	cerr <<"\n\n";
    }
}

void Octree::build_scalar_tree() {
    set_and_return_max_scalar();
    set_and_return_min_scalar();
    set_and_return_avg_scalar();
    for (int i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int k=0; k<2; k++) {
		set_and_return_corner_scalar(i,j,k);
	    }
	}
    }
//    print_tree(0);
}

void Octree::build_vector_tree() {
    set_and_return_avg_vector();
}

// if we're at the bottom, ignore return 0
// otherwise if we're a "leaf" make our kids leaves and return 1
//	     otherwise send it to the right kid
int Octree::push_level(const Point& p) {
    Point max(corner_p[1][1][1]);
    Point min(corner_p[0][0][0]);
    if (p.x()<=max.x() && p.y()<=max.y() && p.z()<=max.z() &&
	p.x()>=min.x() && p.y()>=min.y() && p.z()>=min.z()) {
	if (last_leaf) return 0;
	if (leaf) {
	    for (int i=0; i<2; i++) {
		for (int j=0; j<2; j++) {
		    for (int k=0; k<2; k++) {
			if (child[i][j][k])
			    child[i][j][k]->leaf=1;
		    }
		}
		leaf=0;
	    }
	} else {
	    Octree* kid2=which_child(p);
	    if (kid2)
		return(kid2->push_level(p));
	    else {
		cerr << "Shouldn't have ever gotten here!\n";
		return 0;
	    }
	}
    }
    return 1;
}


// if we're at the top, ignore return 0
// otherwise if our containing child is a leaf, make us a leaf, and
// kill all children leaves
//	     otherwise send it to the right kid
int Octree::pop_level(const Point& p) {
    Octree* kid=which_child(p);
    if (!child) return 0;
    if (kid->leaf) {
	for (int i=0; i<2; i++) {
	    for (int j=0; j<2; j++) {
		for (int k=0; k<2; k++) {
		    Octree *kid2 = child[i][j][k];
		    if (kid2)
			kid2->prune();
		}
	    }
	}
	leaf=1;
	return 1;
    }
    return(kid->pop_level(p));
}

void Octree::push_all_levels() {	
    if (last_leaf) return;
    for (int i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int k=0; k<2; k++) {
		Octree *kid = child[i][j][k];
		if (kid) {
		    if (leaf) kid->leaf=1;
		    else kid->push_all_levels();
		}
	    }
	}
    }
    if (leaf) leaf=0;
}

void Octree::pop_all_levels() {
    if (leaf || last_leaf) return;
    int parent;
    int i;
    for (parent=i=0; i<2; i++) {
	    for (int j=0; j<2; j++) {
		for (int k=0; k<2; k++) {
		    Octree *kid = child[i][j][k];
		    if (kid) {
			if (kid->leaf) {
			    parent=1;
			}
		    }
		}
	    }
	}
    for (i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int k=0; k<2; k++) {
		Octree *kid = child[i][j][k];
		if (kid) {
		    if (parent) {
			kid->prune();
		    } else {
			kid->pop_all_levels();
		    }
		}
	    }
	}
    }
    if (parent) leaf=1;
}

void Octree::top_level() {
    leaf=1;
    for (int i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int k=0; k<2; k++) {
		if (child[i][j][k])
		    child[i][j][k]->prune();
		
	    }
	}
    }
}

void Octree::bottom_level() {
    if (last_leaf) {
	leaf=1;
	return;
    }
    leaf=0;
    for (int i=0; i<2; i++) {
	for (int j=0; j<2; j++) {
	    for (int k=0; k<2; k++) {
		if (child[i][j][k])
		    child[i][j][k]->bottom_level();
	    }
	}
    }
}

double Octree::get_scalar(int x, int y, int z) {
    if (x>=nx || y>=ny || z>=nz) {
	cerr << "Bogus x, y, z indices!\n";
	return 0;
    }
    if (leaf) {
	return corner_s[z][y][x];
    } else {
	if (last_leaf) {
	    cerr << "Got to the bottom without finding a leaf!\n";
	    return 0;
	} else {
	    int ix=0;
	    int iy=0;
	    int iz=0;
	    if (x>((nx+1)-(nx+1)/2)) {x-=((nx+1)-(nx+1)/2); ix=1;}
	    if (y>((ny+1)-(ny+1)/2)) {y-=((ny+1)-(ny+1)/2); iy=1;}
	    if (z>((nz+1)-(nz+1)/2)) {z-=((nz+1)-(nz+1)/2); iz=1;}
	    Octree* kid=child[iz][iy][ix];
	    if (kid) return kid->get_scalar(x,y,z);
	    else {
		cerr << "Child node doesn't exist!\n";
		return 0;
	    }
	}
    }
}

Vector Octree::get_vector(int x, int y, int z) {
    if (x>=nx || y>=ny || z>=nz) {
	cerr << "Bogus x, y, z indices!\n";
	return Vector(0,0,0);
    }
    if (leaf) {
	return corner_v[z][y][x];
    } else {
	if (last_leaf) {
	    cerr << "Got to the bottom without finding a leaf!\n";
	    return Vector(0,0,0);
	} else {
	    int ix=0;
	    int iy=0;
	    int iz=0;
	    if (x>((nx+1)-(nx+1)/2)) {x-=((nx+1)-(nx+1)/2); ix=1;}
	    if (y>((ny+1)-(ny+1)/2)) {y-=((ny+1)-(ny+1)/2); iy=1;}
	    if (z>((nz+1)-(nz+1)/2)) {z-=((nz+1)-(nz+1)/2); iz=1;}
	    Octree* kid=child[iz][iy][ix];
	    if (kid) return kid->get_vector(x,y,z);
	    else {
		cerr << "Child node doesn't exist!\n";
		return Vector(0,0,0);
	    }
	}
    }
}

Point Octree::get_midpoint(int x, int y, int z) {
    if (x>=nx || y>=ny || z>=nz) {
	cerr << "Bogus x, y, z indices!\n";
	return Point(0,0,0);
    }
    if (leaf) {
	return mid;
    } else {
	if (last_leaf) {
	    cerr << "Got to the bottom without finding a leaf!\n";
	    return Point(0,0,0);
	} else {
	    int ix=0;
	    int iy=0;
	    int iz=0;
	    if (x>((nx+1)-(nx+1)/2)) {x-=((nx+1)-(nx+1)/2); ix=1;}
	    if (y>((ny+1)-(ny+1)/2)) {y-=((ny+1)-(ny+1)/2); iy=1;}
	    if (z>((nz+1)-(nz+1)/2)) {z-=((nz+1)-(nz+1)/2); iz=1;}
	    Octree* kid=child[iz][iy][ix];
	    if (kid) return kid->get_midpoint(x,y,z);
	    else {
		cerr << "Child node doesn't exist!\n";
		return Point(0,0,0);
	    }
	}
    }
}

double Octree::get_scalar(const Point &p) {
    if (leaf) {
	return avg_sc;
    } else {
	if (last_leaf) {
	    cerr << "Got to the bottom without finding a leaf!\n";
	    return 0;
	} else {
	    Octree* kid=which_child(p);
	    if (kid) return kid->get_scalar(p);
	    else {
		cerr << "Child node doesn't exist!\n";
		return 0;
	    }
	}
    }
}

Vector Octree::get_vector(const Point &p) {
    if (leaf) {
	return avg_vec;
    } else {
	if (last_leaf) {
	    cerr << "Got to the bottom without finding a leaf!\n";
	    return Vector(0,0,0);
	} else {
	    Octree* kid=which_child(p);
	    if (kid) return kid->get_vector(p);
	    else {
		cerr << "Child node doesn't exist!\n";
		return Vector(0,0,0);
	    }
	}
    }
}

int Octree::set_scalar(int x, int y, int z, double val) {
    if (x>=nx || y>=ny || z>=nz) {
	cerr << "Bogus indices: nx:"<<nx<<"  x:" <<x<<"  ny:"<<ny;
	cerr << "  y:"<<y<<"  nz:"<<nz<<"  z:"<<z<<"\n";
	flag=1;
	return 0;
    }
    if (last_leaf) {
	if (nx==2 && ny==2 && nz==2) {
	    corner_s[z][y][x] = val;
	    return 1;
	} else {
	    cerr << "Got to last level, but not ready to insert!  nx="<< nx
		<< "  ny="<< ny << "  nz=" << nz <<"\n";
	    return 0;
	}
    }
    int ix, iy, iz, xx[2], yy[2], zz[2], xs[2], ys[2], zs[2];
    int midx=nx/2;
    int midy=ny/2;
    int midz=nz/2;
    if (nx==2) midx=nx;
    if (ny==2) midy=ny;
    if (nz==2) midz=nz;
    if (x>midx) 	{ix=1; xs[0]=1; xx[0]=x-midx;}
    else if (x==midx) 	{ix=2; xs[0]=0; xx[0]=midx; xs[1]=1; xx[1]=0;}
    else 		{ix=1; xs[0]=0; xx[0]=x;}
    if (y>midy) 	{iy=1; ys[0]=1; yy[0]=y-midy;}
    else if (y==midy) 	{iy=2; ys[0]=0; yy[0]=midy; ys[1]=1; yy[1]=0;}
    else 		{iy=1; ys[0]=0; yy[0]=y;}
    if (z>midz) 	{iz=1; zs[0]=1; zz[0]=z-midz;}
    else if (z==midz) 	{iz=2; zs[0]=0; zz[0]=midz; zs[1]=1; zz[1]=0;}
    else 		{iz=1; zs[0]=0; zz[0]=z;}
    for (int ii=0; ii<ix; ii++) {
	for (int jj=0; jj<iy; jj++) {
	    for (int kk=0; kk<iz; kk++) {
		Octree*kid=child[zs[kk]][ys[jj]][xs[ii]];
		if (kid) {
		    kid->set_scalar(xx[ii], yy[jj], zz[kk], val);
		    if (flag) {
			flag=0;
			cerr << "ix=" <<ix <<"  iy="<<iy<<"  iz="<<iz<<"\n";
			cerr << "xs[0]=" <<xs[0]<<"  xs[1]="<<xs[1]<<"\n";
			cerr << "ys[0]=" <<ys[0]<<"  ys[1]="<<ys[1]<<"\n";
			cerr << "zs[0]=" <<zs[0]<<"  zs[1]="<<zs[1]<<"\n";
			cerr << "xx[0]=" <<xx[0]<<"  xx[1]="<<xx[1]<<"\n";
			cerr << "yy[0]=" <<yy[0]<<"  yy[1]="<<yy[1]<<"\n";
			cerr << "zz[0]=" <<zz[0]<<"  zz[1]="<<zz[1]<<"\n";
			cerr << "ii=" <<ii <<"  jj="<<jj<<"  kk="<<kk<<"\n";
			cerr << "nx=" <<nx <<"  ny="<<ny<<"  nz="<<nz<<"\n";
		    }
		} else {
		    cerr << "Error traversing octree\n"; 
		}
	    }
	}
    }
    return 0;
}

int Octree::set_vector(int x, int y, int z, const Vector &v) {
    if (x>=nx || y>=ny || z>=nz) {
	cerr << "Bogus x, y, z indices!\n";
	return 0;
    }
    if (last_leaf) {
	if (nx==2 && ny==2 && nz==2) {
	    corner_v[z][y][x] = v;
	    return 1;
	} else {
	    cerr << "Got to last level, but for not ready to insert!\n";
	    return 0;
	}
    }
    int ix, iy, iz, xx[2], yy[2], zz[2], xs[2], ys[2], zs[2];
    int midx=nx/2;
    int midy=ny/2;
    int midz=nz/2;
    if (x>midx) 	{ix=1; xs[0]=1; xx[0]=x-midx;}
    else if (x==midx) 	{ix=2; xs[0]=0; xx[0]=midx; xs[1]=1; xx[1]=0;}
    else 		{ix=1; xs[0]=0; xx[0]=x;}
    if (y>midy) 	{iy=1; ys[0]=1; yy[0]=y-midy;}
    else if (y==midy) 	{iy=2; ys[0]=0; yy[0]=midy; ys[1]=1; yy[1]=0;}
    else 		{iy=1; ys[0]=0; yy[0]=y;}
    if (z>midz) 	{iz=1; zs[0]=1; zz[0]=z-midz;}
    else if (z==midz) 	{iz=2; zs[0]=0; zz[0]=midz; zs[1]=1; zz[1]=0;}
    else 		{iz=1; zs[0]=0; zz[0]=z;}
    for (int ii=0; ii<ix; ii++) {
	for (int jj=0; jj<iy; jj++) {
	    for (int kk=0; kk<iz; kk++) {
		Octree*kid=child[zs[kk]][ys[jj]][xs[ii]];
		if (kid)
		    kid->set_vector(xx[ii], yy[jj], zz[kk], v);
		else {
		    cerr << "Error traversing octree\n"; 
		}
	    }
	}
    }
    return 0;
}

int Octree::set_scalar(const Point &, double) {
    NOT_FINISHED("Octree:set_scalar");
    return 0;
}

int Octree::set_vector(const Point &, const Vector &) {
    NOT_FINISHED("Octree:set_scalar");
    return 0;
}

void Octree::erase_last_level_scalars() {
}

void Octree::erase_last_level_vectors() {
}

#define Octree_VERSION 1

void Pio(Piostream& stream, Octree*& o)
{
    if (stream.reading()) o=new Octree;
    stream.begin_cheap_delim();
    Pio(stream, o->nx);
    Pio(stream, o->ny);
    Pio(stream, o->nz);
    Pio(stream, o->mid);
    Pio(stream, o->bits);
    Pio(stream, o->leaf);
    Pio(stream, o->last_leaf);
    Pio(stream, o->trunk);
    Pio(stream, o->avg_vec);
    Pio(stream, o->min_sc);
    Pio(stream, o->max_sc);
    Pio(stream, o->avg_sc);

    if (o->bits) {
	Pio(stream, o->child[0][0][0]);
	if (o->bits & 1)
	    Pio(stream, o->child[0][0][1]);
	if (o->bits & 2) {
	    Pio(stream, o->child[0][1][0]);
	    if (o->bits & 1) {
		Pio(stream, o->child[0][1][1]);
	    }
	}
	if (o->bits & 4) {
	    Pio(stream, o->child[1][0][0]);
	    if (o->bits & 1)
		Pio(stream, o->child[1][0][1]);
	    if (o->bits & 2) {
	        Pio(stream, o->child[1][1][0]);
		if (o->bits & 1)
		    Pio(stream, o->child[1][1][1]);
	    }
	}
    }
    stream.end_cheap_delim();
}

#ifdef __GNUG__
#include <Classlib/LockingHandle.cc>
template class LockingHandle<OctreeTop>;

#endif
