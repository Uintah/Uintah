
/*
 *  ExtractSurfs.cc:  Extract surfaces from a segmented volume and mask
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 30, 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <Classlib/BitArray1.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Classlib/Queue.h>
#include <Classlib/Stack.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColorMapPort.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Datatypes/ScalarFieldRGchar.h>
#include <Datatypes/ScalarFieldRGint.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/SurfOctree.h>
#include <Datatypes/TriSurface.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geom/Pt.h>
#include <Geom/Triangles.h>
#include <Geom/TriStrip.h>
#include <Geometry/Point.h>
#include <Geometry/Plane.h>
#include <Malloc/Allocator.h>
#include <Math/Expon.h>
#include <Math/MiscMath.h>
#include <TCL/TCLvar.h>
#include <Widgets/ArrowWidget.h>
#include <iostream.h>
#include <strstream.h>

// just so I can see the proccess id...

#include <sys/types.h>
#include <unistd.h>

class ExtractSurfs : public Module {
    ScalarFieldIPort* infield;
    ScalarFieldIPort* inmask;

    GeometryOPort* ogeom;

    Array1<int> ExtractSurfs_id;
    int have_mask;
    Array1<char> matl_idx;
    Array1<GeomPts *>gp;
public:
    ExtractSurfs(const clString& id);
    ExtractSurfs(const ExtractSurfs&, int deep);
    virtual ~ExtractSurfs();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_ExtractSurfs(const clString& id)
{
    return scinew ExtractSurfs(id);
}
}

//static clString module_name("ExtractSurfs");

ExtractSurfs::ExtractSurfs(const clString& id)
: Module("ExtractSurfs", id, Filter)
{
    // Create the input ports
    infield=scinew ScalarFieldIPort(this, "Field", ScalarFieldIPort::Atomic);
    add_iport(infield);
    inmask=scinew ScalarFieldIPort(this, "Mask", ScalarFieldIPort::Atomic);
    add_iport(inmask);

    // Create the output port
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

    ExtractSurfs_id=0;
}

ExtractSurfs::ExtractSurfs(const ExtractSurfs& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("ExtractSurfs::ExtractSurfs");
}

ExtractSurfs::~ExtractSurfs()
{
}

Module* ExtractSurfs::clone(int deep)
{
    return scinew ExtractSurfs(*this, deep);
}

void ExtractSurfs::execute()
{
    // throw away downstream geomoetry
    if(ExtractSurfs_id.size())
	for (int ext=0; ext<ExtractSurfs_id.size(); ext++)
	    if (ExtractSurfs_id[ext])
		ogeom->delObj(ExtractSurfs_id[ext]);

    // get and check validity of input fields
    ScalarFieldHandle fldHand;
    if(!infield->get(fldHand))
	return;
    ScalarFieldRGBase* fldBase = fldHand->getRGBase();
    if (!fldBase)
	return;
    ScalarFieldRGchar* field = fldBase->getRGChar();
    if (!field)
	return;
    ScalarFieldHandle maskHand;    
    ScalarFieldRGchar* mask;
    if(!inmask->get(maskHand)) {
	have_mask=0;
    } else {
	have_mask=1;
	ScalarFieldRGBase* maskBase = maskHand->getRGBase();
	if(!maskBase)
	    return;
	mask = maskBase->getRGChar();
	if(!mask)
	    return;
    }

    // now we have our segmented filed and our mask field, let's make sure
    //   they're the same size and in the same place...
    Point fldMinP, fldMaxP, maskMinP, maskMaxP;
    field->get_bounds(fldMinP, fldMaxP);
    if (have_mask) {
	mask->get_bounds(maskMinP, maskMaxP);
	if (fldMinP != maskMinP || fldMaxP != maskMaxP)
	    return;
	if (field->nx != mask->nx || 
	    field->ny != mask->ny || 
	    field->nz != mask->nz)
	    return;
    }
    
    // ok, looks like all systems are go!  let's do some work...
    int nx=field->nx;
    int ny=field->ny;
    int nz=field->nz;

    int count=nx*ny*nz;
    int visited=0;
    ScalarFieldRGint* SF=scinew ScalarFieldRGint();
    SF->resize(nx,ny,nz);
    SF->grid.initialize(-1);
    Queue<int> q;
    int currMatl=-1;
    matl_idx.resize(0);
    Array1<int>old_matl_idx;
    Array3<char>edges(nx,ny,nz);
    edges.initialize(0);
    for (int i=0; i<nx && visited<count; i++) 
	for (int j=0; j<ny && visited<count; j++) {
	    for (int k=0; k<nz; k++) {
		if ((SF->grid(i,j,k) == -1) && (!have_mask || 
						(mask->grid(i,j,k) != '0'))) {
		    q.append(i);
		    q.append(j);
		    q.append(k);
		    currMatl++;
		    SF->grid(i,j,k)=currMatl;
		    char m=field->grid(i,j,k);
		    matl_idx.add(m);

		    // keep track of how many materials we have
		    int newM=1;
		    for (int o=0; o<old_matl_idx.size(); o++)
			if (m==old_matl_idx[o]) newM=0;
		    if (newM) old_matl_idx.add(m);

		    visited++;
		    while (!q.is_empty()) {
			int ci=q.pop();
			int cj=q.pop();
			int ck=q.pop();
			int aa[6]={ci-1, ci+1, ci, ci, ci, ci};
			int bb[6]={cj, cj, cj-1, cj+1, cj, cj};
			int cc[6]={ck, ck, ck, ck, ck-1, ck+1};
			edges(ci,cj,ck)=0;
			for (int nbr=0; nbr<6; nbr++) {
			    int ii=aa[nbr];
			    int jj=bb[nbr];
			    int kk=cc[nbr];
			    if(ii>=0 && jj>=0 && kk>=0 &&
			       ii<nx && jj<ny && kk<nz) {
				if ((field->grid(ii,jj,kk) == m) &&
				    (!have_mask||(mask->grid(ii,jj,kk)!='0'))){
				    if (SF->grid(ii,jj,kk) == -1) {
					q.append(ii);
					q.append(jj);
					q.append(kk);
					SF->grid(ii,jj,kk)=currMatl;
					visited++;
				    }
				} else {
				    edges(ci,cj,ck) |= 1<<nbr;
				}
			    }
			}
		    }
		}
	    }
	}
    cerr << "ExtractSurfs visited "<<visited<<" of "<<count<< " nodes.\n";
    cerr << "ExtractSurfs found "<<matl_idx.size()<<" regions from ";
    cerr << old_matl_idx.size()<<" materials.\n";

    // allocate the geometry groups
    int maxMat=0;
    for (i=0; i<matl_idx.size(); i++)
	if (matl_idx[i]>maxMat)
	    maxMat=matl_idx[i];
    maxMat-='0';
    Array1<Array1<Point> > pts(maxMat+1);
    for (int xx=0; xx<nx; xx++)
	for (int yy=0; yy<ny; yy++)
	    for (int zz=0; zz<nz; zz++) {
		int mm=SF->grid(xx,yy,zz);
		if ((mm != -1) && (edges(xx,yy,zz))) {
		    Point p = field->get_point(xx,yy,zz);
		    pts[(matl_idx[mm]-'0')].add(p);
		}
	    }
    ExtractSurfs_id.resize(maxMat+1);
    gp.resize(maxMat+1);
    for (i=0; i<maxMat+1; i++) {
	if(pts[i].size()) {
	    gp[i]=scinew GeomPts(pts[i].size());
	    for (xx=0; xx<pts[i].size(); xx++)
		gp[i]->add(pts[i][xx]);
	    ExtractSurfs_id[i]=ogeom->addObj(gp[i], "material"+to_string(i));
	} else {
	    ExtractSurfs_id[i]=0;
	}
    }

    // build the segmented octree for hierarchically storing the field
cerr << "Starting to make the tree!\n";
    SurfOctreeTop top(SF);
    SurfOctree* tree=top.tree;
cerr << "Done making the tree.\n";
    tree->propagate_up_materials();
cerr << "Done propagating materials.\n";
}


#ifdef __GNUG__

#include <Classlib/Queue.cc>

template class Queue<int>;

#endif
