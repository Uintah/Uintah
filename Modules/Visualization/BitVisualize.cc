
/*
 *  BitVisualize.cc:  IsoSurfaces a SFRG bitwise
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <config.h>
#undef SCI_ASSERTION_LEVEL_3
#define SCI_ASSERTION_LEVEL_2
#include <Classlib/BitArray1.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <Geometry/Point.h>
#include <Geom/Geom.h>
#include <Geom/Pt.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geom/Tri.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>
#include <iostream.h>
#include <strstream.h>

#define NUM_MATERIALS 5

inline Point mp(const Point &a, const Point &b) { 
    return Point((a.x()+b.x())/2, (a.y()+b.y())/2, (a.z()+b.z())/2);
}

class BitVisualize : public Module {
    ScalarFieldIPort* infield;

    GeometryOPort* ogeom;
    Array1<SurfaceOPort* > osurfs;
    Array1<SurfaceHandle> surf_hands;
    Array1<TriSurface*> surfs;
    TCLint emit_surface;
    Array1<TCLint*> isovals;
    Array1<GeomPts*> geomPts;
    Array1<int> BitVisualize_id;
    Array1<int> calc_mat;
    int isoChanged;
    int last_emit_surf;
    int sp;
    TCLint show_progress;

    Array1<MaterialHandle> matls;
    Array1<GeomGroup*> groups;
    Array1<int> geom_allocated;
    void vol_render_grid(ScalarFieldRG*);
    void iso_reg_grid(ScalarFieldRG*);

    Point ov[9];
    Point v[9];
public:
    BitVisualize(const clString& id);
    BitVisualize(const BitVisualize&, int deep);
    virtual ~BitVisualize();
    virtual Module* clone(int deep);
    virtual void execute();
};

#define FACE1 8
#define FACE2 4
#define FACE3 2
#define FACE4 1
#define ALLFACES (FACE1|FACE2|FACE3|FACE4)

struct MCubeTable {
    int which_case;
    int permute[8];
    int nbrs;
};

#include "mcube.h"

extern "C" {
Module* make_BitVisualize(const clString& id)
{
    return scinew BitVisualize(id);
}
};

BitVisualize::BitVisualize(const clString& id)
: Module("BitVisualize", id, Filter), emit_surface("emit_surface", id, this),
  show_progress("show_progress", id, this)
{
    // Create the input ports
    infield=scinew ScalarFieldIPort(this, "Field", ScalarFieldIPort::Atomic);
    add_iport(infield);
    // Create the output port
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    last_emit_surf=0;
    groups.resize(NUM_MATERIALS);
    geomPts.resize(NUM_MATERIALS);
    surfs.resize(NUM_MATERIALS);
    surf_hands.resize(NUM_MATERIALS);
    for (int i=0; i<NUM_MATERIALS; i++) {
	osurfs.add(scinew SurfaceOPort(this, "Surface", SurfaceIPort::Atomic));
	surf_hands[i]=surfs[i]=0;
	add_oport(osurfs[i]);
	geom_allocated.add(0);
	calc_mat.add(0);
	BitVisualize_id.add(0);
	surfs.add(scinew TriSurface);
	clString str;
	str = "iso" + to_string(i+1);
	isovals.add(scinew TCLint(str, id, this));
	isovals[i]->set(0);
	int r=(i+1)%2;		// 1 0 1 0 1 0
	int g=(((i+2)/3)%2);	// 0 1 1 1 0 0
	int b=((i/3)%2);	// 0 0 0 1 1 1
	matls.add(scinew Material(Color(0,0,0), Color(r*.6, g*.6, b*.6), 
			       Color(r*.5, g*.5, b*.5), 20));
    }	
}

BitVisualize::BitVisualize(const BitVisualize& copy, int deep)
: Module(copy, deep), emit_surface("emit_surface", id, this),
  show_progress("show_progress", id, this)
{
    NOT_FINISHED("BitVisualize::BitVisualize");
}

BitVisualize::~BitVisualize()
{
}

Module* BitVisualize::clone(int deep)
{
    return scinew BitVisualize(*this, deep);
}

void BitVisualize::execute()
{
    ScalarFieldHandle field;
    if(!infield->get(field))
	return;
    sp=show_progress.get();
    ScalarFieldRG* regular_grid=field->getRG();
    if(regular_grid){
	isoChanged=(int)regular_grid->grid(0,0,0);
	regular_grid->grid(0,0,0)=0;
	int i;
	for (i=0; i<NUM_MATERIALS; i++) {
	    int bit=1<<i;
	    if (isoChanged & bit) {		    // fld changed...
		if (isovals[i]->get()) {	    //   want this material...
		    if (geom_allocated[i]) {	    //     it's allocated...
			ogeom->delObj(BitVisualize_id[i]);//	      DELETE
		    } else {			    //     not allocated
			geom_allocated[i]=1; 	    // 	      MARK as ALLOCED
		    }
		    groups[i]=scinew GeomGroup;	    //	   ALLOCATE THEM
		    geomPts[i]=scinew GeomPts(10000);
		    calc_mat[i]=1;		    //	   *Calculate mat.*
		} else {			    //   don't want material...
		    if (geom_allocated[i]) {	    //     it's allocated...
			ogeom->delObj(BitVisualize_id[i]);//	      DELETE
			geom_allocated[i]=0;	    //	      MARK as UNALLOCED
		    }
		    calc_mat[i]=0;		    //	   *Don't calc. mat.*
		}
	    } else {				    // fld didn't change...
		if (isovals[i]->get()) {	    //	 want this material...
		    if (geom_allocated[i]) {	    //	   it's allocated...
			calc_mat[i]=0;		    //	      *Don't calc. mat*
		    } else {			    //	   not allocated
			geom_allocated[i]=1;	    //	      MARK
			groups[i]=scinew GeomGroup;    //	      ALLOCATE THEM
			geomPts[i]=scinew GeomPts(10000);
			calc_mat[i]=1;		    //	      *Calcluate mat*
		    }		
		} else {			    //   don't want material
		    calc_mat[i]=0;		    //     *Don't calc mat*
		    if (geom_allocated[i]) {	    //	   it's allocated...
			ogeom->delObj(BitVisualize_id[i]);//	      DELETE
			geom_allocated[i]=0;	    //	      MARK as UNALLOCED
		    }				    //	   not allocated
		}				    //        NOTHING TO DO!
	    }
	}
	vol_render_grid(regular_grid);
	for (i=0; i<NUM_MATERIALS; i++) {
	    if (calc_mat[i]) {
		groups[i]->add(geomPts[i]);
		GeomObj* topobj=scinew GeomMaterial(groups[i], matls[i]);
		clString nm = "Material " + to_string(i+1);
		BitVisualize_id[i] = ogeom->addObj(topobj, nm);
	    }
	}
	ogeom->flushViews();
	if (emit_surface.get()) {
	    int build_surfs=0;
	    for (i=0; i<NUM_MATERIALS; i++) {
		calc_mat[i]=0;
		int bit=1<<i;
		if ((isoChanged & bit) || !last_emit_surf) {
		    if (isovals[i]->get()) {
			build_surfs=1;
			calc_mat[i]=1;
		    }
		    surf_hands[i]=surfs[i]=0;
		}
	    }
	    if (build_surfs)
		iso_reg_grid(regular_grid);

	    for (i=0; i<NUM_MATERIALS; i++) {
		osurfs[i]->send(surf_hands[i]);
	    }
	} else {
	    for (i=0; i<NUM_MATERIALS; i++) {
		surf_hands[i]=surfs[i]=0;	    
	    }
	}
	last_emit_surf = emit_surface.get();
    } else {
	error("I can't BitVisualize this type of field...");
    }
}

void BitVisualize::vol_render_grid(ScalarFieldRG* field) {
    int nx=field->nx;
    int ny=field->ny;
    int nz=field->nz;
    Point minP, maxP;
    field->get_bounds(minP, maxP);
    double tweak[NUM_MATERIALS];
    Point pmid;
    for (int m=0; m<NUM_MATERIALS; m++) {
	int bit=1<<m;
	if (calc_mat[m]) {
	    tweak[m]=m/20.;
	    for (int i=0, ii=(int)minP.x(); i<nx; i++, ii++) {
		for (int j=0, jj=(int)minP.y(); j<ny; j++, jj++) {
		    for (int k=0, kk=(int)minP.z(); k<nz; k++, kk++) {
			pmid.x(ii+tweak[m]);
			pmid.y(jj+tweak[m]);
			pmid.z(kk+tweak[m]);
			int val=(int)(field->grid(i,j,k));
			if ((val & bit) != 0)
			    geomPts[m]->pts.add(pmid);
		    }
		}	
	    }
	}
    }
}

void BitVisualize::iso_reg_grid(ScalarFieldRG* field) {
    NOT_FINISHED("BitVisualize::iso_reg_grid is broken for now - it takes too long to compile.");
#if 0
    int nx=field->nx;
    int ny=field->ny;
    int nz=field->nz;
    int o[9];
    int oval[9];
    int val[9];
    Point pmin, pmax;
    field->get_bounds(pmin, pmax);
    int pminx=(int)pmin.x();
    int pminy=(int)pmin.y();
    int pminz=(int)pmin.z();
    int i;
    for (i=0; i<NUM_MATERIALS; i++) {
	if (calc_mat[i]) {
	    surf_hands[i]=surfs[i]=0;
	    surf_hands[i]=surfs[i]=scinew TriSurface();
	    surfs[i]->points.grow(1000);
	    surfs[i]->points.grow(-1000);
	    surfs[i]->elements.grow(1000);
	    surfs[i]->elements.grow(-1000);
	    surfs[i]->construct_hash(nx, ny, pmin, .5);
	}
    }
    for(i=0;i<nx-1;i++){
	//update_progress(i, nx);
	for(int j=0;j<ny-1;j++){
	    for(int k=0;k<nz-1;k++){
		int o1=o[1]=(int)field->grid(i, j, k);
		int o2=o[2]=(int)field->grid(i+1, j, k);
		int o3=o[3]=(int)field->grid(i+1, j+1, k);
		int o4=o[4]=(int)field->grid(i, j+1, k);
		int o5=o[5]=(int)field->grid(i, j, k+1);
		int o6=o[6]=(int)field->grid(i+1, j, k+1);
		int o7=o[7]=(int)field->grid(i+1, j+1, k+1);
		int o8=o[8]=(int)field->grid(i, j+1, k+1);
		int andall=o1 & o2 & o3 & o4 & o5 & o6 & o7 & o8;
		int orall=o1 | o2 | o3 | o4 | o5 | o6 | o7 | o8;
		int xall=andall ^ orall;
		int do_mat[NUM_MATERIALS];
		int check_box=0;
		for (int ii=0; ii<NUM_MATERIALS; ii++) {
		    if ((do_mat[ii]=(calc_mat[ii] && (xall & (1<<ii)))))
			check_box=1;
		}
		if (check_box) {
		    ov[1]=Point(pminx+i,pminy+j,pminz+k);
		    ov[2]=Point(pminx+i+1,pminy+j,pminz+k);
		    ov[3]=Point(pminx+i+1,pminy+j+1,pminz+k);
		    ov[4]=Point(pminx+i,pminy+j+1,pminz+k);
		    ov[5]=Point(pminx+i,pminy+j,pminz+k+1);
		    ov[6]=Point(pminx+i+1,pminy+j,pminz+k+1);
		    ov[7]=Point(pminx+i+1,pminy+j+1,pminz+k+1);
		    ov[8]=Point(pminx+i,pminy+j+1,pminz+k+1);
		    for (int m=0; m<NUM_MATERIALS; m++) {
			if (do_mat[m]) {
			    int a = (1 << m);
			    for (int jj=1; jj<9; jj++)
				if ((o[jj] & a)!=0) oval[jj]=1;else oval[jj]=0;
			    int mask=0;
			    int idx;
			    for(idx=1;idx<=8;idx++){
				if(oval[idx])
				    mask|=1<<(idx-1);
			    }
			    MCubeTable* tab=&mcube_table[mask];
			    for(idx=1;idx<=8;idx++){
				val[idx]=oval[tab->permute[idx-1]];
				v[idx]=ov[tab->permute[idx-1]];
			    }
			    int wcase=tab->which_case;
			    switch(wcase) {
			    case 0:
				cerr << "Shouldn't be here! \n";
				break;
			    case 1:
				{
                                    Point p1(mp(v[1], v[2]));
                                    Point p2(mp(v[1], v[5]));
                                    Point p3(mp(v[1], v[4]));
				    surfs[m]->cautious_add_triangle(p1,p2,p3);
				}
				break;
			    case 2:
				{
                                    Point p1(mp(v[1], v[5]));
                                    Point p2(mp(v[2], v[6]));
                                    Point p3(mp(v[2], v[3]));
				    surfs[m]->cautious_add_triangle(p1,p2,p3);
                                    Point p4(mp(v[1], v[4]));
				    surfs[m]->cautious_add_triangle(p3,p4,p1);
				}
				break;
			    case 3:
				{
                                    Point p1(mp(v[1], v[2]));
                                    Point p2(mp(v[1], v[5]));
                                    Point p3(mp(v[1], v[4]));
				    surfs[m]->cautious_add_triangle(p1,p2,p3);
                                    Point p4(mp(v[3], v[2]));
                                    Point p5(mp(v[3], v[7]));
                                    Point p6(mp(v[3], v[4]));
				    surfs[m]->cautious_add_triangle(p4,p5,p6);
				}
				break;
			    case 4:
				{
                                    Point p1(mp(v[1], v[2]));
                                    Point p2(mp(v[1], v[5]));
                                    Point p3(mp(v[1], v[4]));
				    surfs[m]->cautious_add_triangle(p1,p2,p3);
                                    Point p4(mp(v[7], v[3]));
                                    Point p5(mp(v[7], v[8]));
                                    Point p6(mp(v[7], v[6]));
				    surfs[m]->cautious_add_triangle(p4,p5,p6);
				}
				break;
			    case 5:
				{
                                    Point p1(mp(v[2], v[1]));
                                    Point p2(mp(v[2], v[3]));
                                    Point p3(mp(v[5], v[1]));
				    surfs[m]->cautious_add_triangle(p1,p2,p3);
                                    Point p4(mp(v[5], v[8]));
				    surfs[m]->cautious_add_triangle(p4,p3,p2);
                                    Point p5(mp(v[6], v[7]));
				    surfs[m]->cautious_add_triangle(p5,p4,p2);
				}
				break;
			    case 6:
				{
                                    Point p1(mp(v[1], v[5]));
                                    Point p2(mp(v[2], v[6]));
                                    Point p3(mp(v[2], v[3]));
				    surfs[m]->cautious_add_triangle(p1,p2,p3);
                                    Point p4(mp(v[1], v[4]));
				    surfs[m]->cautious_add_triangle(p3,p4,p1);
                                    Point p5(mp(v[7], v[3]));
                                    Point p6(mp(v[7], v[8]));
                                    Point p7(mp(v[7], v[6]));
				    surfs[m]->cautious_add_triangle(p5,p6,p7);
				}
				break;
			    case 7:
				{
                                    Point p1(mp(v[2], v[1]));
                                    Point p2(mp(v[2], v[3]));
                                    Point p3(mp(v[2], v[6]));
				    surfs[m]->cautious_add_triangle(p1,p2,p3);
                                    Point p4(mp(v[4], v[1]));
                                    Point p5(mp(v[4], v[3]));
                                    Point p6(mp(v[4], v[8]));
				    surfs[m]->cautious_add_triangle(p4,p5,p6);
                                    Point p7(mp(v[7], v[8]));
                                    Point p8(mp(v[7], v[6]));
                                    Point p9(mp(v[7], v[3]));
				    surfs[m]->cautious_add_triangle(p7,p8,p9);
				}
				break;
			    case 8:
				{
                                    Point p1(mp(v[1], v[4]));
                                    Point p2(mp(v[2], v[3]));
                                    Point p3(mp(v[6], v[7]));
				    surfs[m]->cautious_add_triangle(p1,p2,p3);
                                    Point p4(mp(v[5], v[8]));
				    surfs[m]->cautious_add_triangle(p4,p1,p3);
				}
				break;
			    case 9:
				{
                                    Point p1(mp(v[1], v[2]));
                                    Point p2(mp(v[6], v[2]));
                                    Point p3(mp(v[6], v[7]));
				    surfs[m]->cautious_add_triangle(p1,p2,p3);
                                    Point p4(mp(v[8], v[7]));
				    surfs[m]->cautious_add_triangle(p1,p3,p4);
                                    Point p5(mp(v[1], v[4]));
				    surfs[m]->cautious_add_triangle(p1,p4,p5);
                                    Point p6(mp(v[8], v[4]));
				    surfs[m]->cautious_add_triangle(p5,p4,p6);
				}
				break;
			    case 10:
				{
                                    Point p1(mp(v[1], v[2]));
                                    Point p2(mp(v[4], v[3]));
                                    Point p3(mp(v[1], v[5]));
				    surfs[m]->cautious_add_triangle(p1,p2,p3);
                                    Point p4(mp(v[4], v[8]));
				    surfs[m]->cautious_add_triangle(p2,p4,p3);
                                    Point p5(mp(v[6], v[2]));
                                    Point p6(mp(v[6], v[5]));
                                    Point p7(mp(v[7], v[3]));
				    surfs[m]->cautious_add_triangle(p5,p6,p7);
                                    Point p8(mp(v[7], v[8]));
				    surfs[m]->cautious_add_triangle(p2,p8,p3);
				}
				break;
			    case 11:
				{
                                    Point p1(mp(v[1], v[2]));
                                    Point p2(mp(v[6], v[2]));
                                    Point p3(mp(v[7], v[3]));
				    surfs[m]->cautious_add_triangle(p1,p2,p3);
                                    Point p4(mp(v[5], v[8]));
				    surfs[m]->cautious_add_triangle(p1,p3,p4);
                                    Point p5(mp(v[1], v[4]));
				    surfs[m]->cautious_add_triangle(p1,p4,p5);
                                    Point p6(mp(v[7], v[8]));
				    surfs[m]->cautious_add_triangle(p4,p3,p6);
				}
				break;
			    case 12:
				{
                                    Point p1(mp(v[2], v[1]));
                                    Point p2(mp(v[2], v[3]));
                                    Point p3(mp(v[5], v[1]));
				    surfs[m]->cautious_add_triangle(p1,p2,p3);
				    Point p4(mp(v[5], v[8]));
				    surfs[m]->cautious_add_triangle(p3,p2,p4);
                                    Point p5(mp(v[6], v[7]));
				    surfs[m]->cautious_add_triangle(p4,p2,p5);
                                    Point p6(mp(v[4], v[1]));
                                    Point p7(mp(v[4], v[3]));
                                    Point p8(mp(v[4], v[8]));
				    surfs[m]->cautious_add_triangle(p6,p7,p8);
				}
				break;
			    case 13:
				{
                                    Point p1(mp(v[1], v[2]));
                                    Point p2(mp(v[1], v[5]));
                                    Point p3(mp(v[1], v[4]));
				    surfs[m]->cautious_add_triangle(p1,p2,p3);
                                    Point p4(mp(v[3], v[2]));
                                    Point p5(mp(v[3], v[7]));
                                    Point p6(mp(v[3], v[4]));
				    surfs[m]->cautious_add_triangle(p4,p5,p6);
                                    Point p7(mp(v[6], v[2]));
                                    Point p8(mp(v[6], v[7]));
                                    Point p9(mp(v[6], v[5]));
				    surfs[m]->cautious_add_triangle(p7,p8,p9);
                                    Point p10(mp(v[8], v[5]));
                                    Point p11(mp(v[8], v[7]));
                                    Point p12(mp(v[8], v[4]));
				    surfs[m]->cautious_add_triangle(p10,p11,p12);
				}
				break;
			    case 14:
				{
                                    Point p1(mp(v[2], v[1]));
                                    Point p2(mp(v[2], v[3]));
                                    Point p3(mp(v[6], v[7]));
				    surfs[m]->cautious_add_triangle(p1,p2,p3);
                                    Point p4(mp(v[8], v[4]));
				    surfs[m]->cautious_add_triangle(p1,p3,p4);
                                    Point p5(mp(v[5], v[1]));
				    surfs[m]->cautious_add_triangle(p1,p4,p5);
				    Point p6(mp(v[8], v[7]));
				    surfs[m]->cautious_add_triangle(p3,p6,p4);
				}
				break;
			    default:
				break;
			    }
			}
		    }
		}
	    }
	    if(sp && abort_flag)
		return;
	}
    }
#endif
}

#ifdef __GNUG__

#include <Classlib/Array1.cc>
template class Array1<SurfaceOPort*>;
template class Array1<SurfaceHandle>;
template class Array1<TriSurface*>;
template class Array1<TCLint*>;
template class Array1<GeomPts*>;
template class Array1<int>;
template class Array1<MaterialHandle>;
template class Array1<GeomGroup*>;

#endif

