/*
 *  FieldFilter.cc:  Unfinished modules
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/Array2.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRGBase.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldRGdouble.h>
#include <Datatypes/ScalarFieldRGfloat.h>
#include <Datatypes/ScalarFieldRGint.h>
#include <Datatypes/ScalarFieldRGshort.h>
#include <Datatypes/ScalarFieldRGchar.h>
#include <Datatypes/ScalarFieldRGuchar.h>
#include <Geometry/Point.h>
#include <TCL/TCLvar.h>
#include <Widgets/ScaledBoxWidget.h>
#include <stdio.h>
#include <math.h>
#include <iostream.h>

#define RESIZE_MACRO   fX->resize(lastZ, isf->nx, isf->ny);fX->grid.initialize(0);fY->resize(lastY, lastZ, isf->nx);fY->grid.initialize(0);of->resize(lastX, lastY, lastZ);of->grid.initialize(0);fldX=fX;fldY=fY;osf=of;


class FieldFilter : public Module {
    void boxFilter(ScalarFieldRGBase *in, ScalarFieldRGBase *out, int, int);
    void triangleFilter(ScalarFieldRGBase *, ScalarFieldRGBase *, int, int);
    void mitchellFilter(ScalarFieldRGBase *, ScalarFieldRGBase *, int, int);
    void filter(ScalarFieldRGBase *, ScalarFieldRGBase *, int minz, int maxz);
    void buildUndersampleTriangleTable(Array2<double>*, int, int, double);
    void buildOversampleTriangleTable(Array2<double>*, int, double);
    void buildUndersampleMitchellTable(Array2<double>*, int, int, double);
    void buildOversampleMitchellTable(Array2<double>*, int, double);
    virtual void widget_moved(int last);    
    ScalarFieldIPort* iField;
    ScalarFieldOPort* oField;
    ScalarFieldHandle oFldHandle;
    ScalarFieldRGBase* osf;
    ScalarFieldRGBase* isf;
    int lastX, lastY, lastZ;
    int check_widget, init;
    TCLint ox;
    TCLint oy;
    TCLint oz;
    TCLint nx;
    TCLint ny;
    TCLint nz;
    TCLint nMaxX;
    TCLint nMaxY;
    TCLint nMaxZ;
    TCLint range_min_x;
    TCLint range_min_y;
    TCLint range_min_z;
    TCLint range_max_x;
    TCLint range_max_y;
    TCLint range_max_z;
    TCLint sameInput;
    TCLstring filterType;
    clString lastFT;
    GeometryOPort *ogeom;
    CrowdMonitor widget_lock;
    int widget_id;
    ScaledBoxWidget *widget;

public:
    FieldFilter(const clString& id);
    FieldFilter(const FieldFilter&, int deep);
    virtual ~FieldFilter();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_FieldFilter(const clString& id)
{
    return new FieldFilter(id);
}
}

static clString module_name("FieldFilter");
FieldFilter::FieldFilter(const clString& id)
: Module("FieldFilter", id, Filter), ox("ox", id, this), oy("oy", id, this),
  oz("oz",id,this), nx("nx", id, this), ny("ny", id, this), nz("nz", id, this),
  nMaxX("nMaxX", id, this), nMaxY("nMaxY", id, this), nMaxZ("nMaxZ", id, this),
  sameInput("sameInput", id, this), filterType("filterType", id, this),
  range_min_x("range_min_x", id, this), range_min_y("range_min_y", id, this),
  range_min_z("range_min_z", id, this), range_max_x("range_max_x", id, this),
  range_max_y("range_max_y", id, this), range_max_z("range_max_z", id, this)
{
    check_widget=lastX=lastY=lastZ=0;
    iField=new ScalarFieldIPort(this, "Geometry", ScalarFieldIPort::Atomic);
    oFldHandle=osf=0;
    add_iport(iField);
    // Create the output ports
    oField=new ScalarFieldOPort(this, "Geometry", ScalarFieldIPort::Atomic);
    add_oport(oField);
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    init=0;
}

void FieldFilter::widget_moved(int last) {
    if (last && !abort_flag) {
	abort_flag=1;
	check_widget=1;
	want_to_execute();
    }
}

FieldFilter::FieldFilter(const FieldFilter& copy, int deep)
: Module(copy, deep), ox("ox", id, this), oy("oy", id, this),
  oz("oz",id,this), nx("nx", id, this), ny("ny", id, this), nz("nz", id, this),
  nMaxX("nMaxX", id, this), nMaxY("nMaxY", id, this), nMaxZ("nMaxZ", id, this),
  sameInput("sameInput", id, this), filterType("filterType", id, this),
  range_min_x("range_min_x", id, this), range_min_y("range_min_y", id, this),
  range_min_z("range_min_z", id, this), range_max_x("range_max_x", id, this),
  range_max_y("range_max_y", id, this), range_max_z("range_max_z", id, this)
{
}

FieldFilter::~FieldFilter()
{
}

Module* FieldFilter::clone(int deep)
{
    return new FieldFilter(*this, deep);
}

void FieldFilter::filter(ScalarFieldRGBase* in, ScalarFieldRGBase* out,
			 int min_z, int max_x) {
    if (lastFT == "Box") {
	boxFilter(in, out, min_z, max_x);
    } else if (lastFT == "Triangle") {
	triangleFilter(in, out, min_z, max_x);
    } else {
	triangleFilter(in, out, min_z, max_x);
//	mitchellFilter(in, out, min_z, max_x);
    }
}

void FieldFilter::execute() {
    ScalarFieldHandle ifh;
    if(!iField->get(ifh))
	return;
    ScalarFieldRGBase *rgb=isf;
    isf=ifh->getRGBase();
    if(!isf){
	error("FieldFilter can't deal with unstructured grids!");
	return;
    }
    if (!init) {
	init=1;
	widget=new ScaledBoxWidget(this, &widget_lock, 0.1);
	GeomObj *w=widget->GetWidget();
	widget_id = ogeom->addObj(w, module_name, &widget_lock);
	widget->Connect(ogeom);
	//widget->AxisAligned(1);
	widget->SetPosition(Point(.5,.5,.5), Point(1,0,0), Point(0,1,0), 
			    Point(0,0,1));
	widget->SetScale(1./20);
	widget->SetRatioR(1./16);
	widget->SetRatioD(1./16);
	widget->SetRatioI(1./16);
    }
    Point p1;
    Point p2;
    isf->get_bounds(p1, p2);
    int newFld=0;
    if (isf != rgb) {
	newFld=1;
    }
    if (check_widget) {
	// Set the sliders based on the widget parameters
	Point center, right, down, in;
	widget->GetPosition(center, right, down, in);
	int mini, minj, mink;
	int maxi, maxj, maxk;
	Point min(right.x(), down.y(), in.z());
	Point max(center+(center-min));
	isf->locate(min, mini, minj, mink);
	isf->locate(max, maxi, maxj, maxk);
	mini++; minj++; mink++; maxi++; maxj++; maxk++;
	if (mini<1) mini=1;
	if (minj<1) minj=1;
	if (mink<1) mink=1;
	if (maxi<2) maxi=2;
	if (maxj<2) maxj=2;
	if (maxk<2) maxk=2;
	if (maxi>isf->nx) maxi=isf->nx;
	if (maxj>isf->ny) maxj=isf->ny;
	if (maxk>isf->nz) maxk=isf->nz;
	if (mini>isf->nx) mini=isf->nx-1;
	if (minj>isf->ny) minj=isf->ny-1;
	if (mink>isf->nz) mink=isf->nz-1;
	if (mini==maxi) maxi++;
	if (minj==maxj) maxj++;
	if (mink==maxk) maxk++;
	range_min_x.set(mini);
	range_min_y.set(minj);
	range_min_z.set(mink);
	range_max_x.set(maxi);
	range_max_y.set(maxj);
	range_max_z.set(maxk);
    }
    if (newFld) {
	nx.set((int) (1./widget->GetRatioR()));
	ny.set((int) (1./widget->GetRatioD()));
	nz.set((int) (1./widget->GetRatioI()));
    }
    // Put the widget in the right place
    int min_x=range_min_x.get()-1;
    int min_y=range_min_y.get()-1;
    int min_z=range_min_z.get()-1;
    int max_x=range_max_x.get()-1;
    int max_y=range_max_y.get()-1;
    int max_z=range_max_z.get()-1;
    
    Point minPt(isf->get_point(min_x, min_y, min_z));
    Point maxPt(isf->get_point(max_x, max_y, max_z));
    Point ctrPt(Interpolate(minPt, maxPt, .5));
    widget->SetPosition(ctrPt, Point(minPt.x(), ctrPt.y(), ctrPt.z()),
			Point(ctrPt.x(), minPt.y(), ctrPt.z()),
			Point(ctrPt.x(), ctrPt.y(), minPt.z()));
    widget->SetRatioR(1./nx.get());
    widget->SetRatioD(1./ny.get());
    widget->SetRatioI(1./nz.get());

    // Check if anything has changed (i.e. do we need to reexecute)
    if (((max_x-min_x) == 0) || ((max_y-min_y) == 0) || ((max_z-min_z) == 0))
	return;
	
#if 0
    if (!check_widget && sameInput.get() &&
	(isf->nx == ox.get() && isf->ny == oy.get() && 
	 isf->nz == oz.get() && nx.get() == lastX && 
	 ny.get() == lastY && nz.get() == lastZ &&
	 lastFT == filterType.get())) {
	oField->send(oFldHandle);
	cerr << "Field Filter: SAME AS BEFORE!\n";
	return;
    }
#endif

    // reposition the widget

    ox.set(isf->nx);
    oy.set(isf->ny);
    oz.set(isf->nz);
    nMaxX.set(isf->nx*4);
    nMaxY.set(isf->ny*4);
    nMaxZ.set(isf->nz*4);

    reset_vars();
    lastX=nx.get();
    lastY=ny.get();
    lastZ=nz.get();
    lastFT=filterType.get();

    ScalarFieldRGdouble *ifd=isf->getRGDouble();
    ScalarFieldRGfloat *iff=isf->getRGFloat();
    ScalarFieldRGint *ifi=isf->getRGInt();
    ScalarFieldRGshort *ifs=isf->getRGShort();
    ScalarFieldRGchar *ifc=isf->getRGChar();
    ScalarFieldRGuchar *ifu=isf->getRGUchar();

    ScalarFieldRGBase *fldX, *fldY;
    if (ifd) {
	ScalarFieldRGdouble *fX, *fY, *of;
	fX=new ScalarFieldRGdouble();
	fY=new ScalarFieldRGdouble();
	oFldHandle=of=new ScalarFieldRGdouble();
	RESIZE_MACRO
    } else if (iff) {
	ScalarFieldRGfloat *fX, *fY, *of;
	fX=new ScalarFieldRGfloat();
	fY=new ScalarFieldRGfloat();
	oFldHandle=of=new ScalarFieldRGfloat();
	RESIZE_MACRO
    } else if (ifi) {
	ScalarFieldRGint *fX, *fY, *of;
	fX=new ScalarFieldRGint();
	fY=new ScalarFieldRGint();
	oFldHandle=of=new ScalarFieldRGint();
	RESIZE_MACRO
    } else if (ifs) {
	ScalarFieldRGshort *fX, *fY, *of;
	fX=new ScalarFieldRGshort();
	fY=new ScalarFieldRGshort();
	oFldHandle=of=new ScalarFieldRGshort();
	RESIZE_MACRO
    } else if (ifc) {
	ScalarFieldRGchar *fX, *fY, *of;
	fX=new ScalarFieldRGchar();
	fY=new ScalarFieldRGchar();
	oFldHandle=of=new ScalarFieldRGchar();
	RESIZE_MACRO
    } else if (ifu) {
	ScalarFieldRGuchar *fX, *fY, *of;
	fX=new ScalarFieldRGuchar();
	fY=new ScalarFieldRGuchar();
	oFldHandle=of=new ScalarFieldRGuchar();
	RESIZE_MACRO
    } else {
	cerr << "Unknown SFRG type in FieldFilter: "<<isf->getType()<<"\n";
	return;
    }
    fldX->set_bounds(minPt, maxPt);
    fldY->set_bounds(minPt, maxPt);
    osf->set_bounds(minPt, maxPt);
    
    filter(isf, fldX, min_z, max_z);
    filter(fldX, fldY, min_y, max_y);
    filter(fldY, osf, min_x, max_x);
    //    oFldHandle = osf;
    oField->send(oFldHandle);
}

void FieldFilter::boxFilter(ScalarFieldRGBase* in, ScalarFieldRGBase* out,
				  int fZ, int lZ) {
    ScalarFieldRGdouble *ifd, *ofd;
    ScalarFieldRGfloat *iff, *off;
    ScalarFieldRGint *ifi, *ofi;
    ScalarFieldRGshort *ifs, *ofs;
    ScalarFieldRGchar *ifc, *ofc;
    ScalarFieldRGuchar *ifu, *ofu;
    ifd=in->getRGDouble();
    ofd=out->getRGDouble();
    iff=in->getRGFloat();
    off=out->getRGFloat();
    ifi=in->getRGInt();
    ofi=out->getRGInt();
    ifs=in->getRGShort();
    ofs=out->getRGShort();
    ifc=in->getRGChar();
    ofc=out->getRGChar();
    ifu=in->getRGUchar();
    ofu=out->getRGUchar();

    double ratio=(lZ-fZ)/(out->nx-1.);
    for (int i=0; i<out->ny; i++) {
	for (int j=0; j<out->nz; j++) {
	    double curr=fZ;
	    if (ifd) for (int k=0; k<out->nx; k++, curr+=ratio)
		ofd->grid(k,i,j)=ifd->grid(i, j, (int)curr);
	    else if (iff) for (int k=0; k<out->nx; k++, curr+=ratio)
		off->grid(k,i,j)=iff->grid(i, j, (int)curr);
	    else if (ifi) for (int k=0; k<out->nx; k++, curr+=ratio)
		ofi->grid(k,i,j)=ifi->grid(i, j, (int)curr);
	    else if (ifs) for (int k=0; k<out->nx; k++, curr+=ratio)
		ofs->grid(k,i,j)=ifs->grid(i, j, (int)curr);
	    else if (ifc) for (int k=0; k<out->nx; k++, curr+=ratio)
		ofc->grid(k,i,j)=ifc->grid(i, j, (int)curr);
	    else if (ifu) for (int k=0; k<out->nx; k++, curr+=ratio)
		ofu->grid(k,i,j)=ifu->grid(i, j, (int)curr);
	    else {
		cerr << "Unknown SFRG type -- shouldn't ever get here!\n";
		return;
	    }
	}
    }
}

void FieldFilter::triangleFilter(ScalarFieldRGBase* in, ScalarFieldRGBase* out,
				 int fZ, int lZ) {
    ScalarFieldRGdouble *ifd, *ofd;
    ScalarFieldRGfloat *iff, *off;
    ScalarFieldRGint *ifi, *ofi;
    ScalarFieldRGshort *ifs, *ofs;
    ScalarFieldRGchar *ifc, *ofc;
    ScalarFieldRGuchar *ifu, *ofu;
    ifd=in->getRGDouble();
    ofd=out->getRGDouble();
    iff=in->getRGFloat();
    off=out->getRGFloat();
    ifi=in->getRGInt();
    ofi=out->getRGInt();
    ifs=in->getRGShort();
    ofs=out->getRGShort();
    ifc=in->getRGChar();
    ofc=out->getRGChar();
    ifu=in->getRGUchar();
    ofu=out->getRGUchar();
    
    double ratio=(lZ-fZ)/(out->nx-1.);
    if (ratio == 1) {		// trivial filter
	for (int i=0; i<out->ny; i++)
	    for (int j=0; j<out->nz; j++)
		if (ifd) for (int k=0, kk=fZ; k<out->nx; k++, kk++)
		    ofd->grid(k,i,j)=ifd->grid(i, j, kk);
		else if (iff) for (int k=0, kk=fZ; k<out->nx; k++, kk++)
		    off->grid(k,i,j)=iff->grid(i, j, kk);
		else if (ifi) for (int k=0, kk=fZ; k<out->nx; k++, kk++)
		    ofi->grid(k,i,j)=ifi->grid(i, j, kk);
		else if (ifs) for (int k=0, kk=fZ; k<out->nx; k++, kk++)
		    ofs->grid(k,i,j)=ifs->grid(i, j, kk);
		else if (ifc) for (int k=0, kk=fZ; k<out->nx; k++, kk++)
		    ofc->grid(k,i,j)=ifc->grid(i, j, kk);
		else if (ifu) for (int k=0, kk=fZ; k<out->nx; k++, kk++)
		    ofu->grid(k,i,j)=ifu->grid(i, j, kk);
		else {
		    cerr << "Unknown SFRG type -- shouldn't ever get here!\n";
		    return;
		}
    } else if (ratio>1) {		// undersampling     big->small
	int span=ceil(2.*ratio);
	Array2<double> table(out->nx, span);
	buildUndersampleTriangleTable(&table, out->nx, span, ratio);
	for (int i=0; i<out->ny; i++)
	    for (int j=0; j<out->nz; j++)
		for (int k=0, kk=0; kk<out->nx; k++, kk++) {
		    int inPixelIdxBase=(int)((k-1)*ratio+fZ);
		    for (int l=0; l<span; l++) {
			double tEntry=table(k,l);
			int inPixelIdx=inPixelIdxBase+l;
			if (inPixelIdx<fZ) inPixelIdx=fZ;
			else if (inPixelIdx>lZ) inPixelIdx=lZ;
			if (ifd) ofd->grid(k,i,j)+=
				     ifd->grid(i,j,inPixelIdx)*tEntry;
			else if (iff) off->grid(k,i,j)+=
					  iff->grid(i,j,inPixelIdx)*tEntry;
			else if (ifi) ofi->grid(k,i,j)+=
					  ifi->grid(i,j,inPixelIdx)*tEntry;
			else if (ifs) ofs->grid(k,i,j)+=
					  ifs->grid(i,j,inPixelIdx)*tEntry;
			else if (ifc) ofc->grid(k,i,j)+=
					  ifc->grid(i,j,inPixelIdx)*tEntry;
			else if (ifu) ofu->grid(k,i,j)+=
					  ifu->grid(i,j,inPixelIdx)*tEntry;
			else {
			    cerr << "Error - shouldn't ever get here!\n";
			}
		    }
		}
    } else {			// oversampling      small->big
	Array2<double> table(out->nx, 2);
	buildOversampleTriangleTable(&table, out->nx, ratio);
	for (int i=0; i<out->ny; i++)
	    for (int j=0; j<out->nz; j++)
		for (int k=0; k<out->nx; k++) {
		    int left=floor(k*ratio)+fZ;
		    int right=ceil(k*ratio)+fZ;
		    double lEntry=table(k,0);
		    double rEntry=table(k,1);
		    if (ifd)
			ofd->grid(k,i,j)=ifd->grid(i,j,left)*lEntry+
			    ifd->grid(i,j,right)*rEntry;
		    else if (iff)
			off->grid(k,i,j)=iff->grid(i,j,left)*lEntry+
			    iff->grid(i,j,right)*rEntry;
		    else if (ifi)
			ofi->grid(k,i,j)=ifi->grid(i,j,left)*lEntry+
			    ifi->grid(i,j,right)*rEntry;
		    else if (ifs)
			ofs->grid(k,i,j)=ifs->grid(i,j,left)*lEntry+
			    ifs->grid(i,j,right)*rEntry;
		    else if (ifc)
			ofc->grid(k,i,j)=ifc->grid(i,j,left)*lEntry+
			    ifc->grid(i,j,right)*rEntry;
		    else if (ifu)
			ofu->grid(k,i,j)=ifu->grid(i,j,left)*lEntry+
			    ifu->grid(i,j,right)*rEntry;
		    else {
			cerr << "Error - shoudln't ever get here!\n";
		    }
	}
    }
}

void printTable(Array2<double>*a) {
    cerr << "Filter Table (" << a->dim1() << "," << a->dim2() << ")\n";
    for (int i=0; i<a->dim1(); i++) {
	for (int j=0; j<a->dim2(); j++) {
	    cerr << (*a)(i,j) << " ";
	}
	cerr << "\n";
    }
    cerr << "\n";
}
  
void FieldFilter::buildUndersampleTriangleTable(Array2<double> *table,
						int size, int span, 
						double ratio) {
    for (int i=0; i<size; i++) {
        double total=0;
	double inCtr=i*ratio;
	int inIdx=inCtr-ratio;
	int j;
	for (j=0; j<span; j++, inIdx++) {
	    double val=ratio-fabs(inCtr-inIdx);
	    if (val<0) {
		val=0;
	    }
	    (*table)(i,j)=val;
	    total+=val;
	}
	for (j=0; j<span; j++) {
	    (*table)(i,j)/=total;
	}	
    }
//    printTable(table);
}

void FieldFilter::buildOversampleTriangleTable(Array2<double> *table,
					       int size, double ratio) {
    double curr=1.;
    for (int i=0; i<size; i++) {
	(*table)(i,0)=curr;
	(*table)(i,1)=1.-curr;
	curr-=ratio;
	if (curr<0) curr+=1.;
    }
//  printTable(table);
}

void FieldFilter::mitchellFilter(ScalarFieldRGBase* in, ScalarFieldRGBase* out,
				 int fZ, int lZ) {
    ScalarFieldRGdouble *ifd, *ofd;
    ScalarFieldRGfloat *iff, *off;
    ScalarFieldRGint *ifi, *ofi;
    ScalarFieldRGshort *ifs, *ofs;
    ScalarFieldRGchar *ifc, *ofc;
    ScalarFieldRGuchar *ifu, *ofu;
    ifd=in->getRGDouble();
    ofd=out->getRGDouble();
    iff=in->getRGFloat();
    off=out->getRGFloat();
    ifi=in->getRGInt();
    ofi=out->getRGInt();
    ifs=in->getRGShort();
    ofs=out->getRGShort();
    ifc=in->getRGChar();
    ofc=out->getRGChar();
    ifu=in->getRGUchar();
    ofu=out->getRGUchar();

    double ratio=(lZ-fZ)/(out->nx-1.);
    if (ratio == 1) {		// trivial filter
	for (int i=0; i<out->ny; i++)
	    for (int j=0; j<out->nz; j++)
		if (ifd) for (int k=0, kk=fZ; k<out->nx; k++, kk++)
		    ofd->grid(k,i,j)=ifd->grid(i, j, kk);
		else if (iff) for (int k=0, kk=fZ; k<out->nx; k++, kk++)
		    off->grid(k,i,j)=iff->grid(i, j, kk);
		else if (ifi) for (int k=0, kk=fZ; k<out->nx; k++, kk++)
		    ofi->grid(k,i,j)=ifi->grid(i, j, kk);
		else if (ifs) for (int k=0, kk=fZ; k<out->nx; k++, kk++)
		    ofs->grid(k,i,j)=ifs->grid(i, j, kk);
		else if (ifc) for (int k=0, kk=fZ; k<out->nx; k++, kk++)
		    ofc->grid(k,i,j)=ifc->grid(i, j, kk);
		else if (ifu) for (int k=0, kk=fZ; k<out->nx; k++, kk++)
		    ofu->grid(k,i,j)=ifu->grid(i, j, kk);
		else {
		    cerr << "Unknown SFRG type -- shouldn't ever get here!\n";
		    return;
		}
    } else if (ratio>1) {		// undersampling     big->small
	int span=ceil(2.*ratio);
	Array2<double> table(out->nx, span);
	buildUndersampleTriangleTable(&table, out->nx, span, ratio);
	for (int i=0; i<out->ny; i++)
	    for (int j=0; j<out->nz; j++)
		for (int k=0, kk=0; kk<out->nx; k++, kk++) {
		    int inPixelIdxBase=(int)((k-1)*ratio+fZ);
		    for (int l=0; l<span; l++) {
			double tEntry=table(k,l);
			int inPixelIdx=inPixelIdxBase+l;
			if (inPixelIdx<fZ) inPixelIdx=fZ;
			else if (inPixelIdx>lZ) inPixelIdx=lZ;
			if (ifd) ofd->grid(k,i,j)+=
				     ifd->grid(i,j,inPixelIdx)*tEntry;
			else if (iff) off->grid(k,i,j)+=
					  iff->grid(i,j,inPixelIdx)*tEntry;
			else if (ifi) ofi->grid(k,i,j)+=
					  ifi->grid(i,j,inPixelIdx)*tEntry;
			else if (ifs) ofs->grid(k,i,j)+=
					  ifs->grid(i,j,inPixelIdx)*tEntry;
			else if (ifc) ofc->grid(k,i,j)+=
					  ifc->grid(i,j,inPixelIdx)*tEntry;
			else if (ifu) ofu->grid(k,i,j)+=
					  ifu->grid(i,j,inPixelIdx)*tEntry;
			else {
			    cerr << "Error - shouldn't ever get here!\n";
			}
		    }
		}
    } else {			// oversampling      small->big
	Array2<double> table(out->nx, 2);
	buildOversampleTriangleTable(&table, out->nx, ratio);
	for (int i=0; i<out->ny; i++)
	    for (int j=0; j<out->nz; j++)
		for (int k=0; k<out->nx; k++) {
		    int left=floor(k*ratio)+fZ;
		    int right=ceil(k*ratio)+fZ;
		    double lEntry=table(k,0);
		    double rEntry=table(k,1);
		    if (ifd)
			ofd->grid(k,i,j)=ifd->grid(i,j,left)*lEntry+
			    ifd->grid(i,j,right)*rEntry;
		    else if (iff)
			off->grid(k,i,j)=iff->grid(i,j,left)*lEntry+
			    iff->grid(i,j,right)*rEntry;
		    else if (ifi)
			ofi->grid(k,i,j)=ifi->grid(i,j,left)*lEntry+
			    ifi->grid(i,j,right)*rEntry;
		    else if (ifs)
			ofs->grid(k,i,j)=ifs->grid(i,j,left)*lEntry+
			    ifs->grid(i,j,right)*rEntry;
		    else if (ifc)
			ofc->grid(k,i,j)=ifc->grid(i,j,left)*lEntry+
			    ifc->grid(i,j,right)*rEntry;
		    else if (ifu)
			ofu->grid(k,i,j)=ifu->grid(i,j,left)*lEntry+
			    ifu->grid(i,j,right)*rEntry;
		    else {
			cerr << "Error - shoudln't ever get here!\n";
		    }
	}
    }
}
    

void FieldFilter::buildUndersampleMitchellTable(Array2<double> *table,
						int size, int span, 
						double ratio) {
    for (int i=0; i<size; i++) {
        double total=0;
	double inCtr=i*ratio;
	int inIdx=inCtr-ratio;
	int j;
	for (j=0; j<span; j++, inIdx++) {
	    double val=ratio-fabs(inCtr-inIdx);
	    if (val<0) {
		val=0;
	    }
	    (*table)(i,j)=val;
	    total+=val;
	}
	for (j=0; j<span; j++) {
	    (*table)(i,j)/=total;
	}	
    }
//    printTable(table);
}

void FieldFilter::buildOversampleMitchellTable(Array2<double> *table,
					       int size, double ratio) {
    double curr=1.;
    for (int i=0; i<size; i++) {
	(*table)(i,0)=curr;
	(*table)(i,1)=1.-curr;
	curr-=ratio;
	if (curr<0) curr+=1.;
    }
//  printTable(table);
}

