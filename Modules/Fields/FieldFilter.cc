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
#include <Datatypes/ScalarFieldRGchar.h>
#include <Geometry/Point.h>
#include <TCL/TCLvar.h>
#include <Widgets/ScaledBoxWidget.h>
#include <stdio.h>
#include <math.h>
#include <iostream.h>

class FieldFilter : public Module {
    void boxFilterX(ScalarFieldRGBase *in, ScalarFieldRGBase *out, int, int, int, int);
    void boxFilterY(ScalarFieldRGBase *in, ScalarFieldRGBase *out, int, int, int, int);
    void boxFilterZ(ScalarFieldRGBase *in, ScalarFieldRGBase *out, int, int, int, int);
    void triangleFilterX(ScalarFieldRGBase *, ScalarFieldRGBase *, int, int, int, int);
    void triangleFilterY(ScalarFieldRGBase *, ScalarFieldRGBase *, int, int, int, int);
    void triangleFilterZ(ScalarFieldRGBase *, ScalarFieldRGBase *, int, int, int, int);
    void mitchellFilterX(ScalarFieldRGBase *, ScalarFieldRGBase *, int, int, int, int);
    void mitchellFilterY(ScalarFieldRGBase *, ScalarFieldRGBase *, int, int, int, int);
    void mitchellFilterZ(ScalarFieldRGBase *, ScalarFieldRGBase *, int, int, int, int);
    void filter(char dir, ScalarFieldRGBase* in, ScalarFieldRGBase* out, 
		int minx, int miny, int minz, int maxx, int maxy, int maxz);
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
};

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

void FieldFilter::filter(char dir, ScalarFieldRGBase* in, ScalarFieldRGBase* out,
			 int min_x, int min_y, int min_z, int max_x, int max_y,
			 int max_z) {
    if (lastFT == "Box") {
	if (dir == 'x') {
	    boxFilterX(in, out, min_x, min_y, min_z, max_x);
	} else if (dir == 'y') {
	    boxFilterY(in, out, min_x, min_y, min_z, max_y);
	} else {
	    boxFilterZ(in, out, min_x, min_y, min_z, max_z);
	}
    } else if (lastFT == "Triangle") {
	if (dir == 'x') {
	    triangleFilterX(in, out, min_x, min_y, min_z, max_x);
	} else if (dir == 'y') {
	    triangleFilterY(in, out, min_x, min_y, min_z, max_y);
	} else {
	    triangleFilterZ(in, out, min_x, min_y, min_z, max_z);
	}
    } else {
	if (dir == 'x') {
	    triangleFilterX(in, out, min_x, min_y, min_z, max_x);
	} else if (dir == 'y') {
	    triangleFilterY(in, out, min_x, min_y, min_z, max_y);
	} else {
	    triangleFilterZ(in, out, min_x, min_y, min_z, max_z);
	}
//	if (dir == 'x') {
//	    mitchellFilterX(in, out, min_x, min_y, min_z, max_x);
//	} else if (dir == 'y') {
//	    mitchellFilterY(in, out, min_x, min_y, min_z, max_y);
//	} else {
//	    mitchellFilterZ(in, out, min_x, min_y, min_z, max_z);
//	}
    }
}

void FieldFilter::execute() {
    ScalarFieldHandle ifh;
    if(!iField->get(ifh))
	return;
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
    if (check_widget) {
cerr << "Saw that widget moved -- recomputing!\n";
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
    int no_range=0;
    if (((max_x-min_x) == 0) || ((max_y-min_y) == 0) || ((max_z-min_z) == 0))
	no_range=1;
    if (check_widget || no_range || 
	(isf->nx == ox.get() && isf->ny == oy.get() && isf->nz == oz.get() &&
	nx.get() == lastX && ny.get() == lastY && nz.get() == lastZ &&
	sameInput.get() && lastFT == filterType.get())) {
	oField->send(oFldHandle);
	return;
    }
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
    ScalarFieldRGchar *ifc=isf->getRGChar();

    ScalarFieldRGBase *fldX, *fldY;
    if (ifd) {
	ScalarFieldRGdouble *fX, *fY, *of;
	fX=new ScalarFieldRGdouble();
	fX->resize(lastX, isf->ny, isf->nz);
	fX->grid.initialize(0);
	fY=new ScalarFieldRGdouble();
	fY->resize(lastX, lastY, isf->nz);
	fY->grid.initialize(0);
	oFldHandle=of=new ScalarFieldRGdouble();
	of->resize(lastX, lastY, lastZ);
	of->grid.initialize(0);
	fldX=fX;
	fldY=fY;
	osf=of;
    } else if (iff) {
	ScalarFieldRGfloat *fX, *fY, *of;
	fX=new ScalarFieldRGfloat();
	fX->resize(lastX, isf->ny, isf->nz);
	fX->grid.initialize(0);
	fY=new ScalarFieldRGfloat();
	fY->resize(lastX, lastY, isf->nz);
	fY->grid.initialize(0);
	oFldHandle=of=new ScalarFieldRGfloat();
	of->resize(lastX, lastY, lastZ);
	of->grid.initialize(0);
	fldX=fX;
	fldY=fY;
	osf=of;
    } else if (ifi) {
	ScalarFieldRGint *fX, *fY, *of;
	fX=new ScalarFieldRGint();
	fX->resize(lastX, isf->ny, isf->nz);
	fX->grid.initialize(0);
	fY=new ScalarFieldRGint();
	fY->resize(lastX, lastY, isf->nz);
	fY->grid.initialize(0);
	oFldHandle=of=new ScalarFieldRGint();
	of->resize(lastX, lastY, lastZ);
	of->grid.initialize(0);
	fldX=fX;
	fldY=fY;
	osf=of;
    } else {					// must be char field
	ScalarFieldRGchar *fX, *fY, *of;
	fX=new ScalarFieldRGchar();
	fX->resize(lastX, isf->ny, isf->nz);
	fX->grid.initialize(0);
	fY=new ScalarFieldRGchar();
	fY->resize(lastX, lastY, isf->nz);
	fY->grid.initialize(0);
	oFldHandle=of=new ScalarFieldRGchar();
	of->resize(lastX, lastY, lastZ);
	of->grid.initialize(0);
	fldX=fX;
	fldY=fY;
	osf=of;
    }
    fldX->set_bounds(p1, p2);
    fldY->set_bounds(p1, p2);
    osf->set_bounds(p1, p2);
    filter('x', isf, fldX, min_x, min_y, min_z, max_x, max_y, max_z);
    filter('y', fldX, fldY, min_x, min_y, min_z, max_x, max_y, max_z);
    filter('z', fldY, osf, min_x, min_y, min_z, max_x, max_y, max_z);
    oField->send(oFldHandle);
}

void FieldFilter::boxFilterX(ScalarFieldRGBase* in, ScalarFieldRGBase* out,
				  int fX, int fY, int fZ, int lX) {
    ScalarFieldRGdouble *ifd, *ofd;
    ScalarFieldRGfloat *iff, *off;
    ScalarFieldRGint *ifi, *ofi;
    ScalarFieldRGchar *ifc, *ofc;
    ifd=in->getRGDouble();
    ofd=out->getRGDouble();
    iff=in->getRGFloat();
    off=out->getRGFloat();
    ifi=in->getRGInt();
    ofi=out->getRGInt();
    ifc=in->getRGChar();
    ofc=out->getRGChar();
    
    double ratio=1./((out->nx-1.)/(lX-fX));
cerr << "XRatio = " <<ratio<<"\n";
    double curr=0;

    for (int i=0; i<out->nx; i++, curr+=ratio) {
	for (int j=0, jj=fY; j<out->ny; j++, jj++) {
	    for (int k=0, kk=fZ; k<out->nz; k++, kk++) {
		if (ifd)
		    ofd->grid(i,j,k)=ifd->grid((int)curr+fX,jj,kk);
		else if (iff)
		    off->grid(i,j,k)=iff->grid((int)curr+fX,jj,kk);
		else if (ifi)
		    ofi->grid(i,j,k)=ifi->grid((int)curr+fX,jj,kk);
		else 
		    ofc->grid(i,j,k)=ifc->grid((int)curr+fX,jj,kk);
	    }
	}
    }
}

void FieldFilter::boxFilterY(ScalarFieldRGBase* in, ScalarFieldRGBase* out,
				  int fX, int fY, int fZ, int lY) {
    ScalarFieldRGdouble *ifd, *ofd;
    ScalarFieldRGfloat *iff, *off;
    ScalarFieldRGint *ifi, *ofi;
    ScalarFieldRGchar *ifc, *ofc;
    ifd=in->getRGDouble();
    ofd=out->getRGDouble();
    iff=in->getRGFloat();
    off=out->getRGFloat();
    ifi=in->getRGInt();
    ofi=out->getRGInt();
    ifc=in->getRGChar();
    ofc=out->getRGChar();
    
    double ratio=1./((out->ny-1.)/(lY-fY));
cerr << "YRatio = " <<ratio<<"\n";
    double curr=0;
    for (int j=0; j<out->ny; j++, curr+=ratio) {
	for (int i=0, ii=fX; i<out->nx; i++, ii++) {
	    for (int k=0, kk=fZ; k<out->nz; k++, kk++) {
		if (ifd)
		    ofd->grid(i,j,k)=ifd->grid(ii,(int)curr+fY,kk);
		else if (iff)
		    off->grid(i,j,k)=iff->grid(ii,(int)curr+fY,kk);
		else if (ifi)
		    ofi->grid(i,j,k)=ifi->grid(ii,(int)curr+fY,kk);
		else 
		    ofc->grid(i,j,k)=ifc->grid(ii,(int)curr+fY,kk);
	    }
	}
    }
}

void FieldFilter::boxFilterZ(ScalarFieldRGBase* in, ScalarFieldRGBase* out,
			     int fX, int fY, int fZ, int lZ) {
    ScalarFieldRGdouble *ifd, *ofd;
    ScalarFieldRGfloat *iff, *off;
    ScalarFieldRGint *ifi, *ofi;
    ScalarFieldRGchar *ifc, *ofc;
    ifd=in->getRGDouble();
    ofd=out->getRGDouble();
    iff=in->getRGFloat();
    off=out->getRGFloat();
    ifi=in->getRGInt();
    ofi=out->getRGInt();
    ifc=in->getRGChar();
    ofc=out->getRGChar();
    
    double ratio=1./((out->nz-1.)/(lZ-fZ));
cerr << "ZRatio = " <<ratio<<"\n";
    double curr=0;
    for (int k=0; k<out->nz; k++, curr+=ratio) {
	for (int i=0, ii=fX; i<out->nx; i++, ii++) {
	    for (int j=0, jj=fY; j<out->ny; j++, jj++) {
		if (ifd)
		    ofd->grid(i,j,k)=ifd->grid(ii,jj,(int)curr+fZ);
		else if (iff)
		    off->grid(i,j,k)=iff->grid(ii,jj,(int)curr+fZ);
		else if (ifi)
		    ofi->grid(i,j,k)=ifi->grid(ii,jj,(int)curr+fZ);
		else 
		    ofc->grid(i,j,k)=ifc->grid(ii,jj,(int)curr+fZ);
	    }	
	}
    }
}

void FieldFilter::triangleFilterX(ScalarFieldRGBase* in, ScalarFieldRGBase* out,
				  int fX, int fY, int fZ, int lX) {
    ScalarFieldRGdouble *ifd, *ofd;
    ScalarFieldRGfloat *iff, *off;
    ScalarFieldRGint *ifi, *ofi;
    ScalarFieldRGchar *ifc, *ofc;
    ifd=in->getRGDouble();
    ofd=out->getRGDouble();
    iff=in->getRGFloat();
    off=out->getRGFloat();
    ifi=in->getRGInt();
    ofi=out->getRGInt();
    ifc=in->getRGChar();
    ofc=out->getRGChar();
    
    double ratio=(out->nx-1.)/(lX-fX);
    if (ratio == 1) {		// trivial filter
	for (int i=0, ii=fX; i<out->nx; i++, ii++)
	    for (int j=0, jj=fY; j<out->ny; j++, jj++)
		for (int k=0, kk=fZ; k<out->nz; k++, kk++)
		    if (ifd)
			ofd->grid(i,j,k)=ifd->grid(ii, jj, kk);
		    else if (iff)
			off->grid(i,j,k)=iff->grid(ii, jj, kk);
		    else if (ifi)
			ofi->grid(i,j,k)=ifi->grid(ii, jj, kk);
		    else 
			ofc->grid(i,j,k)=ifc->grid(ii, jj, kk);
	return;
    }
    if (ratio<1) {		// undersampling     big->small
	int span=ceil(2./ratio);
	Array2<double> table(out->nx, span);
	buildUndersampleTriangleTable(&table, out->nx, span, ratio);
	for (int i=0; i<out->nx; i++) {
	    for (int l=0; l<span; l++) {
		double tEntry=table(i,l);
		int inPixelIdx=(int)((i-1)/ratio+fX+l);
		if (inPixelIdx<fX) inPixelIdx=fX;
		else if (inPixelIdx>lX) inPixelIdx=lX;
		for (int j=0, jj=fY; j<out->ny; j++, jj++)
		    for (int k=0, kk=fZ; k<out->nz; k++, kk++) 
			if (ifd)
			    ofd->grid(i,j,k)+=ifd->grid(inPixelIdx,jj,kk)*tEntry;
			else if (iff)
			    off->grid(i,j,k)+=iff->grid(inPixelIdx,jj,kk)*tEntry;
			else if (ifi)
			    ofi->grid(i,j,k)+=ifi->grid(inPixelIdx,jj,kk)*tEntry;
			else 
			    ofc->grid(i,j,k)+=ifc->grid(inPixelIdx,jj,kk)*tEntry;
	    }
	}
    } else {			// oversampling      small->big
	Array2<double> table(out->nx, 2);
	buildOversampleTriangleTable(&table, out->nx, ratio);
	for (int i=0; i<out->nx; i++) {
	    int left=floor(i/ratio)+fX;
	    int right=ceil(i/ratio)+fX;
	    double lEntry=table(i,0);
	    double rEntry=table(i,1);
	    for (int j=0, jj=fY; j<out->ny; j++, jj++)
		for (int k=0, kk=fZ; k<out->nz; k++, kk++)
		    if (ifd)
			ofd->grid(i,j,k)=ifd->grid(left,jj,kk)*lEntry+
			    ifd->grid(right,jj,kk)*rEntry;
		    else if (iff)
			off->grid(i,j,k)=iff->grid(left,jj,kk)*lEntry+
			    iff->grid(right,jj,kk)*rEntry;
		    else if (ifi)
			ofi->grid(i,j,k)=ifi->grid(left,jj,kk)*lEntry+
			    ifi->grid(right,jj,kk)*rEntry;
		    else 
			ofc->grid(i,j,k)=ifc->grid(left,jj,kk)*lEntry+
			    ifc->grid(right,jj,kk)*rEntry;
	}
    }
}

void FieldFilter::triangleFilterY(ScalarFieldRGBase* in, ScalarFieldRGBase* out,
				  int fX, int fY, int fZ, int lY) {
    ScalarFieldRGdouble *ifd, *ofd;
    ScalarFieldRGfloat *iff, *off;
    ScalarFieldRGint *ifi, *ofi;
    ScalarFieldRGchar *ifc, *ofc;
    ifd=in->getRGDouble();
    ofd=out->getRGDouble();
    iff=in->getRGFloat();
    off=out->getRGFloat();
    ifi=in->getRGInt();
    ofi=out->getRGInt();
    ifc=in->getRGChar();
    ofc=out->getRGChar();
    
    double ratio=(out->ny-1.)/(lY-fY);
    if (ratio == 1) {		// trivial filter
	for (int i=0, ii=fX; i<out->nx; i++, ii++)
	    for (int j=0, jj=fY; j<out->ny; j++, jj++)
		for (int k=0, kk=fZ; k<out->nz; k++, kk++)
		    if (ifd)
			ofd->grid(i,j,k)=ifd->grid(ii, jj, kk);
		    else if (iff)
			off->grid(i,j,k)=iff->grid(ii, jj, kk);
		    else if (ifi)
			ofi->grid(i,j,k)=ifi->grid(ii, jj, kk);
		    else 
			ofc->grid(i,j,k)=ifc->grid(ii, jj, kk);
	return;
    }
    if (ratio<1) {		// undersampling     big->small
	int span=ceil(2./ratio);
	Array2<double> table(out->ny, span);
	buildUndersampleTriangleTable(&table, out->ny, span, ratio);
	for (int j=0; j<out->ny; j++) {
	    for (int l=0; l<span; l++) {
		double tEntry=table(j,l);
		int inPixelIdx=(int)((j-1)/ratio+fY+l);
		if (inPixelIdx<fY) inPixelIdx=fY;
		else if (inPixelIdx>lY) inPixelIdx=lY;
		for (int i=0, ii=fX; i<out->nx; i++, ii++)
		    for (int k=0, kk=fZ; k<out->nz; k++, kk++) 
			if (ifd)
			    ofd->grid(i,j,k)+=ifd->grid(ii,inPixelIdx,kk)*tEntry;
			else if (iff)
			    off->grid(i,j,k)+=iff->grid(ii,inPixelIdx,kk)*tEntry;
			else if (ifi)
			    ofi->grid(i,j,k)+=ifi->grid(ii,inPixelIdx,kk)*tEntry;
			else 
			    ofc->grid(i,j,k)+=ifc->grid(ii,inPixelIdx,kk)*tEntry;
	    }
	}
    } else {			// oversampling      small->big
	Array2<double> table(out->ny, 2);
	buildOversampleTriangleTable(&table, out->ny, ratio);
	for (int j=0; j<out->ny; j++) {
	    int left=floor(j/ratio)+fY;
	    int right=ceil(j/ratio)+fY;
	    double lEntry=table(j,0);
	    double rEntry=table(j,1);
	    for (int i=0, ii=fY; i<out->nx; i++, ii++)
		for (int k=0, kk=fZ; k<out->nz; k++, kk++)
		    if (ifd)
			ofd->grid(i,j,k)=ifd->grid(ii,left,kk)*lEntry+
			    ifd->grid(ii,right,kk)*rEntry;
		    else if (iff)
			off->grid(i,j,k)=iff->grid(ii,left,kk)*lEntry+
			    iff->grid(ii,right,kk)*rEntry;
		    else if (ifi)
			ofi->grid(i,j,k)=ifi->grid(ii,left,kk)*lEntry+
			    ifi->grid(ii,right,kk)*rEntry;
		    else 
			ofc->grid(i,j,k)=ifc->grid(ii,left,kk)*lEntry+
			    ifc->grid(ii,right,kk)*rEntry;
	}
    }
}

void FieldFilter::triangleFilterZ(ScalarFieldRGBase* in, ScalarFieldRGBase* out,
				  int fX, int fY, int fZ, int lZ) {
    ScalarFieldRGdouble *ifd, *ofd;
    ScalarFieldRGfloat *iff, *off;
    ScalarFieldRGint *ifi, *ofi;
    ScalarFieldRGchar *ifc, *ofc;
    ifd=in->getRGDouble();
    ofd=out->getRGDouble();
    iff=in->getRGFloat();
    off=out->getRGFloat();
    ifi=in->getRGInt();
    ofi=out->getRGInt();
    ifc=in->getRGChar();
    ofc=out->getRGChar();
    
    double ratio=(out->nz-1.)/(lZ-fZ);
    if (ratio == 1) {		// trivial filter
	for (int i=0, ii=fX; i<out->nx; i++, ii++)
	    for (int j=0, jj=fY; j<out->ny; j++, jj++)
		for (int k=0, kk=fZ; k<out->nz; k++, kk++)
		    if (ifd)
			ofd->grid(i,j,k)=ifd->grid(ii, jj, kk);
		    else if (iff)
			off->grid(i,j,k)=iff->grid(ii, jj, kk);
		    else if (ifi)
			ofi->grid(i,j,k)=ifi->grid(ii, jj, kk);
		    else 
			ofc->grid(i,j,k)=ifc->grid(ii, jj, kk);
	return;
    }
    if (ratio<1) {		// undersampling     big->small
	int span=ceil(2./ratio);
	Array2<double> table(out->nz, span);
	buildUndersampleTriangleTable(&table, out->nz, span, ratio);
	for (int k=0; k<out->nz; k++) {
	    for (int l=0; l<span; l++) {
		double tEntry=table(k,l);
		int inPixelIdx=(int)((k-1)/ratio+fZ+l);
		if (inPixelIdx<fZ) inPixelIdx=fZ;
		else if (inPixelIdx>lZ) inPixelIdx=lZ;
		for (int i=0, ii=fX; i<out->nx; i++, ii++) 
		    for (int j=0, jj=fY; j<out->ny; j++, jj++)
			if (ifd)
			    ofd->grid(i,j,k)+=ifd->grid(ii,jj,inPixelIdx)*tEntry;
			else if (iff)
			    off->grid(i,j,k)+=iff->grid(ii,jj,inPixelIdx)*tEntry;
			else if (ifi)
			    ofi->grid(i,j,k)+=ifi->grid(ii,jj,inPixelIdx)*tEntry;
			else 
			    ofc->grid(i,j,k)+=ifc->grid(ii,jj,inPixelIdx)*tEntry;
	    }
	}
    } else {			// oversampling      small->big
	Array2<double> table(out->nz, 2);
	buildOversampleTriangleTable(&table, out->nz, ratio);
	for (int k=0; k<out->nz; k++) {
	    int left=floor(k/ratio)+fZ;
	    int right=ceil(k/ratio)+fZ;
	    double lEntry=table(k,0);
	    double rEntry=table(k,1);
	    for (int i=0, ii=fX; i<out->nx; i++, ii++)
		for (int j=0, jj=fY; j<out->ny; j++, jj++)
		    if (ifd)
			ofd->grid(i,j,k)=ifd->grid(ii,jj,left)*lEntry+
			    ifd->grid(ii,jj,right)*rEntry;
		    else if (iff)
			off->grid(i,j,k)=iff->grid(ii,jj,left)*lEntry+
			    iff->grid(ii,jj,right)*rEntry;
		    else if (ifi)
			ofi->grid(i,j,k)=ifi->grid(ii,jj,left)*lEntry+
			    ifi->grid(ii,jj,right)*rEntry;
		    else 
			ofc->grid(i,j,k)=ifc->grid(ii,jj,left)*lEntry+
			    ifc->grid(ii,jj,right)*rEntry;
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
    double invRatio=1./ratio;
    for (int i=0; i<size; i++) {
        double total=0;
	double inCtr=i*invRatio;
	int inIdx=inCtr-invRatio;
	int j;
	for (j=0; j<span; j++, inIdx++) {
	    double val=invRatio-fabs(inCtr-inIdx);
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
    double inverse=1./ratio;
    double curr=1.;
    for (int i=0; i<size; i++) {
	(*table)(i,0)=curr;
	(*table)(i,1)=1.-curr;
	curr-=inverse;
	if (curr<0) curr+=1.;
    }
}

void FieldFilter::mitchellFilterX(ScalarFieldRGBase* in, ScalarFieldRGBase* out,
				  int fX, int fY, int fZ, int lX) {
    ScalarFieldRGdouble *ifd, *ofd;
    ScalarFieldRGfloat *iff, *off;
    ScalarFieldRGint *ifi, *ofi;
    ScalarFieldRGchar *ifc, *ofc;
    ifd=in->getRGDouble();
    ofd=out->getRGDouble();
    iff=in->getRGFloat();
    off=out->getRGFloat();
    ifi=in->getRGInt();
    ofi=out->getRGInt();
    ifc=in->getRGChar();
    ofc=out->getRGChar();
    
    double ratio=(out->nx-1.)/(lX-fX);
    if (ratio == 1) {		// trivial filter
	for (int i=0, ii=fX; i<out->nx; i++, ii++)
	    for (int j=0, jj=fY; j<out->ny; j++, jj++)
		for (int k=0, kk=fZ; k<out->nz; k++, kk++)
		    if (ifd)
			ofd->grid(i,j,k)=ifd->grid(ii, jj, kk);
		    else if (iff)
			off->grid(i,j,k)=iff->grid(ii, jj, kk);
		    else if (ifi)
			ofi->grid(i,j,k)=ifi->grid(ii, jj, kk);
		    else 
			ofc->grid(i,j,k)=ifc->grid(ii, jj, kk);
	return;
    }
    if (ratio<1) {		// undersampling     big->small
	int span=ceil(4./ratio);
	Array2<double> table(out->nx, span);
	buildUndersampleMitchellTable(&table, out->nx, span, ratio);
	for (int i=0; i<out->nx; i++) {
	    for (int l=0; l<span; l++) {
		double tEntry=table(i,l);
		int inPixelIdx=(int)((i-1)/ratio+fX+l);
		if (inPixelIdx<fX) inPixelIdx=fX;
		else if (inPixelIdx>lX) inPixelIdx=lX;
		for (int j=0, jj=fY; j<out->ny; j++, jj++)
		    for (int k=0, kk=fZ; k<out->nz; k++, kk++) 
			if (ifd)
			    ofd->grid(i,j,k)+=ifd->grid(inPixelIdx,jj,kk)*tEntry;
			else if (iff)
			    off->grid(i,j,k)+=iff->grid(inPixelIdx,jj,kk)*tEntry;
			else if (ifi)
			    ofi->grid(i,j,k)+=ifi->grid(inPixelIdx,jj,kk)*tEntry;
			else 
			    ofc->grid(i,j,k)+=ifc->grid(inPixelIdx,jj,kk)*tEntry;
	    }
	}
    } else {			// oversampling      small->big
	int span=ceil(4./ratio);
	Array2<double> table(out->nx, 2);
	buildOversampleMitchellTable(&table, out->nx, ratio);
	for (int i=0; i<out->nx; i++) {
	    int left=floor(i/ratio)+fX;
	    int right=ceil(i/ratio)+fX;
	    double lEntry=table(i,0);
	    double rEntry=table(i,1);
	    for (int j=0, jj=fY; j<out->ny; j++, jj++)
		for (int k=0, kk=fZ; k<out->nz; k++, kk++)
		    if (ifd)
			ofd->grid(i,j,k)=ifd->grid(left,jj,kk)*lEntry+
			    ifd->grid(right,jj,kk)*rEntry;
		    else if (iff)
			off->grid(i,j,k)=iff->grid(left,jj,kk)*lEntry+
			    iff->grid(right,jj,kk)*rEntry;
		    else if (ifi)
			ofi->grid(i,j,k)=ifi->grid(left,jj,kk)*lEntry+
			    ifi->grid(right,jj,kk)*rEntry;
		    else 
			ofc->grid(i,j,k)=ifc->grid(left,jj,kk)*lEntry+
			    ifc->grid(right,jj,kk)*rEntry;
	}
    }
}

void FieldFilter::mitchellFilterY(ScalarFieldRGBase* in, ScalarFieldRGBase* out,
				  int fX, int fY, int fZ, int lY) {
    ScalarFieldRGdouble *ifd, *ofd;
    ScalarFieldRGfloat *iff, *off;
    ScalarFieldRGint *ifi, *ofi;
    ScalarFieldRGchar *ifc, *ofc;
    ifd=in->getRGDouble();
    ofd=out->getRGDouble();
    iff=in->getRGFloat();
    off=out->getRGFloat();
    ifi=in->getRGInt();
    ofi=out->getRGInt();
    ifc=in->getRGChar();
    ofc=out->getRGChar();
    
    double ratio=(out->ny-1.)/(lY-fY);
    if (ratio == 1) {		// trivial filter
	for (int i=0, ii=fX; i<out->nx; i++, ii++)
	    for (int j=0, jj=fY; j<out->ny; j++, jj++)
		for (int k=0, kk=fZ; k<out->nz; k++, kk++)
		    if (ifd)
			ofd->grid(i,j,k)=ifd->grid(ii, jj, kk);
		    else if (iff)
			off->grid(i,j,k)=iff->grid(ii, jj, kk);
		    else if (ifi)
			ofi->grid(i,j,k)=ifi->grid(ii, jj, kk);
		    else 
			ofc->grid(i,j,k)=ifc->grid(ii, jj, kk);
	return;
    }
    if (ratio<1) {		// undersampling     big->small
	int span=ceil(2./ratio);
	Array2<double> table(out->ny, span);
	buildUndersampleMitchellTable(&table, out->ny, span, ratio);
	for (int j=0; j<out->ny; j++) {
	    for (int l=0; l<span; l++) {
		double tEntry=table(j,l);
		int inPixelIdx=(int)((j-1)/ratio+fY+l);
		if (inPixelIdx<fY) inPixelIdx=fY;
		else if (inPixelIdx>lY) inPixelIdx=lY;
		for (int i=0, ii=fX; i<out->nx; i++, ii++)
		    for (int k=0, kk=fZ; k<out->nz; k++, kk++) 
			if (ifd)
			    ofd->grid(i,j,k)+=ifd->grid(ii,inPixelIdx,kk)*tEntry;
			else if (iff)
			    off->grid(i,j,k)+=iff->grid(ii,inPixelIdx,kk)*tEntry;
			else if (ifi)
			    ofi->grid(i,j,k)+=ifi->grid(ii,inPixelIdx,kk)*tEntry;
			else 
			    ofc->grid(i,j,k)+=ifc->grid(ii,inPixelIdx,kk)*tEntry;
	    }
	}
    } else {			// oversampling      small->big
	Array2<double> table(out->ny, 2);
	buildOversampleMitchellTable(&table, out->ny, ratio);
	for (int j=0; j<out->ny; j++) {
	    int left=floor(j/ratio)+fY;
	    int right=ceil(j/ratio)+fY;
	    double lEntry=table(j,0);
	    double rEntry=table(j,1);
	    for (int i=0, ii=fY; i<out->nx; i++, ii++)
		for (int k=0, kk=fZ; k<out->nz; k++, kk++)
		    if (ifd)
			ofd->grid(i,j,k)=ifd->grid(ii,left,kk)*lEntry+
			    ifd->grid(ii,right,kk)*rEntry;
		    else if (iff)
			off->grid(i,j,k)=iff->grid(ii,left,kk)*lEntry+
			    iff->grid(ii,right,kk)*rEntry;
		    else if (ifi)
			ofi->grid(i,j,k)=ifi->grid(ii,left,kk)*lEntry+
			    ifi->grid(ii,right,kk)*rEntry;
		    else 
			ofc->grid(i,j,k)=ifc->grid(ii,left,kk)*lEntry+
			    ifc->grid(ii,right,kk)*rEntry;
	}
    }
}

void FieldFilter::mitchellFilterZ(ScalarFieldRGBase* in, ScalarFieldRGBase* out,
				  int fX, int fY, int fZ, int lZ) {
    ScalarFieldRGdouble *ifd, *ofd;
    ScalarFieldRGfloat *iff, *off;
    ScalarFieldRGint *ifi, *ofi;
    ScalarFieldRGchar *ifc, *ofc;
    ifd=in->getRGDouble();
    ofd=out->getRGDouble();
    iff=in->getRGFloat();
    off=out->getRGFloat();
    ifi=in->getRGInt();
    ofi=out->getRGInt();
    ifc=in->getRGChar();
    ofc=out->getRGChar();
    
    double ratio=(out->nz-1.)/(lZ-fZ);
    if (ratio == 1) {		// trivial filter
	for (int i=0, ii=fX; i<out->nx; i++, ii++)
	    for (int j=0, jj=fY; j<out->ny; j++, jj++)
		for (int k=0, kk=fZ; k<out->nz; k++, kk++)
		    if (ifd)
			ofd->grid(i,j,k)=ifd->grid(ii, jj, kk);
		    else if (iff)
			off->grid(i,j,k)=iff->grid(ii, jj, kk);
		    else if (ifi)
			ofi->grid(i,j,k)=ifi->grid(ii, jj, kk);
		    else 
			ofc->grid(i,j,k)=ifc->grid(ii, jj, kk);
	return;
    }
    if (ratio<1) {		// undersampling     big->small
	int span=ceil(2./ratio);
	Array2<double> table(out->nz, span);
	buildUndersampleMitchellTable(&table, out->nz, span, ratio);
	for (int k=0; k<out->nz; k++) {
	    for (int l=0; l<span; l++) {
		double tEntry=table(k,l);
		int inPixelIdx=(int)((k-1)/ratio+fZ+l);
		if (inPixelIdx<fZ) inPixelIdx=fZ;
		else if (inPixelIdx>lZ) inPixelIdx=lZ;
		for (int i=0, ii=fX; i<out->nx; i++, ii++) 
		    for (int j=0, jj=fY; j<out->ny; j++, jj++)
			if (ifd)
			    ofd->grid(i,j,k)+=ifd->grid(ii,jj,inPixelIdx)*tEntry;
			else if (iff)
			    off->grid(i,j,k)+=iff->grid(ii,jj,inPixelIdx)*tEntry;
			else if (ifi)
			    ofi->grid(i,j,k)+=ifi->grid(ii,jj,inPixelIdx)*tEntry;
			else 
			    ofc->grid(i,j,k)+=ifc->grid(ii,jj,inPixelIdx)*tEntry;
	    }
	}
    } else {			// oversampling      small->big
	Array2<double> table(out->nz, 2);
	buildOversampleMitchellTable(&table, out->nz, ratio);
	for (int k=0; k<out->nz; k++) {
	    int left=floor(k/ratio)+fZ;
	    int right=ceil(k/ratio)+fZ;
	    double lEntry=table(k,0);
	    double rEntry=table(k,1);
	    for (int i=0, ii=fX; i<out->nx; i++, ii++)
		for (int j=0, jj=fY; j<out->ny; j++, jj++)
		    if (ifd)
			ofd->grid(i,j,k)=ifd->grid(ii,jj,left)*lEntry+
			    ifd->grid(ii,jj,right)*rEntry;
		    else if (iff)
			off->grid(i,j,k)=iff->grid(ii,jj,left)*lEntry+
			    iff->grid(ii,jj,right)*rEntry;
		    else if (ifi)
			ofi->grid(i,j,k)=ifi->grid(ii,jj,left)*lEntry+
			    ifi->grid(ii,jj,right)*rEntry;
		    else 
			ofc->grid(i,j,k)=ifc->grid(ii,jj,left)*lEntry+
			    ifc->grid(ii,jj,right)*rEntry;
	}
    }
}

void FieldFilter::buildUndersampleMitchellTable(Array2<double> *table,
						int size, int span, 
						double ratio) {
    double invRatio=1./ratio;
    for (int i=0; i<size; i++) {
        double total=0;
	double inCtr=i*invRatio;
	int inIdx=inCtr-invRatio;
	int j;
	for (j=0; j<span; j++, inIdx++) {
	    double val=invRatio-fabs(inCtr-inIdx);
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
    printTable(table);
}

void FieldFilter::buildOversampleMitchellTable(Array2<double> *table,
					       int size, double ratio) {
    double inverse=1./ratio;
    double curr=1.;
    for (int i=0; i<size; i++) {
	(*table)(i,0)=curr;
	(*table)(i,1)=1.-curr;
	curr-=inverse;
	if (curr<0) curr+=1.;
    }
}
