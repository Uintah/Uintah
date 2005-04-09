/*
 *  TransformField.cc:  Rotate and flip field to get it into "standard" view
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Dataflow/Module.h>
#include <Datatypes/Matrix.h>
#include <Datatypes/MatrixPort.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Geometry/BBox.h>
#include <Geometry/Transform.h>
#include <TCL/TCLvar.h>
#include <iostream.h>
#include <stdio.h>
#include <Malloc/Allocator.h>

static void MatToTransform(MatrixHandle mH, Transform& t) {
    double a[16];
    double *p=&(a[0]);
    for (int i=0; i<4; i++)
	for (int j=0; j<4; j++)
	    *p++=(*mH.get_rep())[i][j];
    t.set(a);
}

#if 0
static void TransformToMat(Transform& t, MatrixHandle mH) {
    double a[16];
    t.get(a);
    double *p=&(a[0]);
    for (int i=0; i<4; i++)
	for (int j=0; j<4; j++)
	    (*mH.get_rep())[i][j]=*p++;
}
#endif 

class TransformField : public Module {
    ScalarFieldIPort *iport;
    MatrixIPort *imat;
    ScalarFieldOPort *oport;
    ScalarFieldHandle osfh;
    int matGen;
    int sfGen;
public:
    TransformField(const clString& id);
    TransformField(const TransformField&, int deep);
    virtual ~TransformField();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_TransformField(const clString& id)
{
    return scinew TransformField(id);
}
}

static clString module_name("TransformField");

TransformField::TransformField(const clString& id)
: Module("TransformField", id, Source), matGen(-1), sfGen(-1)
{
    // Create the input port
    iport = scinew ScalarFieldIPort(this, "SFRG", ScalarFieldIPort::Atomic);
    add_iport(iport);
    imat = scinew MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(imat);
    oport = scinew ScalarFieldOPort(this, "SFRG",ScalarFieldIPort::Atomic);
    add_oport(oport);
}

TransformField::TransformField(const TransformField& copy, int deep)
: Module(copy, deep), matGen(-1), sfGen(-1)
{
    NOT_FINISHED("TransformField::TransformField");
}

TransformField::~TransformField()
{
}

Module* TransformField::clone(int deep)
{
    return scinew TransformField(*this, deep);
}

void TransformField::execute()
{
    ScalarFieldHandle sfIH;
    iport->get(sfIH);
    if (!sfIH.get_rep()) return;
    ScalarFieldRG* isfrg=sfIH->getRG();
    if (!isfrg) return;

    MatrixHandle mIH;
    imat->get(mIH);
    if (!mIH.get_rep()) return;
    if (matGen == mIH->generation && sfGen == sfIH->generation) return;
    if ((mIH->nrows() != 4) || (mIH->ncols() != 4)) return;
    Transform t;
    MatToTransform(mIH, t);

    ScalarFieldRG* sf=scinew ScalarFieldRG;
    osfh=sf;
    int nx=isfrg->nx;
    int ny=isfrg->ny;
    int nz=isfrg->nz;
    sf->resize(nx, ny, nz);
    Point min, max;
    isfrg->get_bounds(min, max);
    sf->set_bounds(min, max);

    for (int i=0; i<nx; i++)
	for (int j=0; j<ny; j++)
	    for (int k=0; k<nz; k++) {
		Point oldp(isfrg->get_point(i,j,k));
		Point newp(t.unproject(oldp));
		double val;
		if (!isfrg->interpolate(newp, val)) val=0;
		sf->grid(i,j,k)=val;
	    }

    oport->send(osfh);
    return;
}
