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
#include <TCL/TCLvar.h>
#include <iostream.h>
#include <stdio.h>
#include <Malloc/Allocator.h>

static Point transformPt(double m[4][4], const Point& p) {
    double pIn[4];
    double pOut[4];
    pIn[0]=p.x(); pIn[1]=p.y(); pIn[2]=p.z(); pIn[3]=1;
    pOut[0]=pOut[1]=pOut[2]=pOut[3]=0;

    for (int i=0; i<4; i++)
	for (int j=0; j<4; j++)
	    pOut[i] += m[i][j]*pIn[j];

    return Point(pOut[0]/pOut[3], pOut[1]/pOut[3], pOut[2]/pOut[3]);
}

static void buildTransformMatrix(double m[4][4], MatrixHandle mH) {
    for (int i=0; i<4; i++)
	for (int j=0; j<4; j++)
	    m[i][j]=(*mH.get_rep())[i][j];
}

static void buildInverseMatrix(double mA[4][4], double mI[4][4]) {
    double a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p;
    a=mA[0][0]; b=mA[0][1]; c=mA[0][2]; d=mA[0][3];
    e=mA[1][0]; f=mA[1][1]; g=mA[1][2]; h=mA[1][3];
    i=mA[2][0]; j=mA[2][1]; k=mA[2][2]; l=mA[2][3];
    m=mA[3][0]; n=mA[3][1]; o=mA[3][2]; p=mA[3][3];

    double q=a*f*k*p - a*f*l*o - a*j*g*p + a*j*h*o + a*n*g*l - a*n*h*k
	- e*b*k*p + e*b*l*o + e*j*c*p - e*j*d*o - e*n*c*l + e*n*d*k
	+ i*b*g*p - i*b*h*o - i*f*c*p + i*f*d*o + i*n*c*h - i*n*d*g
	- m*b*g*l + m*b*h*k + m*f*c*l - m*f*d*k - m*j*c*h + m*j*d*g;

    if (q<0.000000001) {
	mI[0][0]=mI[1][1]=mI[2][2]=mI[3][3]=1;
	mI[1][0]=mI[1][2]=mI[1][3]=0;
	mI[2][0]=mI[2][1]=mI[2][3]=0;
	mI[3][0]=mI[3][1]=mI[3][2]=0;
	cerr << "ERROR - matrix is singular!!!\n";
	return;
    }
    mI[0][0]=(f*k*p - f*l*o - j*g*p + j*h*o + n*g*l - n*h*k)/q;
    mI[0][1]=-(b*k*p - b*l*o - j*c*p + j*d*o + n*c*l - n*d*k)/q;
    mI[0][2]=(b*g*p - b*h*o - f*c*p + f*d*o + n*c*h - n*d*g)/q;
    mI[0][3]=-(b*g*l - b*h*k - f*c*l + f*d*k + j*c*h - j*d*g)/q;

    mI[1][0]=-(e*k*p - e*l*o - i*g*p + i*h*o + m*g*l - m*h*k)/q;
    mI[1][1]=(a*k*p - a*l*o - i*c*p + i*d*o + m*c*l - m*d*k)/q;
    mI[1][2]=-(a*g*p - a*h*o - e*c*p + e*d*o + m*c*h - m*d*g)/q;
    mI[1][3]=(a*g*l - a*h*k - e*c*l + e*d*k + i*c*h - i*d*g)/q;

    mI[2][0]=(e*j*p - e*l*n - i*f*p + i*h*n + m*f*l - m*h*j)/q;
    mI[2][1]=-(a*j*p - a*l*n - i*b*p + i*d*n + m*b*l - m*d*j)/q;
    mI[2][2]=(a*f*p - a*h*n - e*b*p + e*d*n + m*b*h - m*d*f)/q;
    mI[2][3]=-(a*f*l - a*h*j - e*b*l + e*d*j + i*b*h - i*d*f)/q;

    mI[3][0]=-(e*j*o - e*k*n - i*f*o + i*g*n + m*f*k - m*g*j)/q;
    mI[3][1]=(a*j*o - a*k*n - i*b*o + i*c*n + m*b*k - m*c*j)/q;
    mI[3][2]=-(a*f*o - a*g*n - e*b*o + e*c*n + m*b*g - m*c*f)/q;
    mI[3][3]=(a*f*k - a*g*j - e*b*k + e*c*j + i*b*g - i*c*f)/q;
}
    
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

    ScalarFieldRG* sf=scinew ScalarFieldRG;
    osfh=sf;

    int nx=isfrg->nx;
    int ny=isfrg->ny;
    int nz=isfrg->nz;

    sf->resize(nx, ny, nz);
    Point min, max;
    isfrg->get_bounds(min, max);
    sf->set_bounds(min, max);

    double m[4][4], mInv[4][4];
    buildTransformMatrix(m, mIH);
    buildInverseMatrix(m, mInv);

    for (int i=0; i<nx; i++)
	for (int j=0; j<ny; j++)
	    for (int k=0; k<nz; k++) {
		Point oldp(isfrg->get_point(i,j,k));
		Point newp(transformPt(mInv, oldp));
		double val;
		if (!isfrg->interpolate(newp, val)) val=0;
		sf->grid(i,j,k)=val;
	    }

    oport->send(osfh);
    return;
}
