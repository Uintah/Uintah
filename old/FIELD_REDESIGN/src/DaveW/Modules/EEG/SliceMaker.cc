
/*
 *  SliceMaker.cc:  Create a slice for the cylinder visualization and color it
 *	according to a columnmatrix input
 *  Written by:
 *   Kris Zyp
 *   Department of Computer Science
 *   University of Utah
 *   Sept 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <SCICore/Datatypes/ColumnMatrix.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/Pt.h>
#include <SCICore/Geom/GeomTri.h>
#include <SCICore/Geom/GeomTriangles.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array3.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/Math/Trig.h>
#include <iostream>
using std::cerr;

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Math;
using namespace SCICore::Containers;
using namespace SCICore::Datatypes;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::TclInterface;

#define MAX_CLASS 6
class SliceMaker : public Module {
    ColumnMatrixIPort* icm;
    GeometryOPort* ogeom;

    void mesh_to_geom(const ColumnMatrixHandle&, GeomGroup*);
    TCLint showElems;
    TCLint showNodes;
public:
    SliceMaker(const clString& id);
    virtual ~SliceMaker();
    virtual void execute();
};

extern "C" Module* make_SliceMaker(const clString& id)
{
    return scinew SliceMaker(id);
}

SliceMaker::SliceMaker(const clString& id)
: Module("SliceMaker", id, Filter), showNodes("showNodes", id, this),
  showElems("showElems", id, this)
{
    // Create the input port
    icm=scinew ColumnMatrixIPort(this,"ColumnMatrix",ColumnMatrixIPort::Atomic);
    add_iport(icm);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

SliceMaker::~SliceMaker()
{
}
int wrapAroundAngle(int angle, int limit){  // allows angles to increment and wrap around when necessary
    if ((angle) == limit)	return 0;
    else return angle;
}

double scaleColor(double startColor, double zeroColor){
    //provide a colorful color scale instead of just grey scale
    double colorMult = 5;
    if (startColor * colorMult < zeroColor) return 0;
    if (startColor * colorMult < (zeroColor + 1)) return (startColor * colorMult - zeroColor);
    if (startColor * colorMult < (zeroColor + 2)) return (2 - startColor * colorMult + zeroColor);
    return 0;
}
void SliceMaker::execute()
{
    ColumnMatrixHandle cmH;
    Array1<float> rPos(14); 
    Array1<float> x(3),y(3);
    int nr = 14;
    int cond,r1,r2,angleInner,angleOuter,z;

    update_state(NeedData);
    if (!icm->get(cmH))
	return;
    ColumnMatrix* cm = cmH.get_rep();
    update_state(JustStarted);
    int i;
#if 0
    GeomGroup* groups[MAX_CLASS];
    for(i=0;i<MAX_CLASS;i++) groups[i] = scinew GeomGroup;
#else
    GeomTrianglesP* groups[MAX_CLASS];
    for(i=0;i<MAX_CLASS;i++) groups[i] = scinew GeomTrianglesP;
#endif
    bool have_tris[MAX_CLASS];
    int j; 
    for(j=0;j<MAX_CLASS;j++)
      have_tris[j]=true;
    // the adjusted values for the cylinder problem
    rPos[0] = 0;
    rPos[1] = 7.62;
    rPos[2] = 7.62*2;
    rPos[3] = 7.62*3;
    rPos[4] = 7.62*4;
    rPos[5] = 7.62*5;
    rPos[6] = 7.62*6;
    rPos[7] = 7.62* 7;
    rPos[8] = 7.62* 8;
    rPos[9] = 7.62*9;
    rPos[10] = 76.2;
    rPos[11] = 82.55;
    rPos[12] = 89.65;
    rPos[13] = 96.75;

    r1 = 0;
	for (r1=0; r1<nr-1; r1++) {
	    r2 = r1 + 1;
	    angleOuter = 0;
	    angleInner = 0;
	    while (angleOuter < 6 * r2) {
		if (r1 < 10)
		  cond = 0;
		else
		  if (r1 < 11)
		    cond = 1;
		  else
		    cond = 2;   
		if (angleOuter < 3 * r2)
		  cond += 3;
		if (angleOuter * r1 <= angleInner * r2) {  // move along both radiuses at the same rate, this one is a normal pie slice
		    x[0]=rPos[r1]*cos(angleInner * PI/3 / (r1+0.00000001));
		    y[0]=rPos[r1]*sin(angleInner *PI/3/(r1+0.000000001));
		    x[1]=rPos[r2]*cos(angleOuter * PI/3 / r2);
		    y[1]=rPos[r2]*sin(angleOuter * PI/3/r2);
		    x[2]=rPos[r2]*cos((angleOuter+1) * PI/3 / r2);
		    y[2]=rPos[r2]*sin((angleOuter+1) * PI/3/r2);
		    groups[cond]->add(Point(x[0],y[0],0),Point(x[1],y[1],0),Point(x[2],y[2],0));		
		    angleOuter++;
		}
		else
	  	{
		    x[0]=rPos[r1]*cos(angleInner * PI/3 / r1);
		    y[0]=rPos[r1]*sin(angleInner *PI/3/r1);
		    x[1]=rPos[r2]*cos(angleOuter * PI/3 / r2);
		    y[1]=rPos[r2]*sin(angleOuter * PI/3/r2);
		    x[2]=rPos[r1]*cos((angleInner+1) * PI/3 / r1);
		    y[2]=rPos[r1]*sin((angleInner+1) * PI/3/r1);
		    groups[cond]->add(Point(x[0],y[0],0),Point(x[1],y[1],0),Point(x[2],y[2],0));		
		    angleInner++;

	}
	} 
	}
    GeomMaterial* matlsb[MAX_CLASS];


    ogeom->delAll();
	
    MaterialHandle c[6];
    cerr << (*cm)[0] << " " << (*cm)[1] << " " << (*cm)[2] << "   ";
    cerr << (*cm)[3] << " " << (*cm)[4] << " " << (*cm)[5] << "\n";
    for (i = 0; i < 6; i++)
	c[i]=scinew Material(Color(.2,.2,.2),Color(scaleColor((*cm)[i],0),scaleColor((*cm)[i],1),scaleColor((*cm)[i],2)),Color(.5,.5,.5),20);


    for(i=0;i<MAX_CLASS;i++) {
	matlsb[i] = scinew GeomMaterial(groups[i],
					c[i%6]);

	clString tmps("Data ");
	tmps += (char) ('0' + i);

	clString tmpb("Tris ");
	tmpb += (char) ('0' + i);

	ogeom->addObj(matlsb[i],tmpb());  // and output it
    }	


}

} // End namespace Modules
} // End namespace SCIRun


//
// $Log$
// Revision 1.1.2.3  2000/11/01 23:02:22  mcole
// Fix for previous merge from trunk
//
// Revision 1.1.2.1  2000/09/28 03:20:14  mcole
// merge trunk into FIELD_REDESIGN branch
//
// Revision 1.2  2000/10/29 04:02:46  dmw
// cleaning up DaveW tree
//
// Revision 1.1  2000/09/07 20:43:11  zyp
// This module creates a disc that represents the real and guessed
// conductivity values for a cylinder.  It is for use in a demo.  It
// receives a 6 row ColumnMatrix as the input with three values being the
// real values and three values being the guess.  It outputs a geometry
// that is color coded according to these values.  The idea is that we
// can watch the conductivity values converge as we do inverse solving of
// the conductivity of the fluids in a cylinder.
//
// Revision 1.5  2000/09/07 00:12:19  zyp
// MakeScalarField.cc
//
// Revision 1.4  2000/03/17 09:29:13  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.3  1999/12/09 09:52:44  dmw
// supports more than 7 unique regions now
//
// Revision 1.2  1999/10/07 02:08:20  sparker
// use standard iostreams and complex type
//
// Revision 1.1  1999/09/05 01:15:28  dmw
// added all of the old SCIRun mesh modules
//
