/*
 *  ErrorMetric.cc: Compute and visualize error between two vectors
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;
#include <sstream>
using std::ostringstream;

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Math;
using namespace SCICore::TclInterface;

class ErrorMetric : public Module {
    ColumnMatrixIPort* ivec1P;
    ColumnMatrixIPort* ivec2P;
    ColumnMatrixOPort* errorP;
    TCLstring methodTCL;
    TCLstring pTCL;
public:
    ErrorMetric(const clString& id);
    virtual ~ErrorMetric();
    virtual void execute();
};

extern "C" Module* make_ErrorMetric(const clString& id)
{
    return scinew ErrorMetric(id);
}

ErrorMetric::ErrorMetric(const clString& id)
: Module("ErrorMetric", id, Filter), methodTCL("methodTCL", id, this),
    pTCL("pTCL", id, this)
{
    // Create the input port
    ivec1P=scinew ColumnMatrixIPort(this, "Vec1",ColumnMatrixIPort::Atomic);
    add_iport(ivec1P);
    ivec2P=scinew ColumnMatrixIPort(this, "Vec2", ColumnMatrixIPort::Atomic);
    add_iport(ivec2P);

    // Create the output ports
    errorP=scinew ColumnMatrixOPort(this,"Error out",ColumnMatrixIPort::Atomic);
    add_oport(errorP);
}

ErrorMetric::~ErrorMetric()
{
}

void ErrorMetric::execute()
{
     ColumnMatrixHandle ivec1H;
     ColumnMatrix* ivec1;
     if (!ivec1P->get(ivec1H) || !(ivec1=ivec1H.get_rep())) return;

     ColumnMatrixHandle ivec2H;
     ColumnMatrix *ivec2;
     if (!ivec2P->get(ivec2H) || !(ivec2=ivec2H.get_rep())) return;
     
     if (ivec1->nrows() != ivec2->nrows()) {
         cerr << "Error - can't compute error on vectors of different lengths!\n";
         cerr << "vec1 length="<<ivec1->nrows();
         cerr << "vec2 length="<<ivec2->nrows();
         return;
     }

     int ne=ivec2->nrows();

     ColumnMatrix* error=scinew ColumnMatrix(1);
     double *val=error->get_rhs();
     ColumnMatrixHandle errorH(error);

     // compute CC
     
     double CCnum=0;
     double CCdenom1=0;
     double CCdenom2=0;
     double RMS=0;
     int i;
     double pp;
     pTCL.get().get_double(pp);
     for (i=0; i<ne; i++) {
         CCnum+=(*ivec1)[i]*(*ivec2)[i];
         CCdenom1+=(*ivec1)[i]*(*ivec1)[i];
         CCdenom2+=(*ivec2)[i]*(*ivec2)[i];
         double tmp=fabs((*ivec1)[i]-(*ivec2)[i]);
	 if (pp==1) RMS+=tmp; 
	 else if (pp==2) RMS+=tmp*tmp; 
	 else RMS+=pow(tmp,pp);
     }
     RMS = pow(RMS,1/pp);
     double CCdenom=Sqrt(CCdenom1*CCdenom2);
     double CC=Min(CCnum/CCdenom, 1000000.);
     double CCinv=Min(1.0/(Abs(CCnum)/CCdenom), 1000000.);
     double RMSrel=Min(RMS/CCdenom1, 1000000.);

     ostringstream str;
     str << id << " append_graph " << 1-CC << " " << RMSrel << " \"";
     for (i=0; i<ne; i++)
         str << i << " " << (*ivec1)[i] << " ";
     str << "\" \"";
     for (i=0; i<ne; i++)
         str << i << " " << (*ivec2)[i] << " ";
     str << "\" ; update idletasks";
//     cerr << "str="<<str.str()<<"\n";
     TCL::execute(str.str().c_str());

     if (methodTCL.get() == "CC") {
         *val=CC;
     } else if (methodTCL.get() == "CCinv") {
         *val=CCinv;
     } else if (methodTCL.get() == "RMS") {
         *val=RMS;
     } else if (methodTCL.get() == "RMSrel") {
         *val=RMSrel;
     } else {
         cerr << "Unknown ErrorMetric::methodTCL - "<<methodTCL.get()<<"\n";
         *val=0;
     }
//     cerr << "Error="<<*val<<"\n";
     errorP->send(errorH);
}
} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.7  2000/03/17 09:25:43  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  2000/02/02 21:54:00  dmw
// Makefile, index - added new modules and removed no-longer-used
// libraries
// Radiosity - fixed 64-bit include guards
// EEG/Makefile.in - removed InvEEGSolve from Makefile
// Taubin - constrained relaxation
// ErrorMetrix - no idea
// all others are just new modules
//
// Revision 1.5  1999/12/11 05:43:20  dmw
// need to take sqrt to get RMS error
//
// Revision 1.4  1999/10/07 02:06:34  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/09/22 18:43:25  dmw
// added new GUI
//
// Revision 1.2  1999/09/08 02:26:27  sparker
// Various #include cleanups
//
// Revision 1.1  1999/09/02 04:49:24  dmw
// more of Dave's modules
//
//
