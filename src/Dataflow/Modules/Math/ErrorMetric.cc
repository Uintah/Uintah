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



/***************************************************************
CLASS
 ErrorMetric
 
   The ErrorMetric class is used to compute and visualize error 
   between two vectors

GENERAL INFORMATION
  ErrorMetic.cc - ErrorMetric class declaration and method defintions

  Author: David Weinstein (dmw@cs.utah.edu)

  Creation Date: June 1999
  
  Copyright (C) 1999 SCI Group

KEYWORDS: 
  Vector_Error
  
PATTERNS: 
  None

WARNINGS:
  None

POSSIBLE REVISIONS:
  None

  ************************************************************/



#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
using std::cerr;
#include <sstream>
using std::ostringstream;

namespace SCIRun {

class ErrorMetric : public Module {

public:

  ////////////////////////////////////////////////////////////
  // Constructor taking
  //  [in] clString
  ErrorMetric(const clString& id);

  // GROUP: Destructors:
  ///////////////////////////////////////////////////////////
  // Destructor
  virtual ~ErrorMetric();

  //////////////////////////////////////////////////////////
  // method to calculate and plot error between two vectors
  virtual void execute();

private:

  MatrixIPort* d_ivec1P;
  MatrixIPort* d_ivec2P;
  MatrixOPort* d_errorP;
  GuiInt d_haveUI;
  GuiString d_methodTCL;
  GuiString d_pTCL;
}; 


extern "C" Module* make_ErrorMetric(const clString& id)
{
    return scinew ErrorMetric(id);
}

ErrorMetric::ErrorMetric(const clString& id)
  : Module("ErrorMetric", id, Filter),
    d_haveUI("haveUI", id, this),
    d_methodTCL("methodTCL", id, this),
    d_pTCL("pTCL", id, this)
{
    // Create the input port
    d_ivec1P=scinew MatrixIPort(this, "Vec1",MatrixIPort::Atomic);
    add_iport(d_ivec1P);
    d_ivec2P=scinew MatrixIPort(this, "Vec2", MatrixIPort::Atomic);
    add_iport(d_ivec2P);

    // Create the output ports
    d_errorP=scinew MatrixOPort(this,"Error out",MatrixIPort::Atomic);
    add_oport(d_errorP);
}

ErrorMetric::~ErrorMetric()
{
}

void ErrorMetric::execute()
{
     MatrixHandle ivec1H;
     ColumnMatrix* ivec1;
     if (!d_ivec1P->get(ivec1H) || !(ivec1=dynamic_cast<ColumnMatrix*>(ivec1H.get_rep()))) return;

     MatrixHandle ivec2H;
     ColumnMatrix *ivec2;
     if (!d_ivec2P->get(ivec2H) || !(ivec2=dynamic_cast<ColumnMatrix*>(ivec2H.get_rep()))) return;
     
     if (ivec1->nrows() != ivec2->nrows()) {
         cerr << "Error - can't compute error on vectors of different lengths!\n";
         cerr << "vec1 length="<<ivec1->nrows();
         cerr << "vec2 length="<<ivec2->nrows();
         return;
     }

     int ne=ivec2->nrows();

     ColumnMatrix* error=scinew ColumnMatrix(1);
     double *val=error->get_rhs();
     MatrixHandle errorH(error);

     // compute CC
     
     double avg1=0, avg2=0;
     int iterate;
     for (iterate=0; iterate<ne; iterate++) {
	 avg1+=(*ivec1)[iterate];
	 avg2+=(*ivec2)[iterate];
     }
     avg1/=ne;
     avg2/=ne;

     double ccNum=0;
     double ccDenom1=0;
     double ccDenom2=0;
     double rms=0;
     double pp;
     d_pTCL.get().get_double(pp);
     for (iterate=0; iterate<ne; iterate++) {
	 double shift1=((*ivec1)[iterate]-avg1);
	 double shift2=((*ivec2)[iterate]-avg2);

         ccNum+=shift1*shift2;
         ccDenom1+=shift1*shift1;
         ccDenom2+=shift2*shift2;
//         double tmp=fabs((*ivec1)[iterate]-(*ivec2)[iterate]);
         double tmp=fabs(shift1-shift2);
	 if (pp==1) rms+=tmp; 
	 else if (pp==2) rms+=tmp*tmp; 
	 else rms+=pow(tmp,pp);
     }
     rms = pow(rms/ne,1/pp);
     double ccDenom=Sqrt(ccDenom1*ccDenom2);
     double cc=Min(ccNum/ccDenom, 1000000.);
     double ccInv=Min(1.0-ccNum/ccDenom, 1000000.);
     double rmsRel=Min(rms/ccDenom1, 1000000.);


     if (d_haveUI.get()) {
	 ostringstream str;
	 str << id << " append_graph " << ccInv << " " << rmsRel << " \"";
	 for (iterate=0; iterate<ne; iterate++)
	     str << iterate << " " << (*ivec1)[iterate] << " ";
	 str << "\" \"";
	 for (iterate=0; iterate<ne; iterate++)
	     str << iterate << " " << (*ivec2)[iterate] << " ";
	 str << "\" ; update idletasks";
	 //     cerr << "str="<<str.str()<<"\n";
	 TCL::execute(str.str().c_str());
     }

     clString meth=d_methodTCL.get();
     if (meth == "CC") {
         *val=cc;
     } else if (meth == "CCinv") {
         *val=ccInv;
     } else if (meth == "RMS") {
         *val=rms;
     } else if (meth == "RMSrel") {
         *val=rmsRel;
     } else {
         cerr << "Unknown ErrorMetric::d_methodTCL - "<<meth<<"\n";
         *val=0;
     }
//     cerr << "Error="<<*val<<"\n";
     d_errorP->send(errorH);
} // End namespace SCIRun
}


