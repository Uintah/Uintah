/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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
  //  [in] string
  ErrorMetric(const string& id);

  // GROUP: Destructors:
  ///////////////////////////////////////////////////////////
  // Destructor
  virtual ~ErrorMetric();

  //////////////////////////////////////////////////////////
  // method to calculate and plot error between two vectors
  virtual void execute();

private:

  MatrixIPort* ivec1P_;
  MatrixIPort* ivec2P_;
  MatrixOPort* errorP_;
  GuiInt haveUI_;
  GuiString methodTCL_;
  GuiString pTCL_;
}; 


extern "C" Module* make_ErrorMetric(const string& id)
{
    return scinew ErrorMetric(id);
}

ErrorMetric::ErrorMetric(const string& id)
  : Module("ErrorMetric", id, Filter, "Math", "SCIRun"),
    haveUI_("haveUI", id, this),
    methodTCL_("methodTCL", id, this),
    pTCL_("pTCL", id, this)
{
    // Create the input port
    ivec1P_=scinew MatrixIPort(this, "Vec1",MatrixIPort::Atomic);
    add_iport(ivec1P_);
    ivec2P_=scinew MatrixIPort(this, "Vec2", MatrixIPort::Atomic);
    add_iport(ivec2P_);

    // Create the output ports
    errorP_=scinew MatrixOPort(this,"Error out",MatrixIPort::Atomic);
    add_oport(errorP_);
}

ErrorMetric::~ErrorMetric()
{
}

void ErrorMetric::execute()
{
     MatrixHandle ivec1H;
     ColumnMatrix* ivec1;
     if (!ivec1P_->get(ivec1H) || !(ivec1=dynamic_cast<ColumnMatrix*>(ivec1H.get_rep()))) return;

     MatrixHandle ivec2H;
     ColumnMatrix *ivec2;
     if (!ivec2P_->get(ivec2H) || !(ivec2=dynamic_cast<ColumnMatrix*>(ivec2H.get_rep()))) return;
     
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
     string_to_double(pTCL_.get(), pp);
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


     if (haveUI_.get()) {
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

     string meth=methodTCL_.get();
     if (meth == "CC") {
         *val=cc;
     } else if (meth == "CCinv") {
         *val=ccInv;
     } else if (meth == "RMS") {
         *val=rms;
     } else if (meth == "RMSrel") {
         *val=rmsRel;
     } else {
         cerr << "Unknown ErrorMetric::methodTCL_ - "<<meth<<"\n";
         *val=0;
     }
//     cerr << "Error="<<*val<<"\n";
     errorP_->send(errorH);
} // End namespace SCIRun
}


