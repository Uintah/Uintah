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
#include <Core/Containers/StringUtil.h>
#include <iostream>
#include <sstream>
using std::ostringstream;

namespace SCIRun {

class ErrorMetric : public Module {

public:

  ////////////////////////////////////////////////////////////
  // Constructor taking
  //  [in] string
  ErrorMetric(GuiContext* ctx);

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
  GuiInt       haveUI_;
  GuiString    methodTCL_;
  GuiString    pTCL_;
}; 


DECLARE_MAKER(ErrorMetric)
ErrorMetric::ErrorMetric(GuiContext* ctx)
  : Module("ErrorMetric", ctx, Filter, "Math", "SCIRun"),
    haveUI_(ctx->subVar("haveUI")),
    methodTCL_(ctx->subVar("methodTCL")),
    pTCL_(ctx->subVar("pTCL"))
{
}

ErrorMetric::~ErrorMetric()
{
}

void ErrorMetric::execute()
{
     ivec1P_ = (MatrixIPort *)get_iport("Vec1");
     ivec2P_ = (MatrixIPort *)get_iport("Vec2");
     errorP_ = (MatrixOPort *)get_oport("Error out");

     if (!ivec1P_) {
       error("Unable to initialize iport 'Vec1'.");
       return;
     }
     if (!ivec2P_) {
       error("Unable to initialize iport 'Vec2'.");
       return;
     }
     if (!errorP_) {
       error("Unable to initialize oport 'Error out'.");
       return;
     }
     MatrixHandle ivec1H;
     ColumnMatrix* ivec1;
     if (!ivec1P_->get(ivec1H)) return;
     ivec1=ivec1H->column();

     MatrixHandle ivec2H;
     ColumnMatrix *ivec2;
     if (!ivec2P_->get(ivec2H)) return;
     ivec2=ivec2H->column();
     
     if (ivec1->nrows() != ivec2->nrows()) {
         error("Can't compute error on vectors of different lengths!");
         error("vec1 length = " + to_string(ivec1->nrows()));
	 error("vec2 length = " + to_string(ivec2->nrows()));
         return;
     }

     int ne=ivec2->nrows();

     ColumnMatrix* errorM = scinew ColumnMatrix(1);
     double *val=errorM->get_data();
     MatrixHandle errorH(errorM);

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
	 gui->execute(str.str().c_str());
     }

     const string meth=methodTCL_.get();
     if (meth == "CC") {
         *val=cc;
     } else if (meth == "CCinv") {
         *val=ccInv;
     } else if (meth == "RMS") {
         *val=rms;
     } else if (meth == "RMSrel") {
         *val=rmsRel;
     } else {
         error("Unknown ErrorMetric::methodTCL_ - " + meth);
         *val=0;
     }
     errorP_->send(errorH);
}

} // End namespace SCIRun

