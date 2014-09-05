/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  ReportColumnMatrixMisfit.cc: Compute and visualize error between two vectors
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
 ReportColumnMatrixMisfit
 
   The ReportColumnMatrixMisfit class is used to compute and visualize error 
   between two vectors

GENERAL INFORMATION
  ErrorMetic.cc - ReportColumnMatrixMisfit class declaration and method defintions

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
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>
#include <sstream>
using std::ostringstream;

namespace SCIRun {

class ReportColumnMatrixMisfit : public Module {

public:

  ////////////////////////////////////////////////////////////
  // Constructor taking
  //  [in] string
  ReportColumnMatrixMisfit(GuiContext* ctx);

  // GROUP: Destructors:
  ///////////////////////////////////////////////////////////
  // Destructor
  virtual ~ReportColumnMatrixMisfit();

  //////////////////////////////////////////////////////////
  // method to calculate and plot error between two vectors
  virtual void execute();

private:

  GuiInt       have_ui_;
  GuiString    methodTCL_;
  GuiString    pTCL_;
}; 


DECLARE_MAKER(ReportColumnMatrixMisfit)
ReportColumnMatrixMisfit::ReportColumnMatrixMisfit(GuiContext* ctx)
  : Module("ReportColumnMatrixMisfit", ctx, Filter, "Math", "SCIRun"),
    have_ui_(get_ctx()->subVar("have_ui")),
    methodTCL_(get_ctx()->subVar("methodTCL")),
    pTCL_(get_ctx()->subVar("pTCL"))
{
}


ReportColumnMatrixMisfit::~ReportColumnMatrixMisfit()
{
}


void
ReportColumnMatrixMisfit::execute()
{
     MatrixHandle ivec1H;
     if (!get_input_handle("Vec1", ivec1H)) return;
     ColumnMatrix* ivec1;
     ivec1 = ivec1H->column();
     ivec1H = ivec1;  // prevent mem leakage

     MatrixHandle ivec2H;
     if (!get_input_handle("Vec2", ivec2H)) return;
     ColumnMatrix *ivec2;
     ivec2 = ivec2H->column();
     ivec2H = ivec2; // prevent mem leakage
     
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
     for (iterate=0; iterate<ne; iterate++) 
     {
       avg1+=(*ivec1)[iterate];
       avg2+=(*ivec2)[iterate];
     }
     avg1/=ne;
     avg2/=ne;

     double Norm1=0; 
     double ccNum=0;
     double ccDenom1=0;
     double ccDenom2=0;
     double rms=0;
     double pp;
     string_to_double(pTCL_.get(), pp);
     for (iterate=0; iterate<ne; iterate++) 
     {
       double shift1=((*ivec1)[iterate]-avg1);
    	 double shift2=((*ivec2)[iterate]-avg2);

         ccNum+=shift1*shift2;
         ccDenom1+=shift1*shift1;
         ccDenom2+=shift2*shift2;
         double tmp=fabs((*ivec1)[iterate]-(*ivec2)[iterate]);
	 if (pp==1) {
           rms+=tmp;
           Norm1+=fabs((*ivec1)[iterate]);
          }        
	 else if (pp==2) {
           rms+=tmp*tmp;
           Norm1+=((*ivec1)[iterate])*((*ivec1)[iterate]);
          }         
	 else {
           rms+=pow(tmp,pp);
           Norm1+=pow(fabs((*ivec1)[iterate]),pp);
         }
     }
     rms = pow(rms/ne,1/pp);
     double ccDenom=Sqrt(ccDenom1*ccDenom2);
     double cc=Min(ccNum/ccDenom, 1000000.);
     double ccInv=Min(1.0-ccNum/ccDenom, 1000000.);
     double rmsRel=Min(rms*pow(ne/Norm1,1/pp), 1000000.);


     if (have_ui_.get()) {
	 ostringstream str;
	 str << get_id() << " append_graph " << MakeReal(ccInv) << " " 
	     << MakeReal(rmsRel) << " \"";
	 for (iterate=0; iterate<ne; iterate++)
	     str << iterate << " " << MakeReal((*ivec1)[iterate]) << " ";
	 str << "\" \"";
	 for (iterate=0; iterate<ne; iterate++)
	     str << iterate << " " << MakeReal((*ivec2)[iterate]) << " ";
	 str << "\" ; update idletasks";
	 get_gui()->execute(str.str().c_str());
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
         error("Unknown ReportColumnMatrixMisfit::methodTCL_ - " + meth);
         *val=0;
     }

     send_output_handle("Error Out", errorH);
}


} // End namespace SCIRun

