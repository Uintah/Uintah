/****************************************************************
 *  Simple "OptDip module"for the SCIRun                      *
 *                                                              *
 *  Written by:                                                 *
 *   Leonid Zhukov                                              *
 *   Department of Computer Science                             *
 *   University of Utah                                         *
 *   November 1997                                              *
 *                                                              *
 *  Copyright (C) 1997 SCI Group                                *
 *                                                              *
 *                                                              *
 ****************************************************************/

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;
using std::endl;

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;

double det_3x3(double* p1, double* p2, double* p3){
    double D = p1[0]*p2[1]*p3[2]-p1[0]*p2[2]*p3[1]-p1[1]*p2[0]*p3[2]+p2[0]*p1[2]*p3[1]+p3[0]*p1[1]*p2[2]-p1[2]*p2[1]*p3[0];
    return(D);
} 

int solve_3x3(double a[3][3] ,double b[3], double x[3]){
    double D = det_3x3(a[0],a[1],a[2]);
    double D1 = det_3x3(b,a[1],a[2]);
    double D2 = det_3x3(a[0],b,a[2]);
    double D3 = det_3x3(a[0],a[1],b);


    if ( D!=0){ 
	x[0] = D1/D;
	x[1] = D2/D; 
	x[2] = D3/D; 
    } else {
	cerr << "ERROR, DET = 0!"<< endl;
	x[0] = 1;
	x[1] = 0; 
	x[2] = 0;
	return(-1);
    }
    return(0);
}

double error_norm( double* x1, double* x2,int  n){
    double err= 0;
    for (int i=0;i<n;i++)
	err = err + (x1[i]-x2[i])*(x1[i]-x2[i]);
    return(sqrt(err));
}

class OptDip : public Module {

 ColumnMatrixIPort* v0_port;
 ColumnMatrixIPort* v1_port;
 ColumnMatrixIPort* v2_port;  
 ColumnMatrixIPort* v_port;

 ColumnMatrixOPort* cc_port;  
 ColumnMatrixOPort* w_port;

  
public:
 
  TCLstring tcl_status;
  OptDip(const clString& id);
  virtual ~OptDip();
  virtual void execute();
  
}; //class


extern "C" Module* make_OptDip(const clString& id) {
    return new OptDip(id);
}


//---------------------------------------------------------------
OptDip::OptDip(const clString& id)
  : Module("OptDip", id, Filter),
    tcl_status("tcl_status",id,this)

{

  v_port = new ColumnMatrixIPort(this,"Column Matrix v",ColumnMatrixIPort::Atomic);
  add_iport(v_port);

 v0_port = new ColumnMatrixIPort(this,"Column Matrix v1",ColumnMatrixIPort::Atomic);
  add_iport(v0_port);

  v1_port = new ColumnMatrixIPort(this,"Column Matrix v2",ColumnMatrixIPort::Atomic);
  add_iport(v1_port);

  v2_port = new ColumnMatrixIPort(this,"Column Matrix v3",ColumnMatrixIPort::Atomic);
  add_iport(v2_port);

  cc_port = new ColumnMatrixOPort(this,"Column Matrix cc",ColumnMatrixIPort::Atomic);
  add_oport(cc_port);
  
  w_port = new ColumnMatrixOPort(this,"Column Matrix w",ColumnMatrixIPort::Atomic);
  add_oport(w_port);
  
 
}

//------------------------------------------------------------
OptDip::~OptDip(){}

//--------------------------------------------------------------

void OptDip::execute()
{
  tcl_status.set("Calling OptDip!");
  
  ColumnMatrixHandle v0;
  if(!v0_port->get(v0))
    return; 

  ColumnMatrixHandle v1;
  if(!v1_port->get(v1))
    return; 

  ColumnMatrixHandle v2;
  if(!v2_port->get(v2))
    return;
  
  ColumnMatrixHandle v;
  if(!v_port->get(v))
    return; 
  
   
  int n = v->nrows();
  if (v0->nrows() != n || v1->nrows() != n || v2->nrows() != n) {
      cerr << "Error - input vectors must all be the same length!\n";
      return;
  }

   double a[3][3];
   double a_b[3];

   double* cc = new double[3];
   double* w = new double[n]; 

#if 0   
   cerr << "OptDip -- these are the input vectors...\n";
   cerr << "  v0=";
       for (int ii=0; ii<v->nrows(); ii++) cerr << v0->get_rhs()[ii] << ",";
   cerr << "\n  v1=";
       for (ii=0; ii<v->nrows(); ii++) cerr << v1->get_rhs()[ii] << ",";
   cerr << "\n  v2=";
       for (ii=0; ii<v->nrows(); ii++) cerr << v2->get_rhs()[ii] << ",";
   cerr << "\n";
#endif
   
   for(int i=0;i<3;i++){
     for(int j=0;j<3;j++)  
       a[i][j] =0;
     a_b[i]=0;
   }
   
   
   int s;
   for(s=0;s<n;s++){
     a[0][0] = a[0][0] + (v0->get_rhs())[s]*(v0->get_rhs())[s];
     a[0][1] = a[0][1] + (v0->get_rhs())[s]*(v1->get_rhs())[s];  
     a[0][2] = a[0][2] + (v0->get_rhs())[s]*(v2->get_rhs())[s];
     a[1][1] = a[1][1] + (v1->get_rhs())[s]*(v1->get_rhs())[s];
     a[1][2] = a[1][2] + (v1->get_rhs())[s]*(v2->get_rhs())[s];
     a[2][2] = a[2][2] + (v2->get_rhs())[s]*(v2->get_rhs())[s];
   }
   
   a[1][0] = a[0][1];
   a[2][0] = a[0][2];
   a[2][1] = a[1][2];
   
   for(s=0;s<n;s++){
     a_b[0] = a_b[0] + (v0->get_rhs())[s]*(v->get_rhs())[s];
     a_b[1] = a_b[1] + (v1->get_rhs())[s]*(v->get_rhs())[s];
     a_b[2] = a_b[2] + (v2->get_rhs())[s]*(v->get_rhs())[s]; 
   }
   
   
   solve_3x3(a,a_b,cc);
   
   
   for(s=0;s<n;s++)
     w[s] = cc[0]*(v0->get_rhs())[s] + cc[1]*(v1->get_rhs())[s] + cc[2]*(v2->get_rhs())[s];
   
   
   ColumnMatrix* cc_vector = new ColumnMatrix(4);
   cc_vector->put_lhs(cc);   
   
   ColumnMatrix* w_vector = new ColumnMatrix(n);
   w_vector->put_lhs(w);
   
   
   cc_port->send(cc_vector);
   w_port->send(w_vector);

#if 0
   cerr << "OptDip -- output direction: ("<<cc[0]<<","<<cc[1]<<","<<cc[2]<<")\n   w=";
   for (int ii=0; ii<n; ii++) cerr << w[ii]<<" ";
   cerr << "\n";
#endif
} 
//---------------------------------------------------------------
} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.5  2000/03/17 09:25:47  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.4  1999/10/07 02:06:37  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/09/22 18:43:26  dmw
// added new GUI
//
// Revision 1.2  1999/09/08 02:26:28  sparker
// Various #include cleanups
//
// Revision 1.1  1999/09/02 04:50:04  dmw
// more of Dave's modules
//
//
