//static char *id="@(#) $Id$";

/****************************************************************
 *  cConjGrad.cc   Complex Conjugate Gradient                   *
 *                                                              *
 *  Written by:                                                 *
 *   Leonid Zhukov                                              *
 *   Department of Computer Science                             *
 *   University of Utah                                         *
 *   August 1997                                                *
 *                                                              *
 *  Copyright (C) 1997 SCI Group                                *
 *                                                              *
 *                                                              *
 ****************************************************************/

#include <PSECore/Dataflow/Module.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Datatypes/cSMatrix.h>
#include <SCICore/Datatypes/cDMatrix.h>
#include <PSECore/Datatypes/cMatrixPort.h>
#include <SCICore/Datatypes/cVector.h>
#include <PSECore/Datatypes/cVectorPort.h>
#include <iostream>
using std::cerr;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class cConjGrad : public Module {

  cMatrixIPort* iportA;
  cVectorIPort* iportB; 
  cVectorOPort* oport;

  int stop_flag;

public:
  
  TCLint tcl_max_it;
  TCLint tcl_it;
  TCLdouble tcl_max_err;
  TCLdouble tcl_err;
  TCLint tcl_precond;
  TCLstring tcl_status;

  cConjGrad(const clString& id);
  virtual ~cConjGrad();
  virtual void execute();
  virtual void tcl_command(TCLArgs& args, void* userdata);
  
}; //class



extern "C" Module* make_cConjGrad(const clString& id) {
  return new cConjGrad(id);
}


//---------------------------------------------------------------

cConjGrad::cConjGrad(const clString& id)
  : Module("cConjGrad", id, Filter),
    tcl_max_it("tcl_max_it",id,this),
    tcl_it("tcl_it",id,this),
    tcl_max_err("tcl_max_err",id,this),
    tcl_err("tcl_err",id,this),
    tcl_precond("tcl_precond",id,this),
    tcl_status("tcl_status",id,this)

{
  
// Create an input portA
  iportA=new cMatrixIPort(this, "cMatrix", cMatrixIPort::Atomic);
  add_iport(iportA);
 
    
// Create an intput port
  iportB=new cVectorIPort(this, "cVector", cVectorIPort::Atomic);
  add_iport(iportB);

// Create an output port
  oport=new cVectorOPort(this, "cVector", cVectorIPort::Atomic);
  add_oport(oport);

  
}

//----------------------------------------------------------

cConjGrad::~cConjGrad(){}

//-------------------------------------------------------------

void cConjGrad::execute()
{
 tcl_status.set("Running");

  cMatrixHandle handleA;
  
  if(!iportA->get(handleA))
    return;
  
  cMatrix* A= handleA.get_rep();
  
  if(!A){
    cerr << "Not a cmatrix\n";
    return;
  }

  cVectorHandle handleB;
  
  if(!iportB->get(handleB))
    return;
  
  cVector* b = handleB.get_rep();
  
  if(!b){
    cerr << "Not a cVector\n";
    return;
  } 

 
//Conjugate Gradient:
//----------------------------------------------------------------------------------- 

 
// Here is a test input data, remove and uncomment 'Matrix handle' staff 
#if 0
//***************************************
cDMatrix *A = new cDMatrix(100);
cVector *b = new cVector(100);
A->load("matrix.dat");
b->load("vector.dat");
//******************************
#endif

//initial guess - all 0;
cVector *x = new cVector(b->size());

cVector r((*b) - (*A)*(*x));
cVector p(r);
cVector q(p);
cVector p1(p),q1(q),tmp(q);
cVector::Complex alpha,beta,ro1,ro2;
int flag = -1;
double err;
cVector z(r);
 

 
int Max_it = tcl_max_it.get();
double max_err = tcl_max_err.get();
int precond = tcl_precond.get();


 stop_flag=0;
 for(int i=1;i<Max_it;i++){

   
   
   if (precond == 1){   //Diag Precond
     for(int j=0;j<b->size();j++)
       z(j) = r(j)/A->get(j,j);
     ro1 = r*z;
     if (i==1) p.set(z);
   }
   else
      ro1 = r*r;

   if(i!=1){	
     beta = ro1/ro2;
     p.mult(beta);
     p.add(r);
   }
   
 q.set(p); 
 A->mult(q,tmp);
 alpha = ro1/(p*q);
 p1.set(p);
 p1.mult(alpha); 
 x->add(p1);
 q1.set(q);
 q1.mult(alpha); 
 r.subtr(q1);
 ro2 = ro1;
 err = r.norm(); 
 tcl_err.set(err);
    if (err < max_err){
    flag = 0;
    break;
    }
   if(stop_flag){
     flag=2;
     break;
   }
     

 TCL::execute("update idletasks");
   tcl_it.set(i);
   
}
  
if (flag == 0)
 tcl_status.set("Done!"); 
  else if(flag==2)
    tcl_status.set("Interrupted!");
 else
   tcl_status.set("Failed to Converge!");

  TCL::execute("update idletasks");
  
 // oport->send(cVectorHandle(C));

}
//---------------------------------------------------------------




void cConjGrad::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2){
	args.error("cConjGrad needs a minor command");
	return;
    }
    if(args[1] == "stop"){
      stop_flag=1;
    } else {
      	Module::tcl_command(args, userdata);
    }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.7  2000/03/17 09:27:08  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/10/07 02:06:53  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:47:52  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:49  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:46  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:32  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:46  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:51  dav
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
