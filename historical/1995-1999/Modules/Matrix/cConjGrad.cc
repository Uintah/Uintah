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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <TCL/TCLvar.h>
#include <iostream.h>
#include <Datatypes/cSMatrix.h>
#include <Datatypes/cDMatrix.h>
#include <Datatypes/cMatrixPort.h>
#include <Datatypes/cVector.h>
#include <Datatypes/cVectorPort.h>
#include <Math/Complex.h>

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
  cConjGrad(const cConjGrad&, int deep);
  virtual ~cConjGrad();
  virtual Module* clone(int deep); 
  virtual void execute();
  virtual void tcl_command(TCLArgs& args, void* userdata);
  
}; //class



extern "C" {
  Module* make_cConjGrad(const clString& id)
  {
    return new cConjGrad(id);
  }
  
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

cConjGrad::cConjGrad(const cConjGrad& copy, int deep)
  : Module(copy, deep),
    tcl_max_it("tcl_max_it",id,this),
    tcl_it("tcl_it",id,this),
    tcl_max_err("tcl_max_err",id,this),
    tcl_err("tcl_err",id,this),
    tcl_precond("tcl_precond",id,this),
    tcl_status("tcl_status",id,this)


{}

//------------------------------------------------------------

cConjGrad::~cConjGrad(){}

//-------------------------------------------------------------

Module* cConjGrad::clone(int deep)
{
  return new cConjGrad(*this, deep);
}


//--------------------------------------------------------------

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
Complex alpha,beta,ro1,ro2;
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
