/**************************************************************
 * dmatmul.cc:  Double Precision Matrix Multiply from BLAS via NetSolve
 *
 * Written by:
 *  Dorian Arnold
 *  Department of Computer Science
 *  University of Tennessee
 *  April 1999
 *
 ************************************************************/

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <iostream.h>
#include <Datatypes/DenseMatrix.h>
#include <Datatypes/DenseMatrixPort.h>
#include <Malloc/Allocator.h>
//#include "/hide/homes/darnold/NetSolve_Workdir/NetSolve_bak/include/netsolve.h"
//#include "netsolve.h"

extern "C" void netslerr(int);
extern "C" int netsl(char*,...);

class dmatmul : public Module {

  DenseMatrixIPort* iportA;
  DenseMatrixIPort* iportB; 

  int stop_flag;

public:
  
  dmatmul(const clString& id);
  dmatmul(const dmatmul&, int deep);
  virtual ~dmatmul();
  virtual Module* clone(int deep); 
  virtual void execute();
  virtual void tcl_command(TCLArgs& args, void* userdata);
  
}; //class

extern "C" {
  Module* make_dmatmul(const clString& id)
  {
    return new dmatmul(id);
  }
  
}


//---------------------------------------------------------------

dmatmul::dmatmul(const clString& id)
  : Module("dmatmul", id, Filter)

{
  
// Create an input port
  iportA=new DenseMatrixIPort(this, "Matrix", DenseMatrixIPort::Atomic);
  add_iport(iportA);
 
    
// Create an intput port
  iportB=new DenseMatrixIPort(this, "Matrix", DenseMatrixIPort::Atomic);
  add_iport(iportB);

}

//----------------------------------------------------------

dmatmul::dmatmul(const dmatmul& copy, int deep)
  : Module(copy, deep)
{}

//------------------------------------------------------------

dmatmul::~dmatmul(){}

//-------------------------------------------------------------

Module* dmatmul::clone(int deep)
{
  return new dmatmul(*this, deep);
}


//--------------------------------------------------------------

void dmatmul::execute()
{

  DenseMatrixHandle handleA;
  DenseMatrixHandle handleB;
  
  cerr << "***Executing the matmul module\n";

  if( (!iportA->get(handleA)) || (!iportB->get(handleB)) )
    return;
  
  DenseMatrix* A= handleA.get_rep();
  DenseMatrix* B = handleB.get_rep();
  
  if(!A || !B){
    cerr << "dmatmul: error in inputs \n";
    return;
  } 

  int Anrows, Bnrows,
      Ancols, Bncols;

  Anrows = A->nrows();
  Ancols = A->ncols();
  double * AData = A->getData();
  Bncols = B->ncols();
  Bnrows = B->nrows();
  double * BData = B->getData();
  double * CData = scinew double [Anrows*Bncols];

  if(Ancols != Bnrows){
    cerr << "Error: matrix size mismatch\n";
    return;
  }

  cerr << "Matrix A:\n";
  for (int j=0; j<Anrows*Ancols; j++){
    cerr << AData[j] <<"  ";
    if(j%Ancols == Ancols-1)
      cerr << endl;
  }

  cerr << endl;
  cerr << "Matrix B:\n";
  for (j=0; j<Bnrows*Bncols; j++){
    cerr << BData[j] <<"  ";
    if(j%Bncols == Bncols-1)
      cerr << endl;
  }

  //Matrix Multiply (call to NetSolve):
   int status = netsl("dmatmul()", Anrows,
                               Bncols,
                               Ancols,
                               AData,
                               Anrows,
                               BData,
                               Bnrows,
                               Anrows,
                               CData);
                               
  if(status < 0){
    netslerr(status);
    return;
  }

  cerr << "Matrix C (result):\n";
  for (j=0; j<Anrows*Bncols; j++){
    cerr << CData[j] <<"  ";
    if(j%Bncols == Bncols-1)
      cerr << endl;
  } 

  cerr <<endl <<  "CALL TO NETSOLVE GOES HERE" << endl;
}

void dmatmul::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2){
	args.error("dmatmul needs a minor command");
	return;
    }
    if(args[1] == "stop"){
      stop_flag=1;
    } else {
      	Module::tcl_command(args, userdata);
    }
}
