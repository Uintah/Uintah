/**************************************************************
 * dbuildmat.cc:  Build a Double Precision
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

class dbuildmat : public Module {

  DenseMatrixOPort* oport;

  int stop_flag;

public:
  
  dbuildmat(const clString& id);
  dbuildmat(const dbuildmat&, int deep);
  virtual ~dbuildmat();
  virtual Module* clone(int deep); 
  virtual void execute();
  virtual void tcl_command(TCLArgs& args, void* userdata);
  
}; //class

extern "C" {
  Module* make_dbuildmat(const clString& id)
  {
    return new dbuildmat(id);
  }
  
}


//---------------------------------------------------------------

dbuildmat::dbuildmat(const clString& id)
  : Module("dbuildmat", id, Filter)

{
  
// Create an output port
  oport=new DenseMatrixOPort(this, "DenseMatrix", DenseMatrixIPort::Atomic);
  add_oport(oport);

}

//----------------------------------------------------------

dbuildmat::dbuildmat(const dbuildmat& copy, int deep)
  : Module(copy, deep)
{}

//------------------------------------------------------------

dbuildmat::~dbuildmat(){}

//-------------------------------------------------------------

Module* dbuildmat::clone(int deep)
{
  return new dbuildmat(*this, deep);
}


//--------------------------------------------------------------

#include <time.h>

void dbuildmat::execute()
{
  int nrows = 5;
  int ncols = 5;

  cerr << "***Executing the matbuild module\n";

  DenseMatrix * A= scinew DenseMatrix(nrows, ncols);
  
  srand(time(0));

  for(int i=0; i<nrows; i++){
    for(int j=0; j<ncols; j++){
      A->put(i, j, double(rand()%10) );
    }
  }

  cerr << "Matrix (in build module):\n";
  double *ptr = A->getData();
  for (i=0; i<nrows*ncols; i++){
    cerr << ptr[i] <<"  ";
    if(i%ncols == ncols-1)
      cerr << endl;
  }
  cerr << endl;

  DenseMatrixHandle handle(A);
  oport->send(DenseMatrixHandle(handle));
}




void dbuildmat::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2){
	args.error("dbuildmat needs a minor command");
	return;
    }
    if(args[1] == "stop"){
      stop_flag=1;
    } else {
      	Module::tcl_command(args, userdata);
    }
}
