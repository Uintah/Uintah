/*
 *  SparseSolve.cc:
 *
 *  Written by:
 *   Michelle Miller
 *   Innovative Computing Lab
 *   University of Tennessee
 *   Aug. 1, 2000
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <NetSolve/share/share.h>

#include <stdio.h>
#include <netsolve/netsolveclient.h>

namespace Netsolve {

using namespace SCIRun;

extern "C" {
    int netsl(char* ...);
    int netslmajor(char *);
    void netslerr(int);
}

class NetSolveSHARE SparseSolve : public Module {

  //! Ports
  MatrixIPort*  matrixport;
  MatrixIPort*  rhsport;
  MatrixOPort*  solport;
  ColumnMatrix* solution;
  
  //! GUI variables
  GuiDouble  target_error;
  GuiDouble  final_error;
  GuiInt     final_iterations;
  GuiInt     maxiter;
  GuiString  method;
  
public:
  //! Constructor/Destructor
  SparseSolve(const string& get_id());
  virtual ~SparseSolve();

  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" NetSolveSHARE Module* make_SparseSolve(const string& get_id()) {
  return new SparseSolve(get_id());
}

SparseSolve::SparseSolve(const string& get_id())
  : Module("SparseSolve", get_id(), Source, "Matrix", "NetSolve"), 
  target_error("target_error", get_id(), this), 
  final_error("final_error", get_id(), this),
  final_iterations("final_iterations", get_id(), this), 
  maxiter("maxiter", get_id(), this),
  method("method", get_id(), this)
{
}

SparseSolve::~SparseSolve(){
}

void SparseSolve::execute(){

  MatrixHandle matrix;
  matrixport = (MatrixIPort *)get_iport("Matrix");
  rhsport = (MatrixIPort *)get_iport("RHS");
  solport = (MatrixOPort *)get_oport("Solution");
  if(!matrixport->get(matrix)) {
    msg_stream_ << "The Matrix input is required, but nothing is connected."
               << endl;    
    return;
  }
  
  SparseRowMatrix* srm = matrix->sparse(); 
  
  if (!srm) {
    msg_stream_ << "The supplied matrix is not of type SparseRow" << endl;
    return;
  }
  
  MatrixHandle rhs;
  if(!rhsport->get(rhs)) {
    msg_stream_ << "The RHS input is required, but nothing is connected."
               << endl;
    return;
  }

  
//  if (!matrix.get_rep() || !rhs.get_rep() || !rhs->getColumn()) {
  if (!matrix.get_rep() || !rhs.get_rep() || !rhs->column()) {
    msg_stream_ << "One or more of the inputs are NULL";
    return;
  }
  
  int iterations;
  int maxit = maxiter.get();
  double tolerance = target_error.get();
  
  solution=scinew ColumnMatrix(rhs->nrows());
  solution->zero();
  ColumnMatrix* pRhs = rhs->column();
  
  netslmajor("Row");
   
  double* lhs = NULL; 
 
  msg_stream_ << "Calling NetSolve for 'petsc', blocking :" << endl;
 
  int status = netsl ("iterative_solve_parallel()",
		      "PETSC",
		      srm->nrows(),    	
		      srm->get_nnz(),
		      srm->get_val(),    
		      srm->get_col(),	
		      srm->get_row(),	
		      pRhs->get_data(),
		      &tolerance,
		      &maxit,
		      lhs,
		      &iterations);
  
  if (status < 0) {
    netslerr(status);
    delete solution;
    return;
  } else {
    msg_stream_ << "NetSolve call succeeded.  Passing solution through port" 
	       << endl;
    solution->set_data(lhs);
    msg_stream_ << "Tolerance: " << tolerance << ", iterations = " 
	       << iterations << endl;
    
    solport->send(MatrixHandle(solution));
  }
}

void SparseSolve::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Netsolve


