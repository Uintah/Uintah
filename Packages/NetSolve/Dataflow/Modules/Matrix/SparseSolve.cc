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
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/GuiInterface/GuiVar.h>
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
  SparseSolve(const clString& id);
  virtual ~SparseSolve();

  virtual void execute();
  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" NetSolveSHARE Module* make_SparseSolve(const clString& id) {
  return new SparseSolve(id);
}

SparseSolve::SparseSolve(const clString& id)
  : Module("SparseSolve", id, Source), 
  target_error("target_error", id, this), 
  final_error("final_error", id, this),
  final_iterations("final_iterations", id, this), 
  maxiter("maxiter", id, this),
  method("method", id, this)
{
  matrixport = scinew MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
  add_iport(matrixport);
  rhsport = scinew MatrixIPort(this, "RHS", MatrixIPort::Atomic);
  add_iport(rhsport);

  solport = scinew MatrixOPort(this, "Solution", MatrixIPort::Atomic);
  add_oport(solport);
}

SparseSolve::~SparseSolve(){
}

void SparseSolve::execute(){
#ifdef __sgi
  MatrixHandle matrix;
  if(!matrixport->get(matrix))
    return;
  
  SparseRowMatrix* srm = matrix->getSparseRow(); 
  
  if (!srm) {
    msgStream_ << "The supplied matrix is not of type SparseRow" << endl;
    return;
  }
  
  MatrixHandle rhs;
  if(!rhsport->get(rhs))
    return;

  
  if (!matrix.get_rep() || !rhs.get_rep() || !rhs->getColumn()) {
    msgStream_ << "Netsolve_MatrixSolver::execute() input problems\n";
    return;
  }
  
  int iterations;
  int maxit = maxiter.get();
  double tolerance = target_error.get();
  
  solution=scinew ColumnMatrix(rhs->nrows());
  solution->zero();
  ColumnMatrix* pRhs = rhs->getColumn();
  
  netslmajor("Row");
   
  double* lhs = NULL; 
 
  msgStream_ << "Calling NetSolve for 'petsc', blocking :" << endl;
 
  int status = netsl ("iterative_solve_parallel()",
		      "PETSC",
		      srm->nrows(),    	//matrix->nnrows,
		      srm->get_nnz(),
		      srm->get_val(),    	//matrix->a
		      srm->get_col(),	//matrix->columns,
		      srm->get_row(),	//matrix->rows,
		      pRhs->get_rhs(),
		      &tolerance,
		      &maxit,
		      lhs,
		      &iterations);
  
 
  if (status < 0) {
    netslerr(status);
    delete solution;
    return;

  }
  else {
    msgStream_ << "NetSolve call succeeded.  Passing solution through port" << endl;
    solution->put_lhs(lhs);
    msgStream_ << "Tolerance: " << tolerance << ", iterations = " << iterations << endl;
    //for (int ii=0; ii<solution->nrows(); ii++){
    //  msgStream_ << (*solution)[ii] << "\t";
    //  if (ii%5==0)
    //	msgStream_ << endl;
    //}
    
    solport->send(MatrixHandle(solution));
  }
#endif // __sgi
}

void SparseSolve::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Netsolve


