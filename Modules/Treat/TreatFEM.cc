/*
 *  TreatFEM.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColumnMatrixPort.h>
#include <Datatypes/MatrixPort.h>
#include <Datatypes/Matrix.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldHUG.h>
#include <Datatypes/SparseRowMatrix.h>
#include <Geometry/Point.h>
#include <TCL/TCLvar.h>

class TreatFEM : public Module {
public:
    MatrixOPort* outmatrix;
    ColumnMatrixOPort* outrhs;
    ColumnMatrixIPort* loopsol;
    HexMeshOPort* outmesh;
    ScalarFieldOPort* outfield;
    ScalarFieldOPort* outpower;
    VectorFieldOPort* outvfield;
    TreatFEM(const clString& id);
    TreatFEM(const TreatFEM&, int deep);
    virtual ~TreatFEM();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_TreatFEM(const clString& id)
{
    return new TreatFEM(id);
}
};

TreatFEM::TreatFEM(const clString& id)
: Module("TreatFEM", id, Filter)
{
    loopsol=new ColumnMatrixIPort(this, "Solution - feedback", ColumnMatrixIPort::Atomic);
    add_iport(loopsol);
    // Create the output port
    outmatrix=new MatrixOPort(this, "Matrix", MatrixIPort::Atomic);
    add_oport(outmatrix);
    outrhs=new ColumnMatrixOPort(this, "RHS", ColumnMatrixIPort::Atomic);
    add_oport(outrhs);
    outmesh=new HexMeshOPort*this, "Mesh", HexMeshOPort::Atomic);
    add_oport(outmesh);
    outfield=new ScalarFieldOPort(this, "Temp", ScalarFieldIPort::Atomic);
    add_oport(outfield);
    outpower=new ScalarFieldOPort(this, "Power Field", ScalarFieldIPort::Atomic);
    add_oport(outpower);
    outvfield=new VectorFieldOPort(this, "Velocity", VectorFieldOPort::Atomic);
    add_oport(outfield);
}

TreatFEM::TreatFEM(const TreatFEM& copy, int deep)
: Module(copy, deep)
{
}

TreatFEM::~TreatFEM()
{
}

Module* TreatFEM::clone(int deep)
{
    return new TreatFEM(*this, deep);
}


#include<fstream.h>
#include"list.h"
#include "Array1.h"
#include"property.h"
#include"block.h"
#include"node.h"
#include"element.h"
#include"shape.h"
#include"readData.h"
#include"sparseMatrix.h"
#include <stdlib.h>
#include <stdio.h>


void TreatFEM::execute()
{
  Array1<Node*> nodeList;
  Array1<Element*> elementList;
  Property propertyList;
  ofstream fout;
  int i,j,solve,ss, maxit, num_time_steps, iteration_total, writeEvery;
  double eps,residual,current_residual;
  char str[20];
  
  // ----------------------------------------------------------
  // Begin Program
  // ----------------------------------------------------------

  // ----------------------------------------------------------
  // read data files
  // ----------------------------------------------------------

  cerr << "Reading Data Files " << endl;
  if (!readData(nodeList, elementList, propertyList)) {
    cout << "Error in reading data files. Exiting program." << endl;
    exit(1);
  } else {
    cout << "All data files read successfully." << endl;
  }

  cout << "Creating HexMesh data structure\n";
  HexMesh* hexmesh=new HexMesh;
  for(i=0;i<nodeList.size();i++){
      Node* n=nodeList[i];
      hexmesh->add_node(n->getNum(), n->getX(), n->getY(), n->getZ());
  }
  for(i=0;i<elementList.size();i++){
      Element* in=elementList[i];
      EightHexNodes e;
      e.index[0]=in->getNode(0)->getNum();
      e.index[1]=in->getNode(1)->getNum();
      e.index[2]=in->getNode(2)->getNum();
      e.index[3]=in->getNode(3)->getNum();
      e.index[4]=in->getNode(4)->getNum();
      e.index[5]=in->getNode(5)->getNum();
      e.index[6]=in->getNode(6)->getNum();
      e.index[7]=in->getNode(7)->getNum();
      hexmesh->add_element(i+1, e);
  }
  outmesh->send(hexmesh);
  ScalarFieldHUG* pfield=new ScalarFieldHUG(hexmesh);
  pfield->data.resize(nodeList.size()+1);
  for (i=0;i<nodeList.size();i++) {
      pfield->data[i+1]=nodeList[i]->getSarc();
  }
  outpower->send(pfield);

  ScalarFieldHUG* vfield=new ScalarFieldHUG(hexmesh);
  vfield->data.resize(nodeList.size()+1);
  for (i=0;i<nodeList.size();i++) {
      Node* n=nodeList[i];
      field->data[i+1]=Vector(n->getVX(), n->getVY(), n->getVZ());
  }
  outfield->send(field);

  cout << "Number of nodes:   " << nodeList.size() << endl;
  cout << "Number of element: " << elementList.size() << endl;
 
  // make element shape functions, attach to Element class
  cout << "Creating element shape functions..." << endl;
  Element::makeShape();
  
  // ----------------------------------------------------------
  // loop through elements and create stiffness for each element
  // ----------------------------------------------------------

  cout << "Creating stiffness for each element." << endl;
  for (i=0;i<elementList.size();i++) {
    elementList[i]->makeStiff();
  }

  // ----------------------------------------------------------
  // assemble the global mass and stiffness matrix
  // ----------------------------------------------------------
  
  cout << "Assembling Global System..." << endl;
  SparseMatrix globalSystem(nodeList, elementList, propertyList);
    
  eps =  propertyList.getEPS();
  maxit = nodeList.size();
  num_time_steps = propertyList.getTimeSteps();
  iteration_total = 0;
  ss = propertyList.getSS();
  writeEvery = propertyList.getWriteEvery();

  // ----------------------------------------------------------
  // now march in time or solve for steady state solution
  // ----------------------------------------------------------
  
  do {

    if (!ss) {
      cout << "Working on step: " << iteration_total + 1
	   << " of " << num_time_steps << endl;
    }
    
    // ----------------------------------------------------------
    // Make the right hand side vector
    // ----------------------------------------------------------
    
    cout << "Making Right hand Side..." << endl;
    globalSystem.makeRhs(nodeList,iteration_total);

    // ----------------------------------------------------------
    // Apply Boundary Conditions
    // ----------------------------------------------------------
    
    cout << "Applying boundary Conditions..." << endl;
    globalSystem.direchlet(nodeList);

    // ----------------------------------------------------------
    // Solve Linear System
    // ----------------------------------------------------------
    
    cout << "Solving System..." << endl;
#if 0
    do {
      if (propertyList.getSymmetric()) {
	solve = globalSystem.solveCG(maxit, eps);
      } else {
	solve = globalSystem.solveQMR(maxit, eps);
      }
    } while ( solve == 3 );
#endif
    if(!globalSystem.solveSCIRun(this)){
	cerr << "Did not get reply from solver module...\n";
	return;
    }
    
    // ----------------------------------------------------------
    // Write A Matrix (left hand side) to file
    // ----------------------------------------------------------

    //cout << "\nWriting A matrix... " << endl;
    //globalSystem.writeMatrix(2);
    
    // ----------------------------------------------------------
    // Write Solution to File
    // ----------------------------------------------------------

    cout << "\nWriting solution... " << endl;
    if (ss) {
      fout.open("OUTFILE");
    } else {
      if (!(iteration_total % writeEvery)) {
	sprintf(str,"OUTFILE.%d",iteration_total+1);
	fout.open(str);
      }
    }
    if (ss || !(iteration_total % writeEvery)) {
      for (i=0;i<nodeList.size();i++) {
	fout << i+1 << " " <<  nodeList[i]->getTemp() << endl;
      }
    }
    fout.close();

    // ----------------------------------------------------------
    // compute residual for transient cases
    // ----------------------------------------------------------

    if (!ss) {
      residual = 0;
      for (i=0;i<nodeList.size();i++) {
	current_residual = fabs(nodeList[i]->getTemp() -
				nodeList[i]->getTempPast());
	if (fabs(nodeList[i]->getTemp()) > 0) {
	  current_residual = current_residual/fabs(nodeList[i]->getTemp());
	}
	if (current_residual > residual) residual = current_residual;
	nodeList[i]->setTempPast(nodeList[i]->getTemp());
      }
    }
    
    iteration_total ++;

    if (!ss) {
      cout << "residual: " << residual << endl;
    }
    
  } while ( (!ss) && (iteration_total < num_time_steps) && (residual > eps) );

  cout << "\n\nDone with FEM " << endl;

  ScalarFieldHUG* field=new ScalarFieldHUG(hexmesh);
  field->data.resize(nodeList.size()+1);
  for (i=0;i<nodeList.size();i++) {
      field->data[i+1]=nodeList[i]->getTemp();
  }
  outfield->send(field);

}

class MatrixAdapter : public Matrix {
    SparseMatrix* mat;
    double dtmp;
public:
    MatrixAdapter(SparseMatrix* mat);
    ~MatrixAdapter();
    virtual double& get(int, int);
    virtual void zero();
    virtual int nrows() const;
    virtual int ncols() const;
    virtual void getRowNonzeros(int r, Array1<int>& idx, Array1<double>& v);
    virtual double minValue();
    virtual double maxValue();
    virtual void mult(const ColumnMatrix& x, ColumnMatrix& b,
		      int& flops, int& memrefs, int beg=-1, int end=-1) const;
    virtual void mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				int& flops, int& memrefs, int beg=-1, int end=-1);
};

MatrixAdapter::MatrixAdapter(SparseMatrix* mat)
: Matrix(Matrix::non_symmetric, Matrix::other), mat(mat)
{
}

MatrixAdapter::~MatrixAdapter()
{
}

double& MatrixAdapter::get(int row, int col) {
    dtmp=mat->getA(row, col);
    return dtmp;
}

void MatrixAdapter::zero() {
    NOT_FINISHED("MatrixAdapter::zero");
}

int MatrixAdapter::nrows() const {
    return mat->size();
}

int MatrixAdapter::ncols() const {
    return mat->size();
}


void MatrixAdapter::getRowNonzeros(int r, Array1<int>& idx, Array1<double>& v) {
    NOT_FINISHED("MatrixAdapter::getRowNonzeros");
}

double MatrixAdapter::minValue() {
    NOT_FINISHED("MatrixAdapter::minValue");
    return 0;
}

double MatrixAdapter::maxValue() {
    NOT_FINISHED("MatrixAdapter::maxValue");
    return 0;
}

void MatrixAdapter::mult(const ColumnMatrix& x, ColumnMatrix& b,
			 int& flops, int& memrefs, int beg, int end) const {
    double* xx=x.get_rhs();
    double* bb=b.get_rhs();
    mat->mult(xx, bb);
}

void MatrixAdapter::mult_transpose(const ColumnMatrix& x, ColumnMatrix& b,
				   int& flops, int& memrefs, int beg, int end) {
    NOT_FINISHED("MatrixAdapter::mult_transpose");
}

bool SparseMatrix::solveSCIRun(TreatFEM* module) {

//Quasi-Minimal Residual Method with Jacobi Preconditioner subroutine
//This solves for a full matrix with all values in it
//   Ax=b
//A is nonsymmetric
// ************  INPUT   *************
// x is the inital guess
// A is the matrix 
// rrhs is the right hand side
// n is the size of the matrix [nxn]

  int n = numberRDOF;

  //solves for the initial residual vector
  
  // put initial guess in x
#if 0
  int i;
  for (i=0;i<numberOfRows;i++) {
    int rdof = nodeList[i]->getRdof();
    if (rdof > -1 ) {
      x[rdof] = nodeList[i]->getTemp();
    }
  }
#endif

  ColumnMatrixHandle scirun_rhs(new ColumnMatrix(numberRDOF));
  // This is ugly, but we will come up with something better soon...
  double* old_data=scirun_rhs->get_rhs();
  scirun_rhs->put_lhs(rrhs);
  
  module->outrhs->send(scirun_rhs);

#if 0
  MatrixHandle scirun_matrix(new MatrixAdapter(this));
#else
  // Convert the matrix instead of adapting it...
  int* rows=new int[numberRDOF+1];
  int i;
  int nnz=0;
  for(i=0;i<numberOfRows;i++){
      int rgrow = nodeList[i]->getRdof();
      if (rgrow > -1) {
	  rows[rgrow]=0;
	  for(int j=0;j<matrix[i].size();j++){
	      int gcol = matrix[i][j]->getIndex();
	      int rgcol = nodeList[gcol]->getRdof();
	      if (rgcol > -1) {
		  nnz++;
		  rows[rgrow]++;
	      }
	  }
      }
  }
  int sum=0;
  for(i=0;i<numberRDOF;i++){
      int tmp=sum;
      sum+=rows[i];
      rows[i]=tmp;
  }
  rows[numberRDOF]=sum;
  ASSERTEQ(sum, nnz);
  double* a=new double[nnz];
  int* cols=new int[nnz];
  for(i=0;i<numberOfRows;i++){
      int rgrow = nodeList[i]->getRdof();
      if (rgrow > -1) {
	  int idx=rows[rgrow];
	  for(int j=0;j<matrix[i].size();j++){
	      int gcol = matrix[i][j]->getIndex();
	      int rgcol = nodeList[gcol]->getRdof();
	      if (rgcol > -1) {
		  a[idx]=getAsparse(i,j);
		  cols[idx++]=rgcol;
	      }
	  }
	  ASSERT(idx==rows[rgrow+1]);
      }
  }
  MatrixHandle scirun_matrix(new SparseRowMatrix(numberRDOF, numberRDOF,
						 rows, cols, nnz, a));
#endif
  module->outmatrix->send(scirun_matrix);

  // Sit around and wait for the reply....

  ColumnMatrixHandle scirun_sol;
  if(!module->loopsol->get(scirun_sol))
      return false;

  ColumnMatrix& ssol=*scirun_sol.get_rep();
  for (i=0;i<numberOfRows;i++) {
    int rgcol = nodeList[i]->getRdof();
    if (rgcol > -1) {
      nodeList[i]->setTemp(ssol[rgcol]);
    }
  }
  scirun_rhs->put_lhs(old_data);
  return true;
}
