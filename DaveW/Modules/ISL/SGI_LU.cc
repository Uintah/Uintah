/****************************************************************
 *  Simple "SGI_LU module" for SCIRun                           *
 *                                                              *
 *  Written by:                                                 *
 *   Leonid Zhukov                                              *
 *   Department of Computer Science                             *
 *   University of Utah                                         *
 *   May 99                                                     *
 *                                                              *
 *  Copyright (C) 1999 SCI Group                                *
 *                                                              *
 *                                                              *
 ****************************************************************/

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <SCICore/Datatypes/SparseRowMatrix.h>
#include <SCICore/Datatypes/SymSparseRowMatrix.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <iostream>
using std::cerr;
using std::endl;
#include <time.h>

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::Containers;

extern "C"{
void PSLDLT_Preprocess (int token,int n,int pointers[],int indices[],int *nonz,double *ops);
 void PSLDLT_Factor (int token,int n,int pointers[],int indices[],double values[]);

  void PSLDLT_Ordering (int token,int method);
}


class SGI_LU : public Module {

 MatrixIPort* matrix_in_port;
 ColumnMatrixOPort* factor_port;
  
  
public:
    int gen;
    ColumnMatrixHandle factorH;
  TCLstring tcl_status;
  SGI_LU(const clString& id);
  virtual ~SGI_LU();
  virtual void execute();
  
}; //class


Module* make_SGI_LU(const clString& id) {
    return new SGI_LU(id);
}

//---------------------------------------------------------------
SGI_LU::SGI_LU(const clString& id)
  : Module("SGI_LU", id, Filter),
    tcl_status("tcl_status",id,this)

{

matrix_in_port=new MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(matrix_in_port);
factor_port=new ColumnMatrixOPort(this, "SGI_LU",ColumnMatrixIPort::Atomic);
    add_oport(factor_port);
    gen=-1;
}

//------------------------------------------------------------
SGI_LU::~SGI_LU(){}

//--------------------------------------------------------------

void SGI_LU::execute()
{


  tcl_status.set("Calling SGI_LU!");
  
  MatrixHandle matrix;
  if(!matrix_in_port->get(matrix))
    return; 
  if (matrix->generation == gen) 
{ factor_port->send(factorH);
return;} 
 gen=matrix->generation;
  int nrows = matrix->nrows();
  int nonzero =  matrix->get_row()[nrows];

SymSparseRowMatrix *ss_matrix=matrix->getSymSparseRow();
 
 int count = 0;
 int nnz = (nonzero -nrows)/2 + nrows; 
 int* indices = new int [nnz]; 
 int* pointers = new int [nrows+1];
 double* values = new double[nnz];  
 int i,j,less;

 pointers[0] = ss_matrix->rows[0];

 for(i=0;i<nrows;i++){
   
   less = 0;

   for (j=ss_matrix->rows[i];j<ss_matrix->rows[i+1];j++){

     if (ss_matrix->columns[j] >=i){
       // cout <<"i = "<<i<<", j = "<<j<<endl;
       indices[count] = ss_matrix->columns[j];
       values[count] = ss_matrix->a[j];
       count++;
     }
     else{
      less++;
     }
   }
 //  cout <<"less = "<<less<<endl;
 pointers[i+1] =pointers[i]+(ss_matrix->rows[i+1]-ss_matrix->rows[i])-less;
}


 /*
for (i=0;i<nrows+1;i++){
  cerr <<pointers[i]<<" ";
 }
  cerr <<endl;

 for (i=0;i<nnz;i++){
    cerr <<indices[i]<<" ";
 }
 cerr <<endl;

   
 for (i=0;i<nnz;i++){
   cerr <<values[i]<<" ";
 }
 cerr <<endl;
 */

 int token = 1; 
 double nfpo;
 int method = 2; 


 double start,end;
 
 cerr<<"Odering ...";
 start = time(0);
 PSLDLT_Ordering(token,method);
 end = time(0); 
 cerr<<"...Done!  Time = "<<(end-start)<<" s"<<endl;

 
 cerr<<"Preprocess ...";
 start = time(0);
 PSLDLT_Preprocess(token,nrows,pointers,indices,&nnz,&nfpo);
 end = time(0); 
 cerr<<"...Done!  Time = "<<(end-start)<<" s"<<endl;

 

 cerr<<"Factor ...";
 start = time(0); 
 PSLDLT_Factor(token,nrows,pointers,indices, values);
 end = time(0); 
 cerr<<"...Done!  Time = "<<(end-start)<<" s"<<endl;


 double *factor_data = new double[2];
 factor_data[0] = token;
 factor_data[1] = nrows;
 
 ColumnMatrix* factor = new ColumnMatrix(2);
 factorH=factor;
 factor->put_lhs(factor_data);
 factor_port->send(factorH);
  
} 
//---------------------------------------------------------------

} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.2  1999/10/07 02:06:37  sparker
// use standard iostreams and complex type
//
// Revision 1.1  1999/09/02 04:50:04  dmw
// more of Dave's modules
//
//
