//static char *id="@(#) $Id$";

/*
 *  BuildFEMatrix.cc:  Builds the global finite element matrix
 *
 *  Written by:
 *   Ruth Nicholson Klepfer
 *   Department of Bioengineering
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Util/NotFinished.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/ColumnMatrixPort.h>
#include <PSECore/CommonDatatypes/MatrixPort.h>
#include <SCICore/CoreDatatypes/Matrix.h>
#include <SCICore/CoreDatatypes/SymSparseRowMatrix.h>
#include <PSECore/CommonDatatypes/MeshPort.h>
#include <SCICore/CoreDatatypes/Mesh.h>
#include <PSECore/CommonDatatypes/SurfacePort.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Multitask/ITC.h>
#include <SCICore/Multitask/Task.h>
#include <SCICore/TclInterface/TCLvar.h>

#define PINVAL 1

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Multitask;

class BuildFEMatrix : public Module {
    MeshIPort* inmesh;
    MatrixOPort * outmatrix;
    ColumnMatrixOPort* rhsoport;
    void build_local_matrix(Element*, double lcl[4][4],
			    const MeshHandle&);
    void add_lcl_gbl(Matrix&, double lcl[4][4],
		     ColumnMatrix&, int, const MeshHandle&);
    void add_lcl_gbl(Matrix&, double lcl[4][4],
		     ColumnMatrix&, int, const MeshHandle&,
		     int s, int e);
    int np;
    Barrier barrier;
    int* rows;
    Array1<int> colidx;
    int* allcols;
    Mesh* mesh;
    SymSparseRowMatrix* gbl_matrix;
    ColumnMatrix* rhs;
    TCLstring BCFlag; // do we want Dirichlet conditions applied or PinZero
    int DirSub;	//  matrix decomposition and local regularization later
    TCLint UseCondTCL;
    int UseCond;
    int PinZero;
    MatrixHandle gbl_matrixH;
    ColumnMatrixHandle rhsH;
    int gen;
public:
    void parallel(int);
    BuildFEMatrix(const clString& id);
    virtual ~BuildFEMatrix();
    virtual void execute();
};

static void do_parallel(void* obj, int proc)
{
    BuildFEMatrix* module=(BuildFEMatrix*)obj;
    module->parallel(proc);
}
    

Module* make_BuildFEMatrix(const clString& id) {
  return new BuildFEMatrix(id);
}


BuildFEMatrix::BuildFEMatrix(const clString& id)
: Module("BuildFEMatrix", id, Filter), BCFlag("BCFlag", id, this),
  UseCondTCL("UseCondTCL", id, this)
{
    // Create the input port
    inmesh = scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(inmesh);

    // Create the output ports
    outmatrix=scinew MatrixOPort(this, "FEM Matrix", MatrixIPort::Atomic);
    add_oport(outmatrix);
    rhsoport=scinew ColumnMatrixOPort(this, "RHS", ColumnMatrixIPort::Atomic);
    add_oport(rhsoport);
    gen=-1;
}

BuildFEMatrix::~BuildFEMatrix()
{
}

void BuildFEMatrix::parallel(int proc)
{
    int nnodes=mesh->nodes.size();
    int start_node=nnodes*proc/np;
    int end_node=nnodes*(proc+1)/np;
    int ndof=end_node-start_node;

    int r=start_node;
    int i;
    Array1<int> mycols(0, 15*ndof);
    for(i=start_node;i<end_node;i++){
	rows[r++]=mycols.size();
	if((mesh->nodes[i]->bc && DirSub) || (i==0 && PinZero)) {
	    mycols.add(i); // Just a diagonal term
	} else {
	    mesh->add_node_neighbors(i, mycols, DirSub);
	}
    }
    colidx[proc]=mycols.size();
    if(proc == 0)
      update_progress(1,6);
    barrier.wait(np);
    int st=0;
    if(proc == 0){
      update_progress(2,6);
      for(i=0;i<np;i++){
	int ns=st+colidx[i];
	colidx[i]=st;
	st=ns;
      }
      colidx[np]=st;
      cerr << "st=" << st << endl;
      allcols=scinew int[st];
    }
      
    barrier.wait(np);
    int s=colidx[proc];

    int n=mycols.size();
    for(i=0;i<n;i++){
	allcols[i+s]=mycols[i];
    }
    for(i=start_node;i<end_node;i++){
	rows[i]+=s;
    }
    barrier.wait(np);

    // The main thread makes the matrix and rhs...
    if(proc == 0){
     rows[nnodes]=st;
     update_progress(3,6);
     cerr << "There are " << st << " non zeros" << endl;
     gbl_matrix=scinew SymSparseRowMatrix(nnodes, nnodes, rows, allcols, st);
     rhs=scinew ColumnMatrix(nnodes);
    }
    barrier.wait(np);

    double* a=gbl_matrix->a;
    for(i=start_node;i<end_node;i++){
	(*rhs)[i]=0;
    }
    int ns=colidx[proc];
    int ne=colidx[proc+1];
    for(i=ns;i<ne;i++){
	a[i]=0;
    }
    double lcl_matrix[4][4];

    int nelems=mesh->elems.size();
    for (i=0; i<nelems; i++){
	Element* e=mesh->elems[i];
	if((e->n[0] >= start_node && e->n[0] < end_node)
	   || (e->n[1] >= start_node && e->n[1] < end_node)
	   || (e->n[2] >= start_node && e->n[2] < end_node)
	   || (e->n[3] >= start_node && e->n[3] < end_node)){
	    build_local_matrix(mesh->elems[i],lcl_matrix,mesh);
	    add_lcl_gbl(*gbl_matrix,lcl_matrix,*rhs,i,mesh, start_node, end_node);
	}
    }
    for(i=start_node;i<end_node;i++){
	if((mesh->nodes[i]->bc && DirSub) || (PinZero && i==0)){
	    // This is just a dummy entry...
//	    (*gbl_matrix)[i][i]=1;
	    int id=rows[i];
	    a[id]=1;
	    if (i==0 && PinZero)
	      (*rhs)[i]=PINVAL;
	    else
	      (*rhs)[i]=mesh->nodes[i]->bc->value;
	}
    }
}

void BuildFEMatrix::execute()
{
     MeshHandle mesh;
     if(!inmesh->get(mesh))
	  return;
     if (mesh->generation == gen && gbl_matrixH.get_rep() && rhsH.get_rep()) {
	 outmatrix->send(gbl_matrixH);
	 rhsoport->send(rhsH);
	 return;
     }
     gen=mesh->generation;
     UseCond=UseCondTCL.get();

     this->mesh=mesh.get_rep();
     int nnodes=mesh->nodes.size();
     rows=scinew int[nnodes+1];
     np=Task::nprocessors();
     if (np>10) np=5;
     colidx.resize(np+1);

     DirSub=PinZero=0;
     if (BCFlag.get() == "DirSub") DirSub=1;
     else if (BCFlag.get() == "PinZero") PinZero=1;
  

     Task::multiprocess(np, do_parallel, this);

     gbl_matrixH=gbl_matrix;
     outmatrix->send(gbl_matrixH);
     rhsH=rhs;
     rhsoport->send(rhsH);
     this->mesh=0;
}

void BuildFEMatrix::build_local_matrix(Element *elem, 
				       double lcl_a[4][4],
				       const MeshHandle& mesh)
{
    Point pt;
    Vector grad1,grad2,grad3,grad4;
    double vol = mesh->get_grad(elem,pt,grad1,grad2,grad3,grad4);
    if(vol < 1.e-10){
	cerr << "Skipping element..., volume=" << vol << endl;
	for(int i=0;i<4;i++)
	    for(int j=0;j<4;j++)
		lcl_a[i][j]=0;
	return;
    }
    

    double el_coefs[4][3];
    // this 4x3 array holds the 3 gradients to be used 
    // as coefficients for each of the four nodes of the 
    // element
    
    el_coefs[0][0]=grad1.x();
    el_coefs[0][1]=grad1.y();
    el_coefs[0][2]=grad1.z();
    
    el_coefs[1][0]=grad2.x();
    el_coefs[1][1]=grad2.y();
    el_coefs[1][2]=grad2.z();

    el_coefs[2][0]=grad3.x();
    el_coefs[2][1]=grad3.y();
    el_coefs[2][2]=grad3.z();

    el_coefs[3][0]=grad4.x();
    el_coefs[3][1]=grad4.y();
    el_coefs[3][2]=grad4.z();

    // cond_tensors are the sigma values for this element.
    // where:
    //  [0] => sigma xx
    //  [1] => sigma xy and sigma yx
    //  [2] => sigma xz and sigma zx
    //  [3] => sigma yy
    //  [4] => sigma yz and sigma zy
    //  [5] => sigma zz

    double el_cond[3][3];
    el_cond[0][0]=el_cond[1][1]=el_cond[2][2]=1;
    el_cond[0][1]=el_cond[1][0]=el_cond[0][2]=el_cond[2][0]=el_cond[1][2]=el_cond[2][1]=0;

    // in el_cond, the indices tell you the directions
    // the value is refering to. i.e. 0=x, 1=y, and 2=z
    // so el_cond[1][2] is the same as sigma yz

    if (UseCond) {
	el_cond[0][0] = mesh->cond_tensors[elem->cond][0];
	el_cond[0][1] = mesh->cond_tensors[elem->cond][1];
	el_cond[1][0] = mesh->cond_tensors[elem->cond][1];
	el_cond[0][2] = mesh->cond_tensors[elem->cond][2];
	el_cond[2][0] = mesh->cond_tensors[elem->cond][2];
	el_cond[1][1] = mesh->cond_tensors[elem->cond][3];
	el_cond[1][2] = mesh->cond_tensors[elem->cond][4];
	el_cond[2][1] = mesh->cond_tensors[elem->cond][4];
	el_cond[2][2] = mesh->cond_tensors[elem->cond][5];
    }

    // build the local matrix
    for(int i=0; i< 4; i++) {
	for(int j=0; j< 4; j++) {
	    lcl_a[i][j] = 0.0;
	    for (int k=0; k< 3; k++){
		for (int l=0; l<3; l++){
		    lcl_a[i][j] += 
			el_cond[k][l]*el_coefs[i][k]*el_coefs[j][l];
		}
	    }
	    lcl_a[i][j] *= vol;
	}
    }
}


void BuildFEMatrix::add_lcl_gbl(Matrix& gbl_a, double lcl_a[4][4],
				ColumnMatrix& rhs,
				int el, const MeshHandle& mesh)
{

//    if (mesh->elems[el]->n[0] < 32 ||
//	mesh->elems[el]->n[1] < 32 ||
//	mesh->elems[el]->n[2] < 32 ||
//	mesh->elems[el]->n[3] < 32) { 
//	cerr << "\n\n\nn[0]="<<mesh->elems[el]->n[0]<<" n[1]="<<mesh->elems[el]->n[1]<<" n[2]="<<mesh->elems[el]->n[2]<<" n3="<<mesh->elems[el]->n[3]<<"\n";
 //   }
     for (int i=0; i<4; i++) // this four should eventually be a
	  // variable ascociated with each element that indicates 
	  // how many nodes are on that element. it will change with 
	  // higher order elements
     {	  
	  int ii = mesh->elems[el]->n[i];
	  NodeHandle& n1=mesh->nodes[ii];
	  if (!((n1->bc && DirSub) || (ii==0 && PinZero))) {
	      for (int j=0; j<4; j++) {
		  int jj = mesh->elems[el]->n[j];
		  NodeHandle& n2=mesh->nodes[jj];
		  if (n2->bc && DirSub){
		      rhs[ii] -= n2->bc->value*lcl_a[i][j];
		  } else if (jj==0 && PinZero){
		      rhs[ii] -= PINVAL*lcl_a[i][j];
		  } else {
		      gbl_a[ii][jj] += lcl_a[i][j];
		  }
	      }
	  }
     }
}

void BuildFEMatrix::add_lcl_gbl(Matrix& gbl_a, double lcl_a[4][4],
				ColumnMatrix& rhs,
				int el, const MeshHandle& mesh,
				int s, int e)
{

     for (int i=0; i<4; i++) // this four should eventually be a
	  // variable ascociated with each element that indicates 
	  // how many nodes are on that element. it will change with 
	  // higher order elements
     {	  
	  int ii = mesh->elems[el]->n[i];
	  if(ii >= s && ii < e){
	      NodeHandle& n1=mesh->nodes[ii];
	      if ((!n1->bc || !DirSub) && (ii!=0 || !PinZero)) {
		  for (int j=0; j<4; j++) {
		      int jj = mesh->elems[el]->n[j];
		      NodeHandle& n2=mesh->nodes[jj];
		      if (n2->bc && DirSub){
			  rhs[ii] -= n2->bc->value*lcl_a[i][j];
		      } else if (jj==0 && PinZero){
			  rhs[ii] -= PINVAL*lcl_a[i][j];
		      } else {
			  gbl_a[ii][jj] += lcl_a[i][j];
		      }
		  }
	      }
	  }
     }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.3  1999/08/18 20:19:36  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:25  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:39  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:47  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
