/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <iostream>
using std::cerr;
using std::endl;

#define PINVAL 0

namespace BioPSE {

using namespace SCIRun;

class BuildFEMatrix : public Module {
    FieldIPort* inmesh;
    MatrixIPort* refnodeport;
    MatrixOPort* rhsoport;
    MatrixOPort * outmatrix;
#if 0
    void build_local_matrix(Element*, double lcl[4][4],
			    const FieldHandle&);
    void add_lcl_gbl(Matrix&, double lcl[4][4],
		     ColumnMatrix&, int, const FieldHandle&);
    void add_lcl_gbl(Matrix&, double lcl[4][4],
		     ColumnMatrix&, int, const FieldHandle&,
		     int s, int e);
#endif
    int np;
    Barrier barrier;
    int* rows;
    Array1<int> colidx;
    int* allcols;
    TetVol<double>* mesh;
    SparseRowMatrix* gbl_matrix;
    ColumnMatrix* rhs;
    GuiString BCFlag; // do we want Dirichlet conditions applied or PinZero
    int DirSub;	//  matrix decomposition and local regularization later
    int AverageGround; // make the last row in the matrix all 1's
    GuiInt UseCondTCL;
  GuiString refnodeTCL;
    int UseCond;
    int PinZero;
    MatrixHandle gbl_matrixH;
    MatrixHandle rhsH;
    int gen;
    clString lastBCFlag;
    int refnode;
public:
    void parallel(int);
    BuildFEMatrix(const clString& id);
    virtual ~BuildFEMatrix();
    virtual void execute();
};

extern "C" Module* make_BuildFEMatrix(const clString& id) {
  return new BuildFEMatrix(id);
}


BuildFEMatrix::BuildFEMatrix(const clString& id)
: Module("BuildFEMatrix", id, Filter), barrier("BuildFEMatrix barrier"),
    BCFlag("BCFlag", id, this),
  UseCondTCL("UseCondTCL", id, this), refnodeTCL("refnodeTCL", id, this)
{
    // Create the input port
    inmesh = scinew FieldIPort(this, "Mesh", FieldIPort::Atomic);
    add_iport(inmesh);
    refnodeport=scinew MatrixIPort(this, "RefNode", MatrixIPort::Atomic);
    add_iport(refnodeport);

    // Create the output ports
    outmatrix=scinew MatrixOPort(this, "FEM Matrix", MatrixIPort::Atomic);
    add_oport(outmatrix);
    rhsoport=scinew MatrixOPort(this, "RHS", MatrixIPort::Atomic);
    add_oport(rhsoport);
    gen=-1;
}

BuildFEMatrix::~BuildFEMatrix()
{
}
#if 0
void BuildFEMatrix::parallel(int proc)
{
    int nnodes=mesh->nodesize();
    int start_node=nnodes*proc/np;
    int end_node=nnodes*(proc+1)/np;
    int ndof=end_node-start_node;

    int r=start_node;
    int i;
    Array1<int> mycols(0, 15*ndof);
    for(i=start_node;i<end_node;i++){
	rows[r++]=mycols.size();
	if((mesh->node(i).bc && DirSub) || (i==refnode && PinZero)) {
	    mycols.add(i); // Just a diagonal term
	} else if (i==refnode && AverageGround) { // 1's all across
	    for (int ii=0; ii<mesh->nodesize(); ii++) 
		mycols.add(ii);
	} else if (mesh->node(i).pdBC && DirSub) {
	    int nd=mesh->node(i).pdBC->diffNode;
	    if (nd > i) {
		mycols.add(i);
		mycols.add(nd);
	    } else {
		mycols.add(nd);
		mycols.add(i);
	    }
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
     gbl_matrix=scinew SparseRowMatrix(nnodes, nnodes, rows, allcols, st);
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

    int nelems=mesh->elemsize();
    for (i=0; i<nelems; i++){
	Element* e=mesh->element(i);
	if((e->n[0] >= start_node && e->n[0] < end_node)
	   || (e->n[1] >= start_node && e->n[1] < end_node)
	   || (e->n[2] >= start_node && e->n[2] < end_node)
	   || (e->n[3] >= start_node && e->n[3] < end_node)){
	    build_local_matrix(mesh->element(i),lcl_matrix,mesh);
	    add_lcl_gbl(*gbl_matrix,lcl_matrix,*rhs,i,mesh, start_node, end_node);
	}
    }
    for(i=start_node;i<end_node;i++){
	if((mesh->node(i).bc && DirSub) || (PinZero && i==refnode)){
	    // This is just a dummy entry...
//	    (*gbl_matrix)[i][i]=1;
	    int id=rows[i];
	    a[id]=1;
	    if (i==refnode && PinZero)
	      (*rhs)[i]=PINVAL;
	    else
	      (*rhs)[i]=mesh->node(i).bc->value;
	} else if (mesh->node(i).pdBC && DirSub) {
	    int nd=mesh->node(i).pdBC->diffNode;
	    int id=rows[i];
	    if (nd > i) {
		a[id]=1;
		a[id+1]=-1;
	    } else {
		a[id]=-1;
		a[id+1]=1;
	    }
	    (*rhs)[i]=mesh->node(i).pdBC->diffVal;
	} else if (AverageGround && i==refnode) {
	    for (int ii=rows[i]; ii<rows[i]+mesh->nodesize(); ii++)
		a[ii]=1;
	}
    }
}
#endif
void BuildFEMatrix::execute()
{
     FieldHandle mesh;
     if(!inmesh->get(mesh))
	  return;
#if 0
#if 1
     if (mesh->generation == gen && gbl_matrixH.get_rep() && rhsH.get_rep() &&
	 lastBCFlag == BCFlag.get()) {
	 outmatrix->send(gbl_matrix);
	 rhsoport->send(rhs);
	 return;
     }
     gen=mesh->generation;
     UseCond=UseCondTCL.get();
#endif

     this->mesh=mesh.get_rep();
     int nnodes=mesh->nodesize();
     rows=scinew int[nnodes+1];
     np=Thread::numProcessors();
     if (np>10) np=5;
     colidx.resize(np+1);

     refnode=0;
     DirSub=PinZero=AverageGround=0;
     if (BCFlag.get() == "DirSub") DirSub=1;
     else if (BCFlag.get() == "PinZero") { 
       PinZero=1; DirSub=1;
       refnodeTCL.get().get_int(refnode);
     } else if (BCFlag.get() == "AverageGround") { AverageGround=1; DirSub=1; }
     else cerr << "WARNING: BCFlag not set: " << BCFlag.get() << "!\n";
     lastBCFlag=BCFlag.get();

#if 1
     MatrixHandle refnodeH;
     if (refnodeport->get(refnodeH) && 
	 refnodeH.get_rep()&&refnodeH->nrows()>0 &&
	 refnodeH->ncols()>0){
	 refnode=(*refnodeH.get_rep())[0][0];
     }
#endif

     if (PinZero) cerr << "BuildFEM: pinning node "<<refnode<<" to zero.\n";
     if (AverageGround) cerr << "BuildFEM: averaging of all nodes to zero.\n";

     Thread::parallel(Parallel<BuildFEMatrix>(this, &BuildFEMatrix::parallel),
		      np, true);

     gbl_matrixH=MatrixHandle(gbl_matrix);
     outmatrix->send(gbl_matrixH);
     //outmatrix->send(gbl_matrix);
     //cerr << "sent gbl_matrix to matrix port" << endl;
     rhsH=MatrixHandle(rhs);
     rhsoport->send(rhsH);
     //rhsoport->send(rhs);
     //cerr << "sent rhs to coloumn matrix port" << endl;
     this->mesh=0;
#endif
}

#if 0
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

//    if (mesh->element(el)->n[0] < 32 ||
//	mesh->element(el)->n[1] < 32 ||
//	mesh->element(el)->n[2] < 32 ||
//	mesh->element(el)->n[3] < 32) { 
//	cerr << "\n\n\nn[0]="<<mesh->element(el)->n[0]<<" n[1]="<<mesh->element(el)->n[1]<<" n[2]="<<mesh->element(el)->n[2]<<" n3="<<mesh->element(el)->n[3]<<"\n";
 //   }
     for (int i=0; i<4; i++) // this four should eventually be a
	  // variable ascociated with each element that indicates 
	  // how many nodes are on that element. it will change with 
	  // higher order elements
     {	  
	  int ii = mesh->element(el)->n[i];
	  const Node &n1=mesh->node(ii);
	  if (!((n1.bc && DirSub) || (ii==refnode && PinZero) || 
		(ii==refnode && AverageGround))) {
	      for (int j=0; j<4; j++) {
		  int jj = mesh->element(el)->n[j];
		  const Node &n2 = mesh->node(jj);
		  if (n2.bc && DirSub){
		      rhs[ii] -= n2.bc->value*lcl_a[i][j];
		  } else if (jj==refnode && PinZero){
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
	  int ii = mesh->element(el)->n[i];
	  if(ii >= s && ii < e){
	      const Node &n1 = mesh->node(ii);
	      if (!((n1.bc && DirSub) || (ii==refnode && PinZero) ||
		    (ii==refnode && AverageGround))) {
		  for (int j=0; j<4; j++) {
		      int jj = mesh->element(el)->n[j];
		      const Node &n2 = mesh->node(jj);
		      if (n2.bc && DirSub){
			  rhs[ii] -= n2.bc->value*lcl_a[i][j];
		      } else if (jj==refnode && PinZero){
			  rhs[ii] -= PINVAL*lcl_a[i][j];
		      } else {
			  gbl_a[ii][jj] += lcl_a[i][j];
		      }
		  }
	      }
	  }
     }
}
#endif
} // End namespace BioPSE

