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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColumnMatrixPort.h>
#include <Datatypes/MatrixPort.h>
#include <Datatypes/Matrix.h>
#include <Datatypes/SymSparseRowMatrix.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/SurfacePort.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>

class BuildFEMatrix : public Module {
    MeshIPort* inmesh;
    MatrixOPort * outmatrix;
    ColumnMatrixOPort* rhsoport;
    void build_local_matrix(Element*, double lcl[4][4],
			    const MeshHandle&);
    void add_lcl_gbl(Matrix&, double lcl[4][4],
		     ColumnMatrix&, int, const MeshHandle&);
public:
    BuildFEMatrix(const clString& id);
    BuildFEMatrix(const BuildFEMatrix&, int deep);
    virtual ~BuildFEMatrix();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_BuildFEMatrix(const clString& id)
{
    return scinew BuildFEMatrix(id);
}
};

BuildFEMatrix::BuildFEMatrix(const clString& id)
: Module("BuildFEMatrix", id, Filter)
{
    // Create the input port
    inmesh = scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(inmesh);

    // Create the output ports
    outmatrix=scinew MatrixOPort(this, "FEM Matrix", MatrixIPort::Atomic);
    add_oport(outmatrix);
    rhsoport=scinew ColumnMatrixOPort(this, "RHS", ColumnMatrixIPort::Atomic);
    add_oport(rhsoport);

    // Ask Dave about why this was different originally
    // i.e. it was add_iport(scinew MeshIPort(this,"Geometry",...));
}

BuildFEMatrix::BuildFEMatrix(const BuildFEMatrix& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("BuildFEMatrix::BuildFEMatrix");
}

BuildFEMatrix::~BuildFEMatrix()
{
}

Module* BuildFEMatrix::clone(int deep)
{
    return scinew BuildFEMatrix(*this, deep);
}

void BuildFEMatrix::execute()
{
     MeshHandle mesh;
     if(!inmesh->get(mesh))
	  return;
     int nnodes=mesh->nodes.size();

     // SC94 ONLY
     mesh->nodes[nnodes-1]->ndof=0;
     mesh->nodes[nnodes-2]->ndof=0;
     mesh->nodes[nnodes-3]->ndof=0;
     mesh->nodes[nnodes-4]->ndof=0;
     mesh->nodes[nnodes-1]->value=1;
     mesh->nodes[nnodes-2]->value=-1;
     mesh->nodes[nnodes-3]->value=1;
     mesh->nodes[nnodes-4]->value=-1;
     mesh->nodes[nnodes-1]->nodetype=Node::VSource;
     mesh->nodes[nnodes-2]->nodetype=Node::VSource;
     mesh->nodes[nnodes-3]->nodetype=Node::VSource;
     mesh->nodes[nnodes-4]->nodetype=Node::VSource;

     int ndof=nnodes;
     Array1<int> rows(ndof+1);
     Array1<int> cols;
     int r=0;
     for(int i=0;i<nnodes;i++){
	 rows[r++]=cols.size();
	 if(mesh->nodes[i]->ndof > 0){
	     mesh->add_node_neighbors(i, cols);
	 } else {
	     cols.add(i); // Just a diagonal term
	 }
     }
     rows[r]=cols.size();
     Matrix* gbl_matrix=scinew SymSparseRowMatrix(ndof, ndof, rows, cols);
     gbl_matrix->zero();
     ColumnMatrix* rhs=scinew ColumnMatrix(ndof);
     rhs->zero();
     double lcl_matrix[4][4];

     int nelems=mesh->elems.size();
     for (i=0; i<nelems; i++){
	 if(i%200 == 0)
	     update_progress(i, nelems);
	 build_local_matrix(mesh->elems[i],lcl_matrix,mesh);
	 add_lcl_gbl(*gbl_matrix,lcl_matrix,*rhs,i,mesh);
     }
     for(i=0;i<nnodes;i++){
	 if(mesh->nodes[i]->ndof == 0){
	     // This is just a dummy entry...
	     (*gbl_matrix)[i][i]=1;
	 }
     }

     outmatrix->send(MatrixHandle(gbl_matrix));
     rhsoport->send(ColumnMatrixHandle(rhs));
}

void BuildFEMatrix::build_local_matrix(Element *elem, 
				       double lcl_a[4][4],
				       const MeshHandle& mesh)
{
     Point pt;
     Vector grad1,grad2,grad3,grad4;
     double vol = mesh->get_grad(elem,pt,grad1,grad2,grad3,grad4);


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
     // in el_cond, the indices tell you the directions
     // the value is refering to. i.e. 0=x, 1=y, and 2=z
     // so el_cond[1][2] is the same as sigma yz
     el_cond[0][0] = mesh->cond_tensors[elem->cond][0];
     el_cond[0][1] = mesh->cond_tensors[elem->cond][1];
     el_cond[1][0] = mesh->cond_tensors[elem->cond][1];
     el_cond[0][2] = mesh->cond_tensors[elem->cond][2];
     el_cond[2][0] = mesh->cond_tensors[elem->cond][2];
     el_cond[1][1] = mesh->cond_tensors[elem->cond][3];
     el_cond[1][2] = mesh->cond_tensors[elem->cond][4];
     el_cond[2][1] = mesh->cond_tensors[elem->cond][4];
     el_cond[2][2] = mesh->cond_tensors[elem->cond][5];


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

     for (int i=0; i<4; i++) // this four should eventually be a
	  // variable ascociated with each element that indicates 
	  // how many nodes are on that element. it will change with 
	  // higher order elements
     {	  
	  int ii = mesh->elems[el]->n[i];
	  NodeHandle& n1=mesh->nodes[ii];
	  if(n1->ndof > 0){
	      for (int j=0; j<4; j++) {
		  int jj = mesh->elems[el]->n[j];
		  NodeHandle& n2=mesh->nodes[jj];
		  if(n2->ndof > 0){
		      gbl_a[ii][jj] += lcl_a[i][j];
		  } else {
		      // Eventually look at nodetype...
		      rhs[ii]-=n2->value*lcl_a[i][j];
		  }
	      }
	  }
     }
}
