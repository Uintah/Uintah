/*
 *  BuildFEMatrixQuadratic.cc:
 *
 *  Written by:
 *   vanuiter
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/RobV/share/share.h>

namespace RobV {

using namespace SCIRun;

class RobVSHARE BuildFEMatrixQuadratic : public Module {
public:
  BuildFEMatrixQuadratic(const string& id);

  virtual ~BuildFEMatrixQuadratic();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" RobVSHARE Module* make_BuildFEMatrixQuadratic(const string& id) {
  return scinew BuildFEMatrixQuadratic(id);
}

BuildFEMatrixQuadratic::BuildFEMatrixQuadratic(const string& id)
  : Module("BuildFEMatrixQuadratic", id, Source, "Quadratic", "RobV")
{
}

BuildFEMatrixQuadratic::~BuildFEMatrixQuadratic(){
}

void BuildFEMatrixQuadratic::execute(){
  /*


     MeshHandle mesh;
     if(!inmesh->get(mesh))
	  return;

time_t current_time = time(NULL);
printf("Beginning simulation: %s\n",ctime(&current_time));


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
     int nnodes=mesh->nodes.size(); 
     rows=scinew int[nnodes+1];  
     np=Thread::numProcessors();
     if (np>10) np/=2;
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
     ColumnMatrixHandle refnodeH;
     if (refnodeport->get(refnodeH)&&refnodeH.get_rep()&&refnodeH->nrows()>0){
	 refnode=(*refnodeH.get_rep())[0];
     }
#endif

     if (PinZero) cerr << "BuildFEM: pinning node "<<refnode<<" to zero.\n";
     if (AverageGround) cerr << "BuildFEM: averaging of all nodes to zero.\n";

     Thread::parallel(Parallel<BuildFEMatrixQuadratic>(this, &BuildFEMatrixQuadratic::parallel), np, true);

     //     for (int i=0; i<nnodes; i++) {
     //for (int j=0; j<nnodes; j++) {
     //  cerr << gbl_matrix->get(i,j);
//	 cerr << " ";
//       }
//       cerr << "\n";
//       }
     

     gbl_matrixH=MatrixHandle(gbl_matrix);
     outmatrix->send(gbl_matrixH);
     //outmatrix->send(gbl_matrix);
     //cerr << "sent gbl_matrix to matrix port" << endl;
     rhsH=ColumnMatrixHandle(rhs);
     rhsoport->send(rhsH);
     //rhsoport->send(rhs);
     //cerr << "sent rhs to coloumn matrix port" << endl;
     this->mesh=0;
   */
}

  /*

void BuildFEMatrixQuadratic::parallel(int proc)
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
	if((mesh->nodes[i]->bc && DirSub) || (i==refnode && PinZero)) {
	    mycols.add(i); // Just a diagonal term
	} else if (i==refnode && AverageGround) { // 1's all across
	    for (int ii=0; ii<mesh->nodes.size(); ii++) 
		mycols.add(ii);
	} else if (mesh->nodes[i]->pdBC && DirSub) {
	    int nd=mesh->nodes[i]->pdBC->diffNode;
	    if (nd > i) {
		mycols.add(i);
		mycols.add(nd);
	    } else {
		mycols.add(nd);
		mycols.add(i);
	    }
	} else {
	  mesh->add_node_neighborsQuad(i, mycols, DirSub);
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
    double lcl_matrix[10][10];

    int nelems=mesh->elems.size();
    for (i=0; i<nelems; i++){
	Element* e=mesh->elems[i];
	if((e->n[0] >= start_node && e->n[0] < end_node)
	   || (e->n[1] >= start_node && e->n[1] < end_node)
	   || (e->n[2] >= start_node && e->n[2] < end_node)
	   || (e->n[3] >= start_node && e->n[3] < end_node)
	   || (e->xtrpts[0] >= start_node && e->xtrpts[0] < end_node)
	   || (e->xtrpts[1] >= start_node && e->xtrpts[1] < end_node)
	   || (e->xtrpts[2] >= start_node && e->xtrpts[2] < end_node)
	   || (e->xtrpts[3] >= start_node && e->xtrpts[3] < end_node)
	   || (e->xtrpts[4] >= start_node && e->xtrpts[4] < end_node)
	   || (e->xtrpts[5] >= start_node && e->xtrpts[5] < end_node)){
	  //	  mutex.lock();
	    build_local_matrix(mesh->elems[i],lcl_matrix,mesh);
	    add_lcl_gbl(*gbl_matrix,lcl_matrix,*rhs,i,mesh, start_node, end_node);
	    //	    mutex.unlock();
	}
    }
    for(i=start_node;i<end_node;i++){
	if((mesh->nodes[i]->bc && DirSub) || (PinZero && i==refnode)){
	    // This is just a dummy entry...
//	    (*gbl_matrix)[i][i]=1;
	    int id=rows[i];
	    a[id]=1;
	    if (i==refnode && PinZero)
	      (*rhs)[i]=PINVAL;
	    else
	      (*rhs)[i]=mesh->nodes[i]->bc->value;
	} else if (mesh->nodes[i]->pdBC && DirSub) {
	    int nd=mesh->nodes[i]->pdBC->diffNode;
	    int id=rows[i];
	    if (nd > i) {
		a[id]=1;
		a[id+1]=-1;
	    } else {
		a[id]=-1;
		a[id+1]=1;
	    }
	    (*rhs)[i]=mesh->nodes[i]->pdBC->diffVal;
	} else if (AverageGround && i==refnode) {
	    for (int ii=rows[i]; ii<rows[i]+mesh->nodes.size(); ii++)
		a[ii]=1;
	}
    }
}

void BuildFEMatrixQuadratic::build_local_matrix(Element *elem, 
						double lcl_a[10][10],
				       const MeshHandle& mesh)
{
    Point pt;
    Vector grad1,grad2,grad3,grad4,grad5,grad6,grad7,grad8,grad9,grad10;
    double vol;

    double el_coefs[5][10][3];  
    // this 4x3 array holds the 3 gradients to be used 
    // as coefficients for each of the four nodes of the 
    // element

    double wst[5];
    wst[0] = -0.8;
    wst[1] = 0.45;
    wst[2] = 0.45;
    wst[3] = 0.45;
    wst[4] = 0.45;

    double jac_el;

    for (int l=0; l<5; l++) {

      jac_el = mesh->get_gradQuad(l,elem,pt,grad1,grad2,grad3,grad4,grad5,grad6,grad7,grad8,grad9,grad10);

      Vector l1,l2,l3,l4; 
      vol = mesh->get_grad(elem,pt,l1,l2,l3,l4);  //get volume by using linear
      if(vol < 1.e-10){
	cerr << "Skipping element..., volume=" << vol << endl;
	for(int i=0;i<10;i++)
	  for(int j=i;j<10;j++) {
	    lcl_a[i][j]=0;
	    lcl_a[j][i] = lcl_a[i][j];
	  }
	return;
      }
      
      el_coefs[l][0][0]=grad1.x();
      el_coefs[l][0][1]=grad1.y();
      el_coefs[l][0][2]=grad1.z();
    
      el_coefs[l][1][0]=grad2.x();
      el_coefs[l][1][1]=grad2.y();
      el_coefs[l][1][2]=grad2.z();

      el_coefs[l][2][0]=grad3.x();
      el_coefs[l][2][1]=grad3.y();
      el_coefs[l][2][2]=grad3.z();

      el_coefs[l][3][0]=grad4.x();
      el_coefs[l][3][1]=grad4.y();
      el_coefs[l][3][2]=grad4.z();

      el_coefs[l][4][0]=grad5.x();
      el_coefs[l][4][1]=grad5.y();
      el_coefs[l][4][2]=grad5.z();

      el_coefs[l][5][0]=grad6.x();
      el_coefs[l][5][1]=grad6.y();
      el_coefs[l][5][2]=grad6.z();
    
      el_coefs[l][6][0]=grad7.x();
      el_coefs[l][6][1]=grad7.y();
      el_coefs[l][6][2]=grad7.z();

      el_coefs[l][7][0]=grad8.x();
      el_coefs[l][7][1]=grad8.y();
      el_coefs[l][7][2]=grad8.z();

      el_coefs[l][8][0]=grad9.x();
      el_coefs[l][8][1]=grad9.y();
      el_coefs[l][8][2]=grad9.z();

      el_coefs[l][9][0]=grad10.x();
      el_coefs[l][9][1]=grad10.y();
      el_coefs[l][9][2]=grad10.z();
    }

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
    for(int i=0; i< 10; i++) {
	for(int j=i; j< 10; j++) {
	  double I = 0.0;
	  for(int l=0; l< 5; l++){
	    
	    double xx = el_cond[0][0]*jac_el*el_coefs[l][i][0]*el_coefs[l][j][0];
	    double xy = el_cond[0][1]*jac_el*el_coefs[l][i][0]*el_coefs[l][j][1];
	    double xz = el_cond[0][2]*jac_el*el_coefs[l][i][0]*el_coefs[l][j][2];
	    double yx = el_cond[1][0]*jac_el*el_coefs[l][i][1]*el_coefs[l][j][0];
	    double yy = el_cond[1][1]*jac_el*el_coefs[l][i][1]*el_coefs[l][j][1];
	    double yz = el_cond[1][2]*jac_el*el_coefs[l][i][1]*el_coefs[l][j][2];
	    double zx = el_cond[2][0]*jac_el*el_coefs[l][i][2]*el_coefs[l][j][0];
	    double zy = el_cond[2][1]*jac_el*el_coefs[l][i][2]*el_coefs[l][j][1];
	    double zz = el_cond[2][2]*jac_el*el_coefs[l][i][2]*el_coefs[l][j][2];
	    
	    double I_nl = xx + xy + xz + yx + yy + yz + zx + zy + zz;
	    
	    I += wst[l]*I_nl;
	  }
	  lcl_a[i][j] = I/6.0;
	  lcl_a[j][i] = lcl_a[i][j];
	  //	  cerr << "Local: " << i << " " << j << " " << lcl_a[i][j] << "\n";
	}
    }
}


void BuildFEMatrixQuadratic::add_lcl_gbl(Matrix& gbl_a, double lcl_a[10][10],
				ColumnMatrix& rhs,
				int el, const MeshHandle& mesh)
{

//    if (mesh->elems[el]->n[0] < 32 ||
//	mesh->elems[el]->n[1] < 32 ||
//	mesh->elems[el]->n[2] < 32 ||
//	mesh->elems[el]->n[3] < 32) { 
//	cerr << "\n\n\nn[0]="<<mesh->elems[el]->n[0]<<" n[1]="<<mesh->elems[el]->n[1]<<" n[2]="<<mesh->elems[el]->n[2]<<" n3="<<mesh->elems[el]->n[3]<<"\n";
 //   }
  for (int i=0; i<10; i++) // this four should eventually be a
	  // variable ascociated with each element that indicates 
	  // how many nodes are on that element. it will change with 
	  // higher order elements
     {	  
          int ii;
	  if (i <4 ) ii = mesh->elems[el]->n[i];
	  else ii = mesh->elems[el]->xtrpts[i-4];
	
	  NodeHandle& n1=mesh->nodes[ii];
	  
	  if (!((n1->bc && DirSub) || (ii==refnode && PinZero) || 
		(ii==refnode && AverageGround))) {
	      for (int j=0; j<10; j++) {
		int jj;
		if (j <4 ) jj = mesh->elems[el]->n[j];
		else jj = mesh->elems[el]->xtrpts[j-4];

		  NodeHandle& n2=mesh->nodes[jj];
		  if (n2->bc && DirSub){
		      rhs[ii] -= n2->bc->value*lcl_a[i][j];
		  } else if (jj==refnode && PinZero){
		      rhs[ii] -= PINVAL*lcl_a[i][j];
		  } else {
		      gbl_a[ii][jj] += lcl_a[i][j];
		  }
	      }
	  }
     }
}

void BuildFEMatrixQuadratic::add_lcl_gbl(Matrix& gbl_a, double lcl_a[10][10],
				ColumnMatrix& rhs,
				int el, const MeshHandle& mesh,
				int s, int e)
{

  for (int i=0; i<10; i++) // this four should eventually be a
	  // variable ascociated with each element that indicates 
	  // how many nodes are on that element. it will change with 
	  // higher order elements
     {	  

          int ii;
	  if (i <4 ) ii = mesh->elems[el]->n[i];
	  else ii = mesh->elems[el]->xtrpts[i-4];

	  if(ii >= s && ii < e){
	      NodeHandle& n1=mesh->nodes[ii];
	      if (!((n1->bc && DirSub) || (ii==refnode && PinZero) ||
		    (ii==refnode && AverageGround))) {
		  for (int j=0; j<10; j++) {
		    int jj=0;
		    if (j <4 ) jj = mesh->elems[el]->n[j];
		    else jj = mesh->elems[el]->xtrpts[j-4];

		      NodeHandle& n2=mesh->nodes[jj];
		      if (n2->bc && DirSub){
			  rhs[ii] -= n2->bc->value*lcl_a[i][j];
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
*/


void BuildFEMatrixQuadratic::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace RobV


