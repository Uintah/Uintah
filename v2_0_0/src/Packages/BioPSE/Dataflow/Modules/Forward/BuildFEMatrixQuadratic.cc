//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : BuildFEMatrixQuadratic.cc
//    Author : Robert L. Van Uitert, Martin Cole
//    Date   : Thu Mar 14 19:23:04 2002


#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/QuadraticTetVolField.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geometry/Point.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Barrier.h>
#include <Core/Containers/StringUtil.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>

#define PINVAL 0

namespace BioPSE {

using namespace SCIRun;

class BuildFEMatrixQuadratic : public Module {
public:
  BuildFEMatrixQuadratic(GuiContext *context);

  virtual ~BuildFEMatrixQuadratic();

  virtual void execute();

private:
  FieldIPort               *ifld_;
  MatrixIPort              *imat_;
  MatrixOPort              *rhsoport;
  MatrixOPort              *outmatrix;

  int np;
  Barrier barrier;
  int* rows;
  Array1<int> colidx;
  int* allcols;
  QuadraticTetVolMeshHandle qtvm_;
  SparseRowMatrix* gbl_matrix;
  ColumnMatrix* rhs;
  GuiString BCFlag; // do we want Dirichlet conditions applied or PinZero
  int DirSub;	//  matrix decomposition and local regularization later
  int AverageGround; // make the last row in the matrix all 1's
  GuiInt UseCondGui;
  GuiInt refnodeGui;
  int UseCond;
  int PinZero;
  MatrixHandle gbl_matrixH;
  MatrixHandle rhsH;
  int gen;
  string lastBCFlag;
  int refnode;
  QuadraticTetVolField<int>* qtv;
  vector<bool> bcArray;
  vector<pair<int, double> > dirichlet;

  Mutex mutex;

  void parallel(int);
  void build_local_matrix(double lcl[10][10], TetVolMesh::Cell::index_type);
  void add_lcl_gbl(Matrix&, double lcl[10][10],
		   ColumnMatrix&, TetVolMesh::Cell::index_type, int s, int e);
};

DECLARE_MAKER(BuildFEMatrixQuadratic)


  BuildFEMatrixQuadratic::BuildFEMatrixQuadratic(GuiContext *context)
    : Module("BuildFEMatrixQuadratic", context, Source, "Forward", "BioPSE"),
      barrier("BuildFEMatrixQuadratic barrier"),
      BCFlag(context->subVar("BCFlag")),
    UseCondGui(context->subVar("UseCondTCL")),
    refnodeGui(context->subVar("refnodeTCL")),
    bcArray(256, false),
    mutex("mutex")
{
}


BuildFEMatrixQuadratic::~BuildFEMatrixQuadratic()
{
}


void
BuildFEMatrixQuadratic::execute()
{
  ifld_ = (FieldIPort *)get_iport("QuadTetVolField");
  FieldHandle mesh;
  imat_ = (MatrixIPort *)get_iport("RefNode");
  MatrixHandle mat_handle; //refnodeH
  
  ifld_->get(mesh);
  imat_->get(mat_handle);

  if(!mesh.get_rep())
  {
    error("No Data in port 1 field.");
    return;
  }
  else if (mesh->get_type_name(-1) != "QuadraticTetVolField<int> ")
  {
    error("Input must be a TetVol type, not a '"+mesh->get_type_name(-1)+"'.");
    return;
  }

  time_t current_time = time(NULL);
  remark(string("Beginning simulation: ") + ctime(&current_time) + ".");

  if (mesh->generation == gen && gbl_matrixH.get_rep() && rhsH.get_rep() &&
      lastBCFlag == BCFlag.get())
  {
    outmatrix->send(gbl_matrixH);
    rhsoport->send(rhsH);
    return;
  }
  gen=mesh->generation;
  UseCond=UseCondGui.get();

  // keep a handle on the field.
  qtv = dynamic_cast<QuadraticTetVolField<int>*>(mesh.get_rep());
  if (!qtv)
  {
    error("Failed dynamic cast to QuadraticTetVolField<int>*");
    return;
  }
  QuadraticTetVolMeshHandle mesh_handle;
  qtvm_ = qtv->get_typed_mesh();

  QuadraticTetVolMesh::Node::size_type nnodes;
  qtvm_->size(nnodes);

  qtvm_->get_property("dirichlet", dirichlet);

  bcArray.resize(nnodes, false);

  vector<pair<int, double> >::iterator iter = dirichlet.begin();
  while (iter != dirichlet.end())
  {
    bcArray[(*iter).first] = true;
    ++iter;
  }

  rows=scinew int[nnodes+1];  
  np=Thread::numProcessors();
  if (np>10) np = 5;//np/=2;
  colidx.resize(np+1);

  refnode=0;
  DirSub=PinZero=AverageGround=0;
  if (BCFlag.get() == "DirSub")
  {
    DirSub=1;
  }
  else if (BCFlag.get() == "PinZero")
  { 
    PinZero=1; DirSub=1;
    refnodeGui.reset();
    refnode = refnodeGui.get();
  }
  else if (BCFlag.get() == "AverageGround")
  {
    AverageGround=1;
    DirSub=1;
  }
  else
  {
    warning("BCFlag not set '" + BCFlag.get() + "'!");
  }
  lastBCFlag=BCFlag.get();

  MatrixHandle refnodeH;
  if (imat_->get(refnodeH)&&refnodeH.get_rep()&&refnodeH->nrows()>0){
    refnode=(int)(*((ColumnMatrix*)refnodeH.get_rep()))[0];
  }


  if (PinZero)
  {
    remark("Pinning node " + to_string(refnode) + " to zero.");
  }
  if (AverageGround)
  {
    remark("Averaging of all nodes to zero.");
  }

    QuadraticTetVolMesh::Cell::array_type array;
  qtvm_->get_cells(array,(QuadraticTetVolMesh::Node::index_type)0);

  Thread::parallel(Parallel<BuildFEMatrixQuadratic>(this, &BuildFEMatrixQuadratic::parallel), np, true);


  current_time = time(NULL);
  remark(string("End simulation: ") + ctime(&current_time) + ".");

  outmatrix = (MatrixOPort *)get_oport("FEMMatrix");
  rhsoport = (MatrixOPort *)get_oport("RHS");
  gbl_matrixH=MatrixHandle(gbl_matrix);
  outmatrix->send(gbl_matrixH);
  //outmatrix->send(gbl_matrix);
  rhsH=MatrixHandle(rhs);
  rhsoport->send(rhsH);
  //rhsoport->send(rhs);
  //this->mesh=0;
}

void BuildFEMatrixQuadratic::parallel(int proc)
{
  if (proc==0){
    //    qtvm_->compute_edges();
    //qtvm_->compute_nodes();
  }


  QuadraticTetVolMesh::Node::size_type nnodes;
  qtvm_->size(nnodes);
  int start_node=nnodes*proc/np; 
  int end_node=nnodes*(proc+1)/np; 
  int ndof=end_node-start_node;


  int r=start_node;
  int i;
  
  QuadraticTetVolMesh::Node::array_type mycols(0,15*ndof);


  for(i=start_node;i<end_node;i++){
    rows[r++]=mycols.size();
    if((bcArray[i] && DirSub) || (i==refnode && PinZero)) {
      mycols.push_back(i); // Just a diagonal term
    } else if (i==refnode && AverageGround) { // 1's all across
      for (int ii=0; ii<nnodes; ii++) 
	mycols.push_back(ii);
    } /*else if (qtvm_->nodes[i]->pdBC && DirSub) {
      int nd=qtvm_->nodes[i]->pdBC->diffNode;
      if (nd > i) {
	mycols.add(i);
	mycols.add(nd);
      } else {
	mycols.add(nd);
	mycols.add(i);
	}
	}*/ 
    else {
      //      mutex.lock();      
      qtvm_->add_node_neighbors(mycols, i, bcArray, DirSub);
      // mutex.unlock();
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

  QuadraticTetVolMesh::Cell::size_type nelems;
  qtvm_->size(nelems);

  TetVolMesh::Cell::iterator ii, iie;

  TetVolMesh::Node::array_type cell_nodes(10);

  qtvm_->begin(ii); qtvm_->end(iie);
  for (; ii != iie; ++ii){
    if (qtvm_->test_nodes_range(*ii, start_node, end_node)){ 
      build_local_matrix(lcl_matrix,*ii);   
      add_lcl_gbl(*gbl_matrix,lcl_matrix,*rhs,*ii,start_node, end_node);
    }
  }
      /*  for (i=0; i<nelems; i++){
        Element e=qtvm_->elems[i];
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
      build_local_matrix(qtvm_->elems[i],lcl_matrix,i);
      add_lcl_gbl(*gbl_matrix,lcl_matrix,*rhs,i,start_node, end_node);
      //	    mutex.unlock();
      }
      }*/
  for(i=start_node;i<end_node;i++){
    if((bcArray[i] && DirSub) || (PinZero && i==refnode)){
      // This is just a dummy entry...
      //	    (*gbl_matrix)[i][i]=1;
      int id=rows[i];
      a[id]=1;
      if (i==refnode && PinZero)
	(*rhs)[i]=PINVAL;
      else
	(*rhs)[i]=dirichlet[i].second; //?? dirichlet[i]??
    } /*else if (qtvm_->nodes[i]->pdBC && DirSub) {
      int nd=qtvm_->nodes[i]->pdBC->diffNode;
      int id=rows[i];
      if (nd > i) {
	a[id]=1;
	a[id+1]=-1;
      } else {
	a[id]=-1;
	a[id+1]=1;
      }
      (*rhs)[i]=qtvm_->nodes[i]->pdBC->diffVal;
      } */else if (AverageGround && i==refnode) {
      for (int ii=rows[i]; ii<rows[i]+nnodes; ii++)
	a[ii]=1;
    }
  }
}

void BuildFEMatrixQuadratic::build_local_matrix(double lcl_a[10][10], 
					 TetVolMesh::Cell::index_type c_ind)
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

    jac_el = qtvm_->get_gradient_basis(c_ind,l,pt,grad1,grad2,grad3,grad4,
				       grad5,grad6,grad7,grad8,grad9,grad10);

    Vector l1,l2,l3,l4; 
    vol = ((TetVolMesh*)qtvm_.get_rep())->get_gradient_basis(c_ind,l1,l2,l3,l4);  //get volume by using linear
    if(vol < 1.e-10){
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

  vector<pair<string, Tensor> > tens;
  if (! qtv->get_property("conductivity_table", tens)) {
    remark("Using identity conductivity tensors.");
    pair<int,int> minmax;
    minmax.second=1;
    field_minmax(*qtv, minmax);
    tens.resize(minmax.second+1);
    vector<double> t(6);
    t[0] = t[3] = t[5] = 1;
    t[1] = t[2] = t[4] = 0;
    Tensor ten(t);
    for (unsigned int i = 0; i < tens.size(); i++) {
      tens[i] = pair<string, Tensor>(to_string((int)i), ten);
    }
  }
  int  ind = qtv->value(c_ind);

  if (UseCond) {
    el_cond[0][0] = tens[ind].second.mat_[0][0];
    el_cond[0][1] = tens[ind].second.mat_[0][1];
    el_cond[1][0] = tens[ind].second.mat_[1][0];
    el_cond[0][2] = tens[ind].second.mat_[0][2];
    el_cond[2][0] = tens[ind].second.mat_[2][0];
    el_cond[1][1] = tens[ind].second.mat_[1][1];
    el_cond[1][2] = tens[ind].second.mat_[1][2];
    el_cond[2][1] = tens[ind].second.mat_[2][1];
    el_cond[2][2] = tens[ind].second.mat_[2][2];
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
    }
  }
}


void
BuildFEMatrixQuadratic::add_lcl_gbl(Matrix& gbl_a, double lcl_a[10][10],
				    ColumnMatrix& rhs,
				    TetVolMesh::Cell::index_type c_ind,
				    int s, int e)
{

  for (int i=0; i<10; i++) // this four should eventually be a
    // variable ascociated with each element that indicates 
    // how many nodes are on that element. it will change with 
    // higher order elements
  {	  
    TetVolMesh::Node::array_type cell_nodes(10);
    qtvm_->get_nodes(cell_nodes, c_ind);

    int ii = cell_nodes[i];

    if(ii >= s && ii < e){
      if (!((bcArray[ii] && DirSub) || (ii==refnode && PinZero) ||
	    (ii==refnode && AverageGround))) {
	for (int j=0; j<10; j++) {

	  int jj = cell_nodes[j];

	  if (bcArray[jj] && DirSub){
	    rhs[ii] -= dirichlet[c_ind].second*lcl_a[i][j];
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


} // End namespace BioPSE


