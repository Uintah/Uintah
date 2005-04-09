#include"sparseMatrix.h"
#include<fstream.h>
#include <stdio.h>

SparseMatrix::SparseMatrix(Array1<Node*> &nodeList, Array1<Element*> &elementList, Property property) : nodeList(nodeList) {
  int row,col;
  double mass, stiff;
  double bcval;
  int rdof;
  double ev[8];
  
  numberOfRows = nodeList.size();
  matrix = new Array1<Block*>[numberOfRows];
  rhs = new double[numberOfRows];
  numberRDOF = numberOfRows - Node::numberOfBCS;
  rbcv = new double[numberRDOF];
  rrhs = new double[numberRDOF];

  theta = property.getTheta();
  ss = property.getSS();
  symmetric = property.getSymmetric();
  tsuba = property.getTsubA();
  period = property.getPeriod();
  if (period < 1) period = 1;
  timeSteps = property.getTimeSteps();
  deltaT = property.getDeltaT();

  // clear rhs vectors

  for (int i=0;i<numberOfRows;i++) {
    rhs[i]=0;
    if (i<numberRDOF) {
      rbcv[i]=0;
      rrhs[i]=0;
    }
  }
  
  // assemble elements into matrix

  cout << "degrees of freedom:         " << numberOfRows << endl;
  cout << "reduced degrees of freedom: " << numberRDOF << endl;
  for(int elem=0;elem<elementList.size();elem++) { // loop through elements
    for(int k=0;k<8;k++) { ev[k] = 0; } // clear the ev vector
    for(int i=0;i<8;i++) { // loop through rows in each element
      row = elementList[elem]->getNode(i)->getNum();
      for(int j=0;j<8;j++) { // loop through columns for each element
	col = elementList[elem]->getNode(j)->getNum();
	mass = elementList[elem]->getMassStiff(i,j);
	stiff = elementList[elem]->getThermStiff(i,j);
	rdof = elementList[elem]->getNode(i)->getRdof();
	bcval= elementList[elem]->getNode(i)->getBoundVal();
	this->addIndex(row, col);
	this->addMass(row, col, mass);
	this->addStiff(row, col, stiff);
	// now make the rbcv vector
	if ( rdof < 0){
	  if (ss) {
	    ev[j]=ev[j]-stiff*bcval;
	  } else {
	    ev[j]=ev[j]-(mass/deltaT + theta*stiff)*bcval;
	  }
	}
      } // end of loop through columns
    } // end of loop though rows
    for(int m=0;m<8;m++) {
      rdof = elementList[elem]->getNode(m)->getRdof();
      if (rdof > 0 ) rbcv[rdof] += ev[m];
    }
  } // end of loop through elements

} // end of constructor

// destructor
SparseMatrix::~SparseMatrix()
{
    delete[] matrix;
}


void SparseMatrix::addIndex(int row, int column) {
  
  int indexExists = 0;
  int i = 0;

  if (row < numberOfRows) {
    while (!indexExists && i < matrix[row].size() ) {
      if (matrix[row][i++]->getIndex() == column) {
	indexExists = 1;
      }
    }
    if (!indexExists) {
      matrix[row].add(new Block(column,(double)0,(double)0));
    }
  } else {
    cout << "row out of bounds: " << row << endl;
  }
}
  
int SparseMatrix::findSparseColumn(int row, int column) {
  
  for (int i=0;i<matrix[row].size();i++) {
    if (matrix[row][i]->getIndex() == column) {
      return i;
    }
  }
  return -1;
}
  
void SparseMatrix::addMass(int row, int column, double mass) {
  
  int sparseColumn = this->findSparseColumn(row,column);
  if (sparseColumn < 0) {
    cout << "error in adding mass, invalid column index" << endl;
  } else {
  matrix[row][sparseColumn]->addMass(mass);
  }
  
}

void SparseMatrix::addStiff(int row, int column, double stiff) {
  
  int sparseColumn = this->findSparseColumn(row,column);
  if (sparseColumn < 0 ) {
    cout << "error in adding stiff, invalid column index" << endl;
  } else {
    matrix[row][sparseColumn]->addStiff(stiff);
  }
  
}

double SparseMatrix::getA(int row, int column) { 
  
  int sparseColumn = this->findSparseColumn(row,column);
  double stiff = matrix[row][sparseColumn]->getStiff();
  double mass = matrix[row][sparseColumn]->getMass();
  if (ss) {
    return stiff;
  } else {
    return mass/deltaT + theta*stiff;
  }

}

double SparseMatrix::getAsparse(int row, int column) { 
  
  int sparseColumn = column;
  double stiff = matrix[row][sparseColumn]->getStiff();
  double mass = matrix[row][sparseColumn]->getMass();
  if (ss) {
    return stiff;
  } else {
    return mass/deltaT + theta*stiff;
  }

}

void SparseMatrix::writeMatrix(int val) {
  ofstream fout;
  fout.open("A.dat");
  for (int i=0;i<numberOfRows;i++) {
    int rgrow = nodeList[i]->getRdof();
    if (rgrow > -1) { 
      for (int j=0;j<matrix[i].size();j++) {
	int gcol = matrix[i][j]->getIndex();
	int rgcol = nodeList[gcol]->getRdof(); 
	if (rgcol > -1) {
	  switch (val) {
	  case 1:
	    fout << rgrow << " " << rgcol << " " << getAsparse(i,j) << endl;
	  default:
	    fout << rgrow << " " << rgcol << " " << matrix[i][j]->getStiff() << endl;
	  }
	}
      }
    }
  }
}

void SparseMatrix::printMatrix(int val) {
  for (int i=0;i<numberOfRows;i++) {
    for (int j=0;j<matrix[i].size();j++) {
      switch (val) {
      case 1:
	cout << matrix[i][j]->getIndex() << " ";
	break;
      case 2:
	cout << matrix[i][j]->getMass() << " ";
	break;
      case 3:
	cout << matrix[i][j]->getStiff() << " ";
	break;
      default:
	cout << matrix[i][j]->getIndex() << " ";
      }

	
    }
    cout << endl;
  }
}

void SparseMatrix::makeRhs(Array1<Node*> &nodeList, int iteration_total) {

  int gcol,dummy;
  double mass, stiff, sarc, temp, sarcPast;
  char sarcfile[30];
  ifstream fin;
  
  int i;
  for (i=0;i<numberOfRows;i++) {
    rhs[i]=0;
  }

  if (ss) {
    deltaT = 1;
    theta = 0.5;
  } else {
    sprintf(sarcfile,"SARCF.%d",(iteration_total % period)+1);
    fin.open(sarcfile);
    if (!fin) cout << "using default sarc file" << endl;
    else cout << "using " << sarcfile << " for power " << endl;
  }

  for(i=0;i<numberOfRows;i++) {
    for(int j=0;j<matrix[i].size();j++) {
      gcol = matrix[i][j]->getIndex();
      mass = matrix[i][j]->getMass();
      stiff = matrix[i][j]->getStiff();
      sarcPast = nodeList[gcol]->getSarc();
      if (ss) {
	sarc = sarcPast;
      } else {
	if (!fin) {
	  sarc = sarcPast;
	} else fin >> dummy >> sarc;
      }
      temp = nodeList[gcol]->getTemp();
      if (gcol > -1) {
	rhs[i] = rhs[i] + mass*(theta*sarc + (1-theta)*sarcPast);
	if (!ss) { // transient
	  rhs[i] = rhs[i] + (mass/deltaT - (1-theta)*stiff)*temp;
	}
      }
    }
  }

} // end of method makeRhs

void SparseMatrix::direchlet(Array1<Node*> &nodeList) {

  for (int i=0;i<numberOfRows;i++) {
    int rdof = nodeList[i]->getRdof();
    if (rdof > -1 ) {
      rrhs[rdof] = rhs[i] + rbcv[rdof];
    }
  }

} // end of method direchlet


void SparseMatrix::mult(const double* x, double* b) {
  // send in Vector x, return Vector b
  // for linear System Ax=b

  int i;
  //matrix vector multiplication
  for(i=0;i<numberOfRows;i++){
    int rgrow = nodeList[i]->getRdof();
    if (rgrow > -1) {
      b[rgrow] = 0; //  clear out the b Vector
      for(int j=0;j<matrix[i].size();j++){
	int gcol = matrix[i][j]->getIndex();
	int rgcol = nodeList[gcol]->getRdof();
	if (rgcol > -1) {
	  b[rgrow] += getAsparse(i,j)*x[rgcol];
	}
      }
    }
  }
  //End of multiplication
  
} // end of method mult


void SparseMatrix::multTran(const double* x, double* b) {
  // send in Vector x, return Vector b
  // for linear System A'x=b   (A' = transpose of A)

  int i;

  // clear out b vector first
  for (i=0;i<numberRDOF;i++) {
    b[i] = 0;
  }
  //matrix vector multiplication
  for(i=0;i<numberOfRows;i++){
    int rgrow = nodeList[i]->getRdof();
    if (rgrow > -1) {
      for(int j=0;j<matrix[i].size();j++){
	int gcol = matrix[i][j]->getIndex();
	int rgcol = nodeList[gcol]->getRdof();
	if (rgcol > -1) {
	  b[rgcol] += getAsparse(i,j)*x[rgrow];
	}
      }
    }
  }
  //End of multiplication
  
} // end of method multTran

int SparseMatrix::solveQMR(int maxit, double eps) {

//Quasi-Minimal Residual Method with Jacobi Preconditioner subroutine
//This solves for a full matrix with all values in it
//   Ax=b
//A is nonsymmetric
// ************  INPUT   *************
// x is the inital guess
// A is the matrix 
// rrhs is the right hand side
// n is the size of the matrix [nxn]
// maxit is the maximum number of iterations that will be performed
// eps is a measure of the error 1<eps< (machine precision)

// stopping criteria is either maxit or (<r,r>/<b,b>)^.5 < eps

  double rho,rho_minus_1,eta,delta,eps1=0,beta,k,rr_inprod_rr,calcerr;
  double N,mbets,qtheta=0,qtheta_minus_1,gamma,gamma_minus_1,b_inprod_b;
  double calcerr_last = 0;

  int stall = 1;
  N=-1;
  gamma=1;
  
  int n = numberRDOF;
  double *x=new double[n];
  double *d=new double[n];
  double *s=new double[n];
  double *r=new double[n];
  double *z=new double[n];
  double *v=new double[n];
  double *w=new double[n];
  double *y=new double[n];
  double *p=new double[n];
  double *q=new double[n];
  double *xt=new double[n];
  double *vt=new double[n];
  double *wt=new double[n];
  double *zt=new double[n];
  double *pt=new double[n];
  double *yt=new double[n];
  double *M_inverse=new double[n];
  

  //solves for the initial residual vector
  
  // put initial guess in x
  int i;
  for (i=0;i<numberOfRows;i++) {
    int rdof = nodeList[i]->getRdof();
    if (rdof > -1 ) {
      x[rdof] = nodeList[i]->getTemp();
    }
  }
  
  //Matrix multiplication
  b_inprod_b=0;
  rho=0;
  eta=0;
  for(i=0;i<numberOfRows;i++){
    k=0;
    int rgrow = nodeList[i]->getRdof();
    if (rgrow > -1) {
      M_inverse[rgrow]=(double).5/getA(i,i);
      for(int j=0;j<matrix[i].size();j++){
	int gcol = matrix[i][j]->getIndex();
	int rgcol = nodeList[gcol]->getRdof(); 
	if (rgcol > -1) {
	  k=k+getAsparse(i,j)*x[rgcol];
	}
      }
      b_inprod_b=b_inprod_b+rrhs[rgrow]*rrhs[rgrow];
      r[rgrow]=rrhs[rgrow]-k;
      vt[rgrow]=r[rgrow];
      y[rgrow]=vt[rgrow]*M_inverse[rgrow];   
      rho=rho+y[rgrow]*y[rgrow];
      wt[rgrow]=r[rgrow];
      z[rgrow]=wt[rgrow]*2;
      eta=eta+z[rgrow]*z[rgrow];
    }
  }
  //End of Matrix multiplication
  
  rho= sqrt(rho);
  eta= sqrt(eta);
  
    //Prints out header to the user
  cout << "\n\tQuasi-Minimal Residual Method with Jacobi Precond." << endl;
  
  cout << "\niteration\tcalculated error\tstop criteria\t\tmax iteration\n" << endl;
  
  //start of the iterations of QMR method
  
  for(int kk=1;kk<maxit;kk++){
    if ((rho==0) || (eta==0)) {
      cout << "method failed 1" << endl;
      return(0);
    }
    
    //vector scalar multiplication
    
    delta=0;
    for(i=0;i<(n);i++){
      v[i]=vt[i]/rho;
      y[i]=y[i]/rho;
      w[i]=wt[i]/eta;
      z[i]=z[i]/eta;
      yt[i]=y[i]*2;
      zt[i]=z[i]*M_inverse[i];
      delta=delta+z[i]*y[i];  
    }
    //End of Multiplication
    
    if (delta==0){
      cout << "method fails 2" << endl;
      return(0);
    }
    if (kk==1){
      for(i=0;i<(n);i++){
	p[i]=yt[i];
	q[i]=zt[i];
      }
    } else{
      for(i=0;i<(n);i++){
	p[i]=yt[i]-eta*delta/eps1*p[i];
	q[i]=zt[i]-rho*delta/eps1*q[i];
      }
    }	
    
    //matrix vector multiplication
    eps1=0;
    rr_inprod_rr=0;
    mult(p, pt);
    mult(x, xt);
    for(i=0;i<numberRDOF;i++){
	eps1=eps1+q[i]*pt[i];
	double rri=rrhs[i]-xt[i];
	rr_inprod_rr=rr_inprod_rr+rri*rri;
    }
/*
    for(i=0;i<numberOfRows;i++){
      k=0;
      ks=0;
      int rgrow = nodeList[i]->getRdof();
      if (rgrow > -1) {
	for(int j=0;j<matrix[i].size();j++){
	  int gcol = matrix[i][j]->getIndex();
	  int rgcol = nodeList[gcol]->getRdof();
	  if (rgcol > -1) {
	      //k=k+getAsparse(i,j)*p[rgcol];
	    ks=ks+getAsparse(i,j)*x[rgcol];
	  }
	}
	rri=rrhs[rgrow]-ks;
	//pt[rgrow]=k;
	eps1=eps1+q[rgrow]*pt[rgrow];
	rr_inprod_rr=rr_inprod_rr+rri*rri;
      }
    }
*/

    //End of multiplication
    
    //outputs iteration, calculated error, stopcriteria(eps), maxit
    calcerr= sqrt(rr_inprod_rr)/sqrt(b_inprod_b);
    cout << kk << "\t\t" << calcerr << "\t\t" << eps << "\t\t\t" << maxit << endl;

    if (fabs(calcerr - calcerr_last) < eps*eps) stall ++;
    calcerr_last = calcerr;

    if (stall > 2) {
      cout << " \n Method has stalled, Trying Solution again... " << endl;
    }
    if (calcerr<eps || stall > 2){
      for (i=0;i<numberOfRows;i++) {
	int rgcol = nodeList[i]->getRdof();
	if (rgcol > -1) {
	  nodeList[i]->setTemp(x[rgcol]);
	}
      }
      if (stall > 2) {
	return(3);
      } else {
	return(0);
      }
    }

    
    beta=eps1/delta;
    if ((eps1==0) || (beta==0)){
      cout << "method fails 3 4" << endl;
      return(0);
    }
    
    for(i=0;i<(n);i++){
      vt[i]=pt[i]-beta*v[i];
    }
    
    //Matrix Vector multipliation
    eta=0;
    rho_minus_1=rho;
    rho=0;
    for(i=0;i<numberOfRows;i++){
      k=0;
      int rgrow = nodeList[i]->getRdof(); 
      if (rgrow > -1 ) {
	for(int j=0;j<matrix[i].size();j++) {
	  int gcol = matrix[i][j]->getIndex();
	  int rgcol = nodeList[gcol]->getRdof();
	  if (rgcol > -1) {
	    k=k+getA(gcol,i)*q[rgcol];
	  }
	}
	wt[rgrow]=k-beta*w[rgrow];
	y[rgrow]=vt[rgrow]*M_inverse[rgrow];
	z[rgrow]=wt[rgrow]*2;
	rho=rho+y[rgrow]*y[rgrow];
	eta=eta+z[rgrow]*z[rgrow];
      }
    }
    //End of Multiplication
    
    rho= sqrt(rho);
    eta= sqrt(eta);
    mbets=beta;
    if (beta<0){
      mbets=-beta;
    }
    
    qtheta_minus_1=qtheta;
    qtheta=rho/(gamma*mbets);
    gamma_minus_1=gamma;
    gamma=1/sqrt((1+qtheta*qtheta));
    N=-N*rho_minus_1*gamma*gamma/(gamma_minus_1*gamma_minus_1*beta);
    
    if (gamma==0){
      cout << "method fails 5" << endl;
      return(0);
    }
    
    k=0;
    
    if (kk==1){
      for(i=0;i<(n);i++){
	d[i]=N*p[i];
	s[i]=N*pt[i];
	x[i]=x[i]+d[i];
	r[i]=r[i]-s[i];
	k=k+r[i]*r[i];
      }
    } else{
      for(i=0;i<(n);i++){
	d[i]=N*p[i]+gamma*gamma*qtheta_minus_1*qtheta_minus_1*d[i];
	s[i]=N*pt[i]+gamma*gamma*qtheta_minus_1*qtheta_minus_1*s[i];
	x[i]=x[i]+d[i];
	r[i]=r[i]-s[i];
	k=k+r[i]*r[i];
      }
    }
  }

  for (i=0;i<numberOfRows;i++) {
    int rgcol = nodeList[i]->getRdof();
    if (rgcol > -1) {
      nodeList[i]->setTemp(x[rgcol]);
    }
  }
  
  delete x;
  delete d;
  delete s;
  delete r;
  delete z;
  delete v;
  delete w;
  delete y;
  delete p;
  delete q;
  delete vt;
  delete wt;
  delete zt;
  delete pt;
  delete yt;
  delete M_inverse;
    return(1);

} // end of method solveQMR


//Conjugate Gradient with Jacobi Preconditioner subroutine
//This solves for a full matrix with all values in it
//   Ax=b
//A is symmetric positive definate
//************  INPUT   *************
// x is the inital guess
// A is the matrix 
// rrhs is the right hand side
// n is the size of the matrix [nxn]
// maxit is the maximum number of iterations that will be performed
// eps is a measure of the error 1<eps< (machine precision)

// stopping criteria is either maxit or (<r,r>/<b,b>)^.5 < eps

int SparseMatrix::solveCG(int maxit,double eps) {

  int n = numberRDOF;
  double *x=new double[n];
  double *r=new double[n];
  double *z=new double[n];
  double *p=new double[n];
  double *q=new double[n];
  double *M_inverse=new double[n];
  
  double ks,p_inprod_q,alpha,rr_inprod_rr,rri;
  double r_inprod_z,rho,rhoold,beta,kks,calcerr,b_inprod_b=0;

  //solves for the inital residual vector (r=b-Ax)

  // put initial guess in x
  int i;
  for (i=0;i<numberOfRows;i++) {
    int rdof = nodeList[i]->getRdof();
    if (rdof > -1 ) {
      x[rdof] = nodeList[i]->getTemp();
    }
  }

  // Matrix vector multiplication
  for(i=0;i<numberOfRows;i++) {
    int rgrow = nodeList[i]->getRdof();
    if (rgrow > -1) {
      M_inverse[rgrow]=1/getA(i,i);
      ks=0;      
      for(int j=0;j<matrix[i].size();j++){
	int gcol = matrix[i][j]->getIndex();
	int rgcol = nodeList[gcol]->getRdof();
	if (rgcol > -1) {
	  ks=ks+getAsparse(i,j)*x[rgcol];
	}
      }
      r[rgrow]=rrhs[rgrow]-ks;
      b_inprod_b=b_inprod_b+rrhs[rgrow]*rrhs[rgrow];;
    }
  }
  // End of Matrix vector multiplication

  //Prints out header to the user
  cout<<"\n\n\t\tConjugate Gradient Method with Jacobi Preconditioner";

  cout<<"\n\niteration\tcalculated error\tstop criteria\t\tmax iteration\n";

    
  //start of the iterations of conjugate gradient method

  for(int kk=1;kk<maxit;kk++){
    r_inprod_z=0;
    if (kk==1){
      for(i=0;i<numberOfRows;i++){
	int rgrow = nodeList[i]->getRdof();
	if (rgrow > -1) {
	  z[rgrow]=r[rgrow]*M_inverse[rgrow];
	  r_inprod_z=r_inprod_z+r[rgrow]*z[rgrow];
	  p[rgrow]=z[rgrow];
	}
      }
    } else{
      for(i=0;i<numberOfRows;i++){
	int rgrow = nodeList[i]->getRdof();
	if (rgrow > -1 ) {
	  z[rgrow]=r[rgrow]*M_inverse[rgrow];
	  r_inprod_z=r_inprod_z+r[rgrow]*z[rgrow];
	}
      }
    }
    rho=r_inprod_z;
    
    
    //Matrix multiplation q=Ap
    
    p_inprod_q=0;rr_inprod_rr=0;
    for(i=0;i<numberOfRows;i++){
      ks=0;kks=0;
      int rgrow = nodeList[i]->getRdof();
      if (rgrow > -1) {
	for(int j=0;j<matrix[i].size();j++){
	  int gcol = matrix[i][j]->getIndex();
	  int rgcol = nodeList[gcol]->getRdof();
	  if (rgcol > -1) {
	    if(i==0 && kk>1){
	      beta=rho/rhoold;
	      p[rgcol]=z[rgcol]+beta*p[rgcol];
	    }
	    ks=ks+getAsparse(i,j)*p[rgcol];
	    kks=kks+getAsparse(i,j)*x[rgcol];
	  }
	}
	q[rgrow]=ks;
	rri=rrhs[rgrow]-kks;
	p_inprod_q=p_inprod_q+p[rgrow]*q[rgrow];
	rr_inprod_rr=rr_inprod_rr+rri*rri;
      }
    }
    //End of Matrix multiplictaion
    
    //outputs iteration, calculated error, stopcriteria(eps), maxit
    calcerr=sqrt(rr_inprod_rr)/sqrt(b_inprod_b);
    cout<<"\n"<<kk<<"\t\t";
    cout.width(15);
    cout<<calcerr<<"\t\t";
    cout.width(12);
    cout<<eps<<"\t\t\t";
    cout<<maxit;
    
    if (calcerr<eps){
      for (i=0;i<numberOfRows;i++) {
	int rgcol = nodeList[i]->getRdof();
	if (rgcol > -1) {
	  nodeList[i]->setTemp(x[rgcol]);
	}
      }
      return(0);
    }
    
    //Update x and r
    
    alpha=rho/p_inprod_q;
    rhoold=rho;
    for(i=0;i<numberOfRows;i++){
      int rgrow = nodeList[i]->getRdof();
      if (rgrow > -1 ) {
	x[rgrow]=x[rgrow]+alpha*p[rgrow];
	r[rgrow]=r[rgrow]-alpha*q[rgrow];
      }
    }
  }
  return 0;
}
//end of conjugate gradient method

















