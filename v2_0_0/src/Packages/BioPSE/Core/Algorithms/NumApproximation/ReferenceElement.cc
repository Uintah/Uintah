/*
 * file:     ReferenceElement.cc
 * @version: 1.0
 * @author:  Sascha Moehrs
 * email:    sascha@sci.utah.edu
 * date:     February 2003
 *
 * to do:    -> replace mapping functions such that 'distorted' cubic
 *              elements can be used
 *
 *           -> documentation
 *
 */

#include <math.h>
#include <stdlib.h>
#include <Packages/BioPSE/Core/Algorithms/NumApproximation/ReferenceElement.h>
#include <stdio.h>

namespace BioPSE {

using namespace SCIRun;

bool ReferenceElement::initialize() {

  int i;
  // memory for the quadratue points
  qP = new double*[numQuadPoints];    // double*
  if(qP == NULL)
	return false;
  for(i = 0; i < numQuadPoints; i++) {
	qP[i] = new double[3]; /* x,y,z */
	if(qP[i] == NULL)
	  return false;
  }
  // memory for the quadrate weights
  qW = new double[numQuadPoints];
  if(qW == NULL)
	return false;
  // memory for the nodes
  node = new double*[numNodes];
  if(node == NULL)
	return false;
  for(i = 0; i < numNodes; i++) {
	node[i] = new double[3]; 
	if(node[i] == NULL)
	  return false;
  }
  // set quadrature points
  /* x */                       /* y */                       /* z */
  qP[0][0] = -1.0; qP[0][1] = -1.0; qP[0][2] = -1.0; qW[0] =  1.0/3.0;
  qP[1][0] =  1.0; qP[1][1] = -1.0; qP[1][2] = -1.0; qW[1] =  1.0/3.0;
  qP[2][0] =  1.0; qP[2][1] =  1.0; qP[2][2] = -1.0; qW[2] =  1.0/3.0;
  qP[3][0] = -1.0; qP[3][1] =  1.0; qP[3][2] = -1.0; qW[3] =  1.0/3.0;
  qP[4][0] = -1.0; qP[4][1] = -1.0; qP[4][2] =  1.0; qW[4] =  1.0/3.0;
  qP[5][0] =  1.0; qP[5][1] = -1.0; qP[5][2] =  1.0; qW[5] =  1.0/3.0;
  qP[6][0] =  1.0; qP[6][1] =  1.0; qP[6][2] =  1.0; qW[6] =  1.0/3.0;
  qP[7][0] = -1.0; qP[7][1] =  1.0; qP[7][2] =  1.0; qW[7] =  1.0/3.0;
  qP[8][0] =  0.0; qP[8][1] =  0.0; qP[8][2] =  0.0; qW[8] = 16.0/3.0;
  // set nodes
  /* x */            /* y */            /* z */
  node[0][0] = -1.0; node[0][1] = -1.0; node[0][2] = -1.0;
  node[1][0] =  1.0; node[1][1] = -1.0; node[1][2] = -1.0;
  node[2][0] =  1.0; node[2][1] =  1.0; node[2][2] = -1.0;
  node[3][0] = -1.0; node[3][1] =  1.0; node[3][2] = -1.0;
  node[4][0] = -1.0; node[4][1] = -1.0; node[4][2] =  1.0;
  node[5][0] =  1.0; node[5][1] = -1.0; node[5][2] =  1.0;
  node[6][0] =  1.0; node[6][1] =  1.0; node[6][2] =  1.0;
  node[7][0] = -1.0; node[7][1] =  1.0; node[7][2] =  1.0;

  return true;
  
}

bool ReferenceElement::destroy() {
  int i;
  // free memory of quadrature points
  for(i = 0; i < numQuadPoints; i++) {
	delete []qP[i];
	qP[i] = NULL;
  }
  delete []qP;
  qP = NULL;
  // free memory of quadrature weights
  delete []qW;
  qW = NULL;
  // free memory of nodes
  for(i = 0; i < numNodes; i++) {
	delete []node[i];
	node[i] = NULL;
  }
  delete []node;
  node = NULL;
  return true;
}

//Constructors
ReferenceElement::ReferenceElement() {
  numQuadPoints = 9;
  numNodes      = 8;
  if(!initialize()) {
	fprintf(stderr, "ReferenceElement::ReferenceElement() -> impossible to build ReferenceElement\n");
  }
}

//Destructor
ReferenceElement::~ReferenceElement() {
  if(!destroy()) {
	fprintf(stderr, "ReferenceElement::~ReferenceElement() -> impossible to cleanup ReferenceElement\n");
  }
}

//Basis functions
double ReferenceElement::phi(int i, double x, double y, double z) {
  switch(i) {
  case 0: return (0.125*(1.0-x)*(1.0-y)*(1.0-z)); // (-1,-1,-1) 1
  case 1: return (0.125*(1.0+x)*(1.0-y)*(1.0-z)); // ( 1,-1,-1) 2
  case 2: return (0.125*(1.0+x)*(1.0+y)*(1.0-z)); // ( 1, 1,-1) 3
  case 3: return (0.125*(1.0-x)*(1.0+y)*(1.0-z)); // (-1, 1,-1) 4
  case 4: return (0.125*(1.0-x)*(1.0-y)*(1.0+z)); // (-1,-1, 1) 5
  case 5: return (0.125*(1.0+x)*(1.0-y)*(1.0+z)); // ( 1,-1, 1) 6
  case 6: return (0.125*(1.0+x)*(1.0+y)*(1.0+z)); // ( 1, 1, 1) 7
  case 7: return (0.125*(1.0-x)*(1.0+y)*(1.0+z)); // (-1, 1, 1) 8
  default: fprintf(stderr, "ReferenceElement::phi() - invalid basis function index\n");
  }
  return 0.0;
}

// gradients of the basis functions
double ReferenceElement::dphidx(int i, double x, double y, double z) {
  switch(i) {
  case 0: return (-0.125*(1.0-y)*(1.0-z)); // (-1,-1,-1) 1
  case 1: return ( 0.125*(1.0-y)*(1.0-z)); // ( 1,-1,-1) 2
  case 2: return ( 0.125*(1.0+y)*(1.0-z)); // ( 1, 1,-1) 3
  case 3: return (-0.125*(1.0+y)*(1.0-z)); // (-1, 1,-1) 4
  case 4: return (-0.125*(1.0-y)*(1.0+z)); // (-1,-1, 1) 5
  case 5: return ( 0.125*(1.0-y)*(1.0+z)); // ( 1,-1, 1) 6
  case 6: return ( 0.125*(1.0+y)*(1.0+z)); // ( 1, 1, 1) 7
  case 7: return (-0.125*(1.0+y)*(1.0+z)); // (-1, 1, 1) 8
  default: fprintf(stderr, "ReferenceElement::dphidx() - invalid basis function index\n");
  }
  return 0.0;
}
double ReferenceElement::dphidy(int i, double x, double y, double z) {
  switch(i) {
  case 0: return (-0.125*(1.0-x)*(1.0-z)); // (-1,-1,-1) 1
  case 1: return (-0.125*(1.0+x)*(1.0-z)); // ( 1,-1,-1) 2
  case 2: return ( 0.125*(1.0+x)*(1.0-z)); // ( 1, 1,-1) 3
  case 3: return ( 0.125*(1.0-x)*(1.0-z)); // (-1, 1,-1) 4
  case 4: return (-0.125*(1.0-x)*(1.0+z)); // (-1,-1, 1) 5
  case 5: return (-0.125*(1.0+x)*(1.0+z)); // ( 1,-1, 1) 6
  case 6: return ( 0.125*(1.0+x)*(1.0+z)); // ( 1, 1, 1) 7
  case 7: return ( 0.125*(1.0-x)*(1.0+z)); // (-1, 1, 1) 8
  default: fprintf(stderr, "ReferenceElement::dphidy() - invalid basis function index\n");
  }
  return 0.0;
}
double ReferenceElement::dphidz(int i, double x, double y, double z) {
  switch(i) {
  case 0: return (-0.125*(1.0-x)*(1.0-y)); // (-1,-1,-1) 1
  case 1: return (-0.125*(1.0+x)*(1.0-y)); // ( 1,-1,-1) 2
  case 2: return (-0.125*(1.0+x)*(1.0+y)); // ( 1, 1,-1) 3
  case 3: return (-0.125*(1.0-x)*(1.0+y)); // (-1, 1,-1) 4
  case 4: return ( 0.125*(1.0-x)*(1.0-y)); // (-1,-1, 1) 5
  case 5: return ( 0.125*(1.0+x)*(1.0-y)); // ( 1,-1, 1) 6
  case 6: return ( 0.125*(1.0+x)*(1.0+y)); // ( 1, 1, 1) 7
  case 7: return ( 0.125*(1.0-x)*(1.0+y)); // (-1, 1, 1) 8
  default: fprintf(stderr, "ReferenceElement::dphidz() - invalid basis function index\n");
  }
  return 0.0;
}

// mapping functions - from reference to actual element
double ReferenceElement::psi1(double x_ref, double xa, double xb) { // transformation of x 
  return 0.5*(xa+xb - x_ref*(xa-xb));
}
double ReferenceElement::psi2(double y_ref, double ya, double yb) { // transformation of y
  return 0.5*(ya+yb - y_ref*(ya-yb));
}  
double ReferenceElement::psi3(double z_ref, double za, double zb) { // transformation of z
  return 0.5*(za+zb - z_ref*(za-zb));
} 

// mapping functions from actual to reference element
double ReferenceElement::isp1(double x, double xa, double xb) {
  return ((xa+xb-2.0*x)/(xa-xb));
}
double ReferenceElement::isp2(double y, double ya, double yb) {
  return ((ya+yb-2.0*y)/(ya-yb));
}
double ReferenceElement::isp3(double z, double za, double zb) {
  return ((za+zb-2.0*z)/(za-zb));
}

// derivatives of mapping functions
double ReferenceElement::dpsi1dx(double xa, double xb) {
  return -0.5*(xa-xb);
}
double ReferenceElement::dpsi2dy(double ya, double yb) {
  return -0.5*(ya-yb);
}
double ReferenceElement::dpsi3dz(double za, double zb) {
  return -0.5*(za-zb);
}

// absolute value of the determinant of the jacobian
double ReferenceElement::getAbsDetJacobian(double xa, double xb, double ya, double yb, double za, double zb) {
  return fabs(-0.125*(xa-xb)*(ya-yb)*(za-zb));
}

} // end of namespace BioPSE
