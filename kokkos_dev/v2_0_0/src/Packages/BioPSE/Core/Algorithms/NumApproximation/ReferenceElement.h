/*
 * file:     ReferenceElement.h
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

#ifndef REFERENCE_ELEMENT_H
#define RERERENCE_ELEMENT_H

#include <Core/Datatypes/Matrix.h>

namespace BioPSE {

using namespace SCIRun;

class ReferenceElement {

public:

  // constructors
  ReferenceElement();

  // destructor
 ~ReferenceElement();

  // basis function
 double phi(int i, double x, double y, double z);

  // gradients of the basis functions
 double dphidx(int i, double x, double y, double z);
 double dphidy(int i, double x, double y, double z);
 double dphidz(int i, double x, double y, double z);

  // mapping functions
  // from reference to actual element
 double psi1(double x, double x1, double x2);
 double psi2(double y, double y1, double y2);  
 double psi3(double z, double z1, double z2); 
  // from actual to reference element
 double isp1(double x, double xa, double xb);
 double isp2(double y, double ya, double yb);
 double isp3(double z, double za, double zb);
  
  // derivatives of the mapping functions
 double dpsi1dx(double x1, double x2);
 double dpsi2dy(double y1, double y2);
 double dpsi3dz(double z1, double z2);

  // return absolute value of the determinant of the Jacobian Matrix 
 double getAbsDetJacobian(double xa, double xb, double ya, double yb, double za, double zb);

  // nodes of the reference element
  double **node;

  // quadrature points within the reference element
  double **qP;

  // quadrature weights within the reference element
  double *qW;

  // total number of quadrature points
  int numQuadPoints;

  // total number of nodes
  int numNodes;

 private:
  
  bool initialize();
  bool destroy();

};

} // end namespace BioPSE

#endif

// ~sam //
