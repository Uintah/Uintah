/*
 * file:     SphericalVolumeConductor.cc 
 * @version: 1.0 
 * @author:  Sascha Moehrs 
 * email:    sascha@sci.utah.edu 
 * date:     January 2003 
 * purpose:  computation of the potential on the surface of a four layer 
 *           spherical volume conductor 
 * 
 * to do:    -> correct handling of multiple dipoles
 *           -> review: conversion of dipole positions
 *           -> documentation
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <Packages/BioPSE/Core/Algorithms/Forward/SphericalVolumeConductor.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>

namespace BioPSE {

using namespace SCIRun;

/**
 * SphericalVolumeConductor::SphericalVolumeConductor(...)
 * Constructor
 *
 * @param  dipoles  reference to the dipole; assumed format of the matrix for dipole i:
 *                  dipoles[i][0] => x coordinate; dipoles[i][1] => y coordinate; dipoles[i][2] => z coordinate; 
 *                  dipoles[i][3] => x moment;     dipoles[i][4] => y moment;     dipoles[i][5] => z moment;
 * @param  elecCart  reference to the cartesian coordinates of the electrodes; the electrodes are projected on the 
 *                  outer sphere surface; assumed format of the matrix for electrode j:
 *                  elecCart[j][0] => x coordinate; elecCart[j][1] => y coordinate; elecCart[j][2] => z coordinate
 * @param  radii  radii of the four layers, i.e.: radii[0] => scalp radius; radii[1] => skull radius; 
 *                radii[2] => cbsf radius; radii[3] => brain radius;
 * @param  radCond  radial conductivities of the four layers;
 * @param  tanCond  tangential conductivities of the four layers;
 * @param  result   reference to the result matrix where the potential values will be stored (must have the same size
 *                  as the matrix elecCart);
 * @param  numDipoles  total number of dipoles (so far only one dipole is considered!);
 * @param  numElectrodes  total number of electrodes, where the potential has to be computed;
 * @param  acc  required accuracy of the series expansion;
 */
SphericalVolumeConductor::SphericalVolumeConductor(DenseMatrix &dipoles, DenseMatrix &elecCart, ColumnMatrix const& radii,
												   ColumnMatrix const& radCond, ColumnMatrix const& tanCond, ColumnMatrix &result,
												   int numDipoles, int numElectrodes, double acc, double uS) :
  dipCart_(dipoles), elecCart_(elecCart), radii_(radii), radCond_(radCond), tanCond_(tanCond), result_(result),
  eps(acc), numDip(numDipoles), numElec(numElectrodes), unitsScale(uS) {

  // get memory for local copies
  if(!initialize()) {
	printf("Impossible to initialize SphericalVolumeConductor\n");
  }

  // project electrodes on outer sphere surface and get locations in spherical coordinates
  //getSphericalElectrodePositions();

  // get dipole positions and moments in spherical coordinates and rotate the system such that
  // the dipole moment is in the xz plane
  //if(!getSphericalDipoleRepresentation()) {
  //printf("Dipole is not within inner sphere (brain)\n");
  //}

  if(!getSphericalDipoleAndElectrodePositions()) {
	printf("Dipole is not within inner sphere\n");
  }

  // normalize radii, dipole position, ...
  //normalizeLengthScales();

  // precompute the series coefficients according to the required accuracy / maximun number of series terms
  precompSeriesTerms();

  //refPotential = getRefPotential();
  refPotential = 0.0;

}

/**
 * SphericalVolumeConductor::~SphericalVolumeConductor()
 * Destructor
 */
SphericalVolumeConductor::~SphericalVolumeConductor() {
  // free some memory
  destroy();
}

/**
 * SphericalVolumeConductor::initialize()
 * allocates some memory
 * @return  error code  true if initialization was successful, else false;
 */
bool SphericalVolumeConductor::initialize() {
  nt = MAX_TERMS;
  dipole = new double[5];
  if(dipole == NULL)
	return false;
  moment = new double[6];
  if(moment == NULL)
	return false;
  momentcart = new double[6];
  if(momentcart == NULL)
	return false;
  seriesCoeff = new double[nt];
  if(seriesCoeff == NULL)
	return false;
  electrode = new double*[numElec];
  eleccart  = new double*[numElec];
  if(electrode == NULL)
	return false;
  if(eleccart == NULL)
	return false;
  for(int i = 0; i < numElec; i++) {
	electrode[i] = new double[6];
	eleccart[i]  = new double[6];
	if(electrode[i] == NULL)
	  return false;
	if(eleccart[i] == NULL)
	  return false;
  }
  // get local copies with index starting at 1 (to be in sync with the paper)
  double *tmp = new double[4];
  if(tmp == NULL)
	return false;
  r = tmp-1;
  double *tmp2 = new double[4];
  if(tmp2 == NULL)
	return false;
  rc = tmp2-1;
  double *tmp3 = new double[4];
  if(tmp3 == NULL)
	return false;
  tc = tmp3-1;
  for(int i = 0; i < 4; i++) {
	r[i+1]  = radii_[i];
	rc[i+1] = radCond_[i]*unitsScale; /* unit */
	tc[i+1] = tanCond_[i]*unitsScale; /* unit */
  }
  // get local copies of electrode positions in cartesian coordinates
  for(int i=0; i<numElec; i++) {
	eleccart[i][XC] = elecCart_[i][XC];
	eleccart[i][YC] = elecCart_[i][YC];
	eleccart[i][ZC] = elecCart_[i][ZC];
  }
  return true;
}

/**
 * SphericalVolumeConductor::destroy()
 * releases some memory
 */
void SphericalVolumeConductor::destroy() {
  delete []dipole;
  dipole = NULL;
  delete []moment;
  moment = NULL;
  delete []momentcart;
  momentcart = NULL;
  delete []seriesCoeff;
  seriesCoeff = NULL;
  double *tmp = r+1;
  delete []tmp;
  tmp = NULL;
  double *tmp2 = rc+1;
  delete []tmp2;
  tmp2 = NULL;
  double *tmp3 = tc+1;
  delete []tmp3;
  tmp3 = NULL;
  for(int i = 0; i < numElec; i++) {
	delete []electrode[i];
	delete []eleccart[i];
	electrode[i] = NULL;
	eleccart[i]  = NULL;
  }
  delete []electrode;
  electrode = NULL;
  delete []eleccart;
  eleccart  = NULL;
}

/**
 * SphericalVolumeConductor::computePotential()
 * calculates the potential at the electrode positions
 */
void SphericalVolumeConductor::computePotential() {
  int n;
  int i;
  for(i = 0; i < numElec; i++) {
	result_[i] = 0.0;
	for(n = 1; n <= seriesTerms; n++) {
	  result_[i] += seriesCoeff[n] * (((dipole[RM] * P0(n, cos(electrode[i][THETA]))) / dipole[RAD])
									  + (dipole[TM] * P1(n, cos(electrode[i][THETA])) * cos(electrode[i][PHI]) / (v(n,4)*dipole[RAD])));
	}
	result_[i] *= 1.0/(4.0*M_PI);
	result_[i] -= refPotential; 
  }
}

double SphericalVolumeConductor::getRefPotential() {
  int n;
  double value = 0.0;
  for(n = 1; n <= seriesTerms; n++) {
	value += seriesCoeff[n] * (((dipole[RM] * P0(n, cos(/*electrode[i][THETA]*/M_PI))) / dipole[RAD])
							   + (dipole[TM] * P1(n, cos(/*electrode[i][THETA]*/M_PI)) * cos(/*electrode[i][PHI]*/0.0) / (v(n,4)*dipole[RAD])));
  }
  value *= 1.0/(4.0*M_PI);
  return value;
}

/**
 * SphericalVolumeConductor::precompSeriesTerms()
 * The series terms have to be computed only once, since they are independent of
 * the electrode position
 */
void SphericalVolumeConductor::precompSeriesTerms() {
  int n;
  double delSum = 0.0;
  double crit = 0.0;
  for(n = 1; n < nt; n++) {
	seriesCoeff[n] = (((2.0*n+1.0)*v(n,4))/((2.0*v(n,4)+1.0)* rc[4])) * 
	  ((2.0*v(n,1)+1.0) / v(n,1)) *
	  ((e(n,1)*e(n,2)*e(n,3)*e(n,4)) / (D(n)*r[1]));
	delSum += seriesCoeff[n];
	crit = fabs(seriesCoeff[n]/delSum);
	if(crit < eps)
	  break;
  }
  seriesTerms = n;  
  if(seriesTerms == MAX_TERMS) 
	printf("Warning: SphericalVolumeConductor::getPotential -> Impossible to get requested accuracy with %d series terms;\n", MAX_TERMS);
}

/**
 * SphericalVolumeConductor::getNumberOfSeriesTerms()
 * @return  int  number of series expansions required for a certain accuracy (set in the constructor)
 */
int SphericalVolumeConductor::getNumberOfSeriesTerms() {
  return seriesTerms;
}

/**
 *
 */
double SphericalVolumeConductor::v(int n, int j) {
  return 0.5*(-1.0 + sqrt(1.0+4.0*n*(n+1)*tc[j]/rc[j]));
}

/**
 *
 */
double SphericalVolumeConductor::e(int n, int i) {
  if(i != 4)
	return exp(log(r[i+1]/r[i]) * v(n,i));
  return exp(log(dipole[RAD]/r[i]) * v(n,i));
}

/**
 *
 */
double SphericalVolumeConductor::D(int n) {
  double val = ((d11(n) * e(n,1) * e(n,1) * e(n,2) * e(n,2) * e(n,3) * e(n,3) * r[4] / r[1] + 
				 d12(n) * e(n,1) * e(n,1) * e(n,3) * e(n,3) * r[4] * r[2] / (r[3] * r[1]) +
				 d13(n) * e(n,1) * e(n,1) * e(n,2) * e(n,2) * r[3] / r[1] +
				 d14(n) * e(n,1) * e(n,1) * r[2] / r[1]) * ((v(n,1) + 1.0)/v(n,1)) +
				d21(n) * e(n,2) * e(n,2) * e(n,3) * e(n,3) * r[4] / r[2] +
				d22(n) * e(n,3) * e(n,3) * r[4] / r[3] +
				d23(n) * e(n,2) * e(n,2) * r[3] / r[2] +
				d24(n));
  return val;
}

/**
 * legendre function
 */ 
double SphericalVolumeConductor::P0(int n, double x) {

  if(fabs(x) < 1e-6)
	x = 0.0;

  if(n == 0) {
	p0[0] = 0.0;
	p0[1] = 0.0;
	return 0.0;
  }

  if(n == 1) {
	p0[0] = 1.0;
	p0[1] = x;
	return p0[1];
  }
  double pnew = (1.0/n) * ( (2.0*n-1.0)*x*p0[1] - (n-1)*p0[0]);
  if(fabs(pnew) < 1e-6)
	pnew = 0.0;
  p0[0] = p0[1];
  p0[1] = pnew;
  return pnew;

}

/**
 * derivative of legendre function
 */ 
double SphericalVolumeConductor::P1(int n, double x) {

  if(fabs(x) < 1e-6)
	x = 0.0;

  if(n == 0) {
	p1[0] = 0.0;
	p1[1] = 0.0;
	return 0.0;
  }

  if(n == 1) {
	p1[0] = -sqrt(1.0-x*x);
	p1[1] = 0.0;
	if(fabs(p1[0]) < 1e-6)
	  p1[0] = 0.0;
	return p1[0];
  }

  if(n == 2) {
	p1[1] = -3.0*x*sqrt(1.0-x*x);
	if(fabs(p1[1]) < 1e-6)
	  p1[1] = 0.0;
	return p1[1];
  }

  double pnew = (1.0/(n-1.0)) * (x*(2.0*n-1.0)*p1[1] - n*p1[0]);
  if(fabs(pnew) < 0.0)
	pnew = 0.0;
  p1[0] = p1[1];
  p1[1] = pnew;
  
  return pnew;
}

/**
 *
 */
double SphericalVolumeConductor::c11(int n, int j) {
  return (rc[j-1]*v(n,j-1) + rc[j]*(v(n,j)+1.0)) / ((2.0*v(n,j)+1.0)*rc[j]);
}

/**
 *
 */
double SphericalVolumeConductor::c12(int n, int j) {
  return (-rc[j-1]*(v(n,j-1)+1.0) + rc[j]*(v(n,j)+1.0)) / ((2.0*v(n,j)+1.0)*rc[j]);
}

/**
 *
 */
double SphericalVolumeConductor::c21(int n, int j) {
  return (-rc[j-1]*v(n,j-1) + rc[j]*v(n,j)) / ((2.0*v(n,j)+1.0)*rc[j]);
}

/**
 *
 */
double SphericalVolumeConductor::c22(int n, int j) {
  return (rc[j-1]*(v(n,j-1)+1.0) + rc[j]*v(n,j)) / ((2.0*v(n,j)+1.0)*rc[j]);
}

/**
 *
 */
double SphericalVolumeConductor::d11(int n) {
  return (c21(n,4)*c11(n,3)*c11(n,2));
}

/**
 *
 */
double SphericalVolumeConductor::d12(int n) {
  return (c21(n,4)*c12(n,3)*c21(n,2));
}

/**
 *
 */
double SphericalVolumeConductor::d13(int n) {
  return (c22(n,4)*c21(n,3)*c11(n,2));
}

/**
 *
 */
double SphericalVolumeConductor::d14(int n) {
  return (c22(n,4)*c22(n,3)*c21(n,2));
}

/**
 *
 */
double SphericalVolumeConductor::d21(int n) {
  return (c21(n,4)*c11(n,3)*c12(n,2));
}

/**
 *
 */
double SphericalVolumeConductor::d22(int n) {
  return (c21(n,4)*c12(n,3)*c22(n,2));
}

/**
 *
 */
double SphericalVolumeConductor::d23(int n) {
  return (c22(n,4)*c21(n,3)*c12(n,2));
}

/**
 *
 */
double SphericalVolumeConductor::d24(int n) {
  return (c22(n,4)*c22(n,3)*c22(n,2));
}

/**
 * conversion: cartesian -> spherical coordinates of the electrodes
 */
void SphericalVolumeConductor::getSphericalElectrodePositions() {
  double r_tmp;
  for(int i = 0; i < numElec; i++) {
	// project electrodes on the outer sphere surface (scalp) 
	//r_tmp = getRadius(elecCart_[i][XC], elecCart_[i][YC], elecCart_[i][ZC]);
	//elecCart_[i][XC]  *= radii_[SCALP]/r_tmp; // x
	//elecCart_[i][YC]  *= radii_[SCALP]/r_tmp; // y
	//elecCart_[i][ZC]  *= radii_[SCALP]/r_tmp; // z
	// get electrode position in spherical coordinates 
	electrode[i][PHI] = getPhi(elecCart_[i][XC], elecCart_[i][YC]);
	r_tmp = getRadius(elecCart_[i][XC], elecCart_[i][YC], elecCart_[i][ZC]);
	electrode[i][THETA] = getTheta(elecCart_[i][ZC], r_tmp);
	electrode[i][RAD] = radii_[SCALP];
  }
}

/**
 * coversion: cartesian -> spherical coordinates of the dipol
 * rotate system such that:
 *  1) dipole position is on z-axis and
 *  2) dipole moment lies in xz-plane
 */
bool SphericalVolumeConductor::getSphericalDipoleRepresentation() {

  // get dipole position in spherical coordinates
  dipole[RAD]   = getRadius(dipCart_[0][XC], dipCart_[0][YC], dipCart_[0][ZC]);
  dipole[THETA] = getTheta(dipCart_[0][ZC], dipole[RAD]);
  dipole[PHI]   = getPhi(dipCart_[0][XC], dipCart_[0][YC]);

  // get intersection point sphere surface & dipole moment
  moment[XM] = dipCart_[0][XM]; moment[YM] = dipCart_[0][YM]; moment[ZM] = dipCart_[0][ZM];
  double length = sqrt(moment[XM]*moment[XM] + moment[YM]*moment[YM] + moment[ZM]*moment[ZM]);
  moment[XM] /= length; moment[YM] /= length; moment[ZM] /= length;

  double B = 2.0 * (moment[XM]*dipCart_[0][XC] + moment[YM]*dipCart_[0][YC] + moment[ZM]*dipCart_[0][ZC]);
  double C = dipCart_[0][XC]*dipCart_[0][XC] + dipCart_[0][YC]*dipCart_[0][YC] + dipCart_[0][ZC]*dipCart_[0][ZC] - radii_[SCALP]*radii_[SCALP];
  double d = B*B - 4.0*C;
  if(d < 0.0) {
	printf("SphericalVolumeConductor -> impossible to determine ray sphere intersection\n");
  }
  d = sqrt(d);
  double t0 = (-B + d)*0.5;
  double t1 = (-B - d)*0.5;
  double t  = 0.0;

  if(t0 > 0.0) {
	t = t0;
  }
  else {
	if(t1 > 0.0) {
	  t = t1;
	}
	else {
	  printf("SphericalVolumeConductor -> impossible to determine ray sphere intersection\n");
	}
  }
  
  double sol[3]; sol[XC] = dipCart_[0][XC] + t*moment[XM]; sol[YC] = dipCart_[0][YC] + t*moment[YM]; sol[ZC] = dipCart_[0][ZC] + t*moment[ZM];
  
  moment[RAD]   = getRadius(sol[XC], sol[YC], sol[ZC]);
  moment[PHI]   = getPhi(sol[XC], sol[YC]);
  moment[THETA] = getTheta(sol[ZC], moment[RAD]);

  // a) put dipole into xz plane -> moment vector changes
  if((moment[THETA] != 0.0) && (moment[THETA] != M_PI)) {
	moment[PHI] -= dipole[PHI];
	if(moment[PHI] < 0.0)
	  moment[PHI] += 2.0*M_PI;
  }
  // b) put dipole onto z axis -> moment vector changes
  if((moment[PHI] > (0.5*M_PI)) && (moment[PHI] < (1.5*M_PI))) {
	moment[THETA] += dipole[THETA];
  }
  else {
	if((moment[PHI] != (0.5*M_PI)) && (moment[PHI] != (1.5*M_PI)))
	  moment[THETA] -= dipole[THETA];
  }
  if(moment[THETA] < 0.0) {
	moment[THETA] *= -1.0;
	moment[PHI] += M_PI;
	if(moment[PHI] > (2.0*M_PI))
	  moment[PHI] -= 2.0*M_PI;
  }
  if(moment[THETA] > M_PI) {
	moment[THETA] = 2.0 * M_PI - moment[THETA];
	moment[PHI] += M_PI;
	if(moment[PHI] > (2.0*M_PI))
	  moment[PHI] -= 2.0*M_PI;
  }
  for(int i = 0; i < numElec; i++) {	
	// do the same rotations to all electrodes 
	// a)
	if((electrode[i][THETA] != 0.0) && (electrode[i][THETA] != M_PI)) {
	  electrode[i][PHI] -= dipole[PHI];
	  if(electrode[i][PHI] < 0.0) {
		electrode[i][PHI] += 2.0*M_PI;
	  }
	}

	// b)
	if((electrode[i][PHI]>(0.5*M_PI)) && (electrode[i][PHI]<(1.5*M_PI))) {
	  electrode[i][THETA] += dipole[THETA];
	}
	else {
	  if((electrode[i][PHI] != (0.5*M_PI)) && (electrode[i][PHI] != (1.5*M_PI))) {
		electrode[i][THETA] -= dipole[THETA];
	  }
	}
	if(electrode[i][THETA] < 0.0) {
	  electrode[i][THETA] *= -1.0;
	  electrode[i][PHI] += M_PI;
	  if(electrode[i][PHI] > (2.0*M_PI)) {
		electrode[i][PHI] -= 2.0*M_PI;
	  }
	}
	if(electrode[i][THETA] > M_PI) {
	  electrode[i][THETA] = 2.0*M_PI - electrode[i][THETA];
	  electrode[i][PHI] += M_PI;
	  if(electrode[i][PHI] > (2.0*M_PI)) {
		electrode[i][PHI] -= 2.0*M_PI;
	  }
	}

	// phi correction: put moment vector into xz plane -> electrode positions are changing
	if((electrode[i][THETA] != 0.0) && (electrode[i][THETA] != M_PI)) {
	  electrode[i][PHI] -= moment[PHI];
	  if(electrode[i][PHI] < 0.0) {
		electrode[i][PHI] += 2.0*M_PI;
	  }
	}
  }

  double dz = moment[RAD] * cos(moment[THETA]) - dipole[RAD];
  double dx = moment[RAD] * sin(moment[THETA]);
  double theta = getPhi(dz, dx);

  // get radial & tangential moment of the moment vector which is now in the xz plane
  dipole[RM] = length * cos(theta);
  dipole[TM] = -1.0 * length * sin(theta);

  if(dipole[RAD] > radii_[BRAIN])
	return false;
  return true;

}

/**
 *
 */
double SphericalVolumeConductor::getRadius(double x, double y, double z) {
  return sqrt(x*x + y*y + z*z);
}

/**
 *
 */
double SphericalVolumeConductor::getPhi(double x, double y) {
  if(x == 0.0) { // x = 0.0
	if(y > 0.0) { // x = 0.0; y > 0.0
	  return (0.5 * M_PI);
	}
	else {
	  if(y < 0.0) { // x = 0.0; y < 0.0
		return (1.5 * M_PI);
	  }
	  else { // x = 0; y = 0;
		return 0.0;
	  }
	}
  }
  else { 
	if(y == 0.0) { // y = 0.0
	  if(x > 0.0) { // y = 0.0; x > 0.0
		return 0.0;
	  }
	  else { // y = 0.0; x < 0.0
		return M_PI;
	  }
	}
	else { // x != 0; y != 0
	  if((x > 0.0) && (y > 0.0)) { // x > 0; y > 0
		return (atan(y/x));
	  }
	  else {
		if((x < 0.0) && (y > 0.0)) { // x < 0; y > 0
		  return (atan(y/x) + M_PI);
		}
		else {
		  if((x > 0.0) && (y < 0.0)) { // x > 0; y < 0
			return (atan(y/x) + 2.0 * M_PI);
		  }
		  else { // x < 0; y < 0
			return (atan(y/x) + M_PI);
		  }
		}
	  }
	}
  }
}

/**
 *
 */
double SphericalVolumeConductor::getTheta(double z, double r) {
  return acos(z/r);
}

/**
 *
 */
void SphericalVolumeConductor::normalizeLengthScales() {
  // normalize all length scales w.r.t. the scalp
  double norm = radii_[SCALP];
  r[1] /= norm;
  r[2] /= norm;
  r[3] /= norm;
  r[4] /= norm;
  dipole[RAD] /= norm;
  dipole[RM] /= norm;
  dipole[TM] /= norm;
  rc[1] *= norm; tc[1] *= norm;
  rc[2] *= norm; tc[2] *= norm;
  rc[3] *= norm; tc[3] *= norm;
  rc[4] *= norm; tc[4] *= norm;
}

bool SphericalVolumeConductor::getSphericalDipoleAndElectrodePositions() {
  double tmp;
  double phi_tmp;
  // get dipole position in spherical coordinates
  dipole[RAD]   = getRadius(dipCart_[0][XC], dipCart_[0][YC], dipCart_[0][ZC]);
  dipole[THETA] = getTheta(dipCart_[0][ZC], dipole[RAD]);
  dipole[PHI]   = getPhi(dipCart_[0][XC], dipCart_[0][YC]);
  // get intersection point sphere surface & dipole moment
  moment[XM] = dipCart_[0][XM]; moment[YM] = dipCart_[0][YM]; moment[ZM] = dipCart_[0][ZM];
  abs_moment = sqrt(moment[XM]*moment[XM] + moment[YM]*moment[YM] + moment[ZM]*moment[ZM]);
  moment[XM] /= abs_moment; moment[YM] /= abs_moment; moment[ZM] /= abs_moment;
  double B = 2.0 * (moment[XM]*dipCart_[0][XC] + moment[YM]*dipCart_[0][YC] + moment[ZM]*dipCart_[0][ZC]);
  double C = dipCart_[0][XC]*dipCart_[0][XC] + dipCart_[0][YC]*dipCart_[0][YC] + dipCart_[0][ZC]*dipCart_[0][ZC] - radii_[SCALP]*radii_[SCALP];
  double d = B*B - 4.0*C;
  if(d < 0.0) {
	printf("SphericalVolumeConductor -> impossible to determine ray sphere intersection\n");
  }
  d = sqrt(d);
  double t0 = (-B + d)*0.5;
  double t1 = (-B - d)*0.5;
  double t  = 0.0;
  if(t0 > 0.0) {
	t = t0;
  }
  else {
	if(t1 > 0.0) {
	  t = t1;
	}
	else {
	  printf("SphericalVolumeConductor -> impossible to determine ray sphere intersection\n");
	}
  }
  momentcart[XC] = dipCart_[0][XC] + t*moment[XM]; momentcart[YC] = dipCart_[0][YC] + t*moment[YM]; momentcart[ZC] = dipCart_[0][ZC] + t*moment[ZM];
  // a) put dipole into xz-plane
  if(fabs(dipole[PHI]) > 1e-8) {
	tmp = momentcart[XC];
	momentcart[XC] = momentcart[XC]*cos(dipole[PHI]) + momentcart[YC]*sin(dipole[PHI]); 
	momentcart[YC] = -tmp*sin(dipole[PHI]) + momentcart[YC]*cos(dipole[PHI]);
  }
  // b) put dipole onto z-axis
  if(fabs(dipole[THETA]) > 1e-8) {
	tmp = momentcart[XC];
	momentcart[XC] = momentcart[XC]*cos(dipole[THETA]) - momentcart[ZC]*sin(dipole[THETA]);
	momentcart[ZC] = tmp*sin(dipole[THETA]) + momentcart[ZC]*cos(dipole[THETA]);
  }
  // c) put dipole moment into xz-plane
  phi_tmp = getPhi(momentcart[XC], momentcart[YC]);
  if(fabs(phi_tmp) > 1e-8) {
	tmp = momentcart[XC];
	momentcart[XC] = momentcart[XC]*cos(phi_tmp) + momentcart[YC]*sin(phi_tmp); 
	momentcart[YC] = -tmp*sin(phi_tmp) + momentcart[YC]*cos(phi_tmp);
  }
  // get moment position in spherical coordinates
  moment[RAD]   = getRadius(momentcart[XC], momentcart[YC], momentcart[ZC]);
  moment[PHI]   = getPhi(momentcart[XC], momentcart[YC]);
  moment[THETA] = getTheta(momentcart[ZC], moment[RAD]);
  // get transformed dipole info ... //
  double dz = momentcart[ZC]-dipole[RAD];
  double dx = momentcart[XC];
  getDipoleMoment(dx,dz);
  // get transformed electrode positions //
  double r_tmp;
  for(int i = 0; i < numElec; i++) {
	// a)
	if(fabs(dipole[PHI]) > 1e-8) {
	  tmp = eleccart[i][XC];
	  eleccart[i][XC] = eleccart[i][XC]*cos(dipole[PHI]) + eleccart[i][YC]*sin(dipole[PHI]); 
	  eleccart[i][YC] = -tmp*sin(dipole[PHI]) + eleccart[i][YC]*cos(dipole[PHI]);
	}
	// b)
	if(fabs(dipole[THETA]) > 1e-8) {
	  tmp = eleccart[i][XC];
	  eleccart[i][XC] = eleccart[i][XC]*cos(dipole[THETA]) - eleccart[i][ZC]*sin(dipole[THETA]);
	  eleccart[i][ZC] = tmp*sin(dipole[THETA]) + eleccart[i][ZC]*cos(dipole[THETA]);
	}
	// c)
	if(fabs(phi_tmp) > 1e-8) {
	  tmp = eleccart[i][XC];
	  eleccart[i][XC] = eleccart[i][XC]*cos(phi_tmp) + eleccart[i][YC]*sin(phi_tmp); 
	  eleccart[i][YC] = -tmp*sin(phi_tmp) + eleccart[i][YC]*cos(phi_tmp);
	}
	// get spherical coordinates of electrode positions
	electrode[i][PHI] = getPhi(eleccart[i][XC], eleccart[i][YC]);
	r_tmp = getRadius(eleccart[i][XC], eleccart[i][YC], eleccart[i][ZC]);
	electrode[i][THETA] = getTheta(eleccart[i][ZC], r_tmp);
	electrode[i][RAD] = radii_[SCALP];
  }

  if(dipole[RAD] > radii_[BRAIN])
	return false;
  return true;
 
}

void SphericalVolumeConductor::getDipoleMoment(double dx, double dz) {
  if(dx < 0.0) {
	printf("\n error: dx < 0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
	dipole[RM] = 0.0; dipole[TM] = 0.0;
	return;
  }
  if(dx == 0.0 && dz == 0.0) {
	dipole[RM] = 0.0;
	dipole[TM] = 0.0;
	return;
  }
  if((dx == 0.0) && (dz != 0.0)) {
	dipole[TM] = 0.0;
	if(dz > 0) 
	  dipole[RM] = abs_moment;
	else
	  dipole[RM] = -abs_moment;
	return;
  }
  if((dz == 0) && (dx != 0.0)) {
	dipole[RM] = 0.0;
	dipole[TM] = -abs_moment; 
	return;
  }
  //case 1
  if(dz > 0.0) {
	double theta = atan(dz/dx);
	dipole[RM] = abs_moment * sin(theta);
	dipole[TM] = -abs_moment * cos(theta); 
	return;
  }
  // case 2
  if(dz < 0.0) {
	double theta = atan(fabs(dz/dx));
	dipole[RM] = -abs_moment * sin(theta);
	dipole[TM] = -abs_moment * cos(theta); 
  }
}

} // end namespace
