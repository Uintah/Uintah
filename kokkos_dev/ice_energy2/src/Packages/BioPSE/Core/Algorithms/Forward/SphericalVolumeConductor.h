/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

/*
 * file:     SphericalVolumeConductor.h 
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

#ifndef SPHERICAL_VOLUME_CONDUCTOR_H
#define SPHERICAL_VOLUME_CONDUCTOR_H

/* Define M_PI if necessary */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_TERMS 200

// define some indices
#define RAD 0 
#define PHI 1 
#define THETA 2 
#define RM 3 
#define TM 4

#define TAN 1

#define XC 0 
#define YC 1 
#define ZC 2 
#define XM 3 
#define YM 4
#define ZM 5 

#define SCALP 0
#define SKULL 1
#define CBSF  2
#define BRAIN 3
#define AIR   4

#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>

#include <Packages/BioPSE/Core/Algorithms/Forward/share.h>

namespace BioPSE {

using namespace SCIRun;

class SCISHARE SphericalVolumeConductor {

public:

  // constructor
  SphericalVolumeConductor(DenseMatrix  &dipoles, DenseMatrix &electrodes, ColumnMatrix const& radii,
						   ColumnMatrix const& radCond, ColumnMatrix const& tanCond, ColumnMatrix &result, 
						   int numDipoles, int numElectrodes, double accuracy, double us);

  // destructor
  virtual ~SphericalVolumeConductor();

  // public access methods
  void computePotential();
  int getNumberOfSeriesTerms();

private:

  bool initialize();
  void destroy();
  void getSphericalElectrodePositions();
  bool getSphericalDipoleRepresentation();
  bool getSphericalDipoleAndElectrodePositions();
  double getRadius(double x, double y, double z);
  double getPhi(double x, double y);
  double getTheta(double z, double r);
  void normalizeLengthScales();
  void precompSeriesTerms();
  double getRefPotential();
  void getDipoleMoment(double dx, double dz);

  double g(int i);

  double e(int n, int i);

  double c11(int n, int j);
  double c12(int n, int j);
  double c21(int n, int j);
  double c22(int n, int j);

  double d11(int n);
  double d12(int n);
  double d13(int n);
  double d14(int n);
  double d21(int n);
  double d22(int n);
  double d23(int n);
  double d24(int n);
  
  double v(int n, int c);

  double D(int n);

  // Legendre Polynomials
  double P0(int n, double x);
  double P1(int n, double x);
  
  // input handles
  DenseMatrix  &dipCart_;
  DenseMatrix  &elecCart_;
  ColumnMatrix const& radii_;
  ColumnMatrix const& radCond_;
  ColumnMatrix const& tanCond_;
  ColumnMatrix &result_;

  // local copies
  double *dipole;
  double *moment;
  double *momentcart;
  double **electrode;
  double **eleccart;
  double *r;
  double *rc;
  double *tc;
  double abs_moment;

  // series coefficients
  double *seriesCoeff;
  int seriesTerms;

  double eps;    // accuracy
  int nt;        // max number of series terms
  int numDip;    // number of dipoles
  int numElec;   // number of electrodes

  double refPotential;

  double p0[2]; // legendre terms
  double p1[2]; // legendre terms (derivative)

  double unitsScale;

}; // end class

} // end namespace

#endif
