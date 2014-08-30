/*
 * Copyright (c) 2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef GAUSS_KRONROD_H
#define GAUSS_KRONROD_H

#include <iostream>
#include <fstream>
#include <string>

#include <tabprops/prepro/mixmdl/Integrator.h>

//------------------------------------------------------------------
// this include file must define:
//    nGaussKronrod      - number of Gauss-Kronrod quadrature points
//    gaussKronrodPoint  - double[nGaussKronrod]
//                         Gauss-Kronrod node locations [-1,1]
//    gaussKronrodWeight - double[nGaussKronrod]
//                         Gauss-Kronrod weights.
//    nGauss             - number of underlying Gaussian quadrature points
//    gaussPoint         - double[nGauss]
//                         gaussian quadrature points
//    gaussWeight        - double[nGauss]
//                         gaussian quadrature weights
#define DQAGP_21
//#define JACOBI_7
#include <tabprops/prepro/mixmdl/GaussKronrodPts.h>
//------------------------------------------------------------------


/**
 *  @class GaussKronrod
 *  @author James C. Sutherland
 *  @date   April, 2005
 *
 *  Adaptive Gaussian quadrature integrator suitable for singular
 *  functions.  This integrator can use any Gaussian-Kronrod scheme,
 *  with the appropriate weights and points defined.
 */
class GaussKronrod : public Integrator
{

 public:

  /** Create an instance of a GaussKronrod integrator */
  GaussKronrod( const double lo,
                const double hi,
                const int nIntervals=1 );

  ~GaussKronrod();

  double integrate();

  /** set the maximum recursion levels */
  void set_max_levels( const int ml ){ maxLevels_ = ml; }

  /**
   *  This method activates interrogation of the integrator.
   *  Each function evaluation will be written to a file, along
   *  with the function value.
   */
  void interrogate(){ interrogate_ = true; }

  /**
   *  if interrogation is active, this will set the file name to write
   *  the information to.
   */
  void set_outfile_name( const std::string & n ){ outfileName_=n; }

  int get_n_intervals() const{ return nIntervals_; };

 private:

  int nIntervals_;
  const int initIntervals_;
  int maxLevels_, maxIntervals_;
  bool interrogate_;

  std::string outfileName_;

  void set_bounds( std::set<double> & bounds );

  double evaluate( const double a,
		   const double b,
		   const int level );

  std::ofstream fout;

};

#endif
