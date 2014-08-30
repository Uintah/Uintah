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

#ifndef BETA_MIXMDL_H
#define BETA_MIXMDL_H

#include <tabprops/prepro/mixmdl/PresumedPDFMixMdl.h>

/** @class BetaMixMdl
 *
 *  @author  James C. Sutherland
 *  @date    February, 2005
 *
 *  @brief Derived class from PresumedPDFMixMdl to represent the
 *         Beta PDF mixing model.
 *
 *  See "PresumedPDFMixingModel.h" for more documentation.
 */
class BetaMixMdl : public PresumedPDFMixMdl
{
 public:

  //-- class constructor
  BetaMixMdl();

  ~BetaMixMdl(){};

  /** @brief function to integrate the beta-PDF. */
  double integrate();

  /** @brief function to calculate the beta-PDF */
  double get_pdf( const double );

 protected:

 private:
  // scaled variance at which we no longer trust the integrator.
  const double betaCutoffVariance_;

  double bpdf( const double );

};

#endif
