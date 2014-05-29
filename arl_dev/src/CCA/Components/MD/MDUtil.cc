/*
 *
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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
 *
 * ----------------------------------------------------------
 * MDUtil.cc
 *
 *  Created on: May 13, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/MDUtil.h>

using namespace Uintah;

const double MDConstants::PI= acos(-1.0);
const double MDConstants::PI_Over_2 = PI/2.0;
const double MDConstants::PI2 = PI*PI;
const double MDConstants::orthogonalAngle = 90.0;
const double MDConstants::degToRad = PI/180.0;
const double MDConstants::radToDeg = 180.0/PI;
const double MDConstants::zeroTol = 1.0e-13;
const double MDConstants::defaultDipoleMixRatio = 0.15;
const double MDConstants::defaultPolarizationTolerance = 1.0e-9;
// Define some useful vector constants
const SCIRun::IntVector MDConstants::IV_ZERO(0,0,0);
const SCIRun::IntVector MDConstants::IV_ONE(1,1,1);
const SCIRun::IntVector MDConstants::IV_X(1,0,0);
const SCIRun::IntVector MDConstants::IV_Y(0,1,0);
const SCIRun::IntVector MDConstants::IV_Z(0,0,1);

const SCIRun::Vector    MDConstants::V_ZERO(0.0, 0.0, 0.0);
const SCIRun::Vector    MDConstants::V_ONE(1.0, 1.0, 1.0);
const SCIRun::Vector    MDConstants::V_X(1.0, 0.0, 0.0);
const SCIRun::Vector    MDConstants::V_Y(0.0, 1.0, 0.0);
const SCIRun::Vector    MDConstants::V_Z(0.0, 0.0, 1.0);

const Uintah::Matrix3   MDConstants::M3_I(1.0, 0.0, 0.0,
                                          0.0, 1.0, 0.0,
                                          0.0, 0.0, 1.0);


