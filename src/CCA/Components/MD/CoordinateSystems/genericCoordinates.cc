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
 * genericCoordinates.cc
 *
 *  Created on: May 13, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/CoordinateSystems/genericCoordinates.h>

using namespace Uintah;

genericCoordinates::genericCoordinates(const SCIRun::IntVector& _extent,
                                       const SCIRun::IntVector& _periodic,
                                       const Uintah::Matrix3& _cell)
                                      :coordinateSystem(_extent,_periodic),
                                       d_unitCell(_cell) {

  isUnitCellCurrent = true;
  calculateInverse();
  calculateBasisLengths();
  calculateBasisAngles();
  calculateCellVolume();
  markCellChanged();

}

genericCoordinates::genericCoordinates(const SCIRun::IntVector& _extent,
                                       const SCIRun::IntVector& _periodic,
                                       const SCIRun::Vector& _lengths,
                                       const SCIRun::Vector& _angles)
                                      :coordinateSystem(_extent, _periodic),
                                       d_basisLengths(_lengths),
                                       d_basisAngles(_angles) {
  areBasisLengthsCurrent = true;
  areBasisAnglesCurrent = true;
  cellFromLengthsAndAngles();
  calculateInverse();
  calculateCellVolume();
  markCellChanged();

}

genericCoordinates::~genericCoordinates()
{

}

void genericCoordinates::updateUnitCell(const Uintah::Matrix3& cellIn) {
  d_unitCell = cellIn;
  isUnitCellCurrent = true;
  isInverseCurrent = false;
  isVolumeCurrent = false;
  areBasisLengthsCurrent = false;
  areBasisAnglesCurrent = false;
  markCellChanged();

}

void genericCoordinates::updateUnitCell(const SCIRun::Vector& lengths,
                                        const SCIRun::Vector& angles) {
  d_basisLengths = lengths;
  d_basisAngles = angles;
  areBasisLengthsCurrent = true;
  areBasisAnglesCurrent = true;
  isUnitCellCurrent = false;
  isInverseCurrent = false;
  isVolumeCurrent = false;
  markCellChanged();
}

void genericCoordinates::minimumImageDistance(const SCIRun::Point& P1,
                                              const SCIRun::Point& P2,
                                                    SCIRun::Vector& Out) const {
  SCIRun::Vector offset = (P2 - P1), temp;


  // Avoid unnecessary temporary creation; this will be in inner loops.
  temp = d_inverseCell * offset;

  // Only need minimum image projection for directions in which we are periodic
  temp[0] = temp[0] - trunc(2.0*offset[0]*this->periodicX());
  temp[1] = temp[1] - trunc(2.0*offset[1]*this->periodicY());
  temp[2] = temp[2] - trunc(2.0*offset[2]*this->periodicZ());

  // Recast into cartesian
  offset = d_unitCell * temp;

}
