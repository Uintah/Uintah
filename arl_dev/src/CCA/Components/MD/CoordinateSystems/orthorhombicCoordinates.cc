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
 * orthorhombicCoordinates.cc
 *
 *  Created on: May 12, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/CoordinateSystems/coordinateSystem.h>
#include <CCA/Components/MD/CoordinateSystems/orthorhombicCoordinates.h>

#include <Core/Exceptions/InvalidState.h>

using namespace Uintah;

orthorhombicCoordinates::orthorhombicCoordinates(const SCIRun::IntVector& _extent,
                                                 const SCIRun::IntVector& _periodic,
                                                 const Uintah::Matrix3& _cell)
                                                :coordinateSystem(_extent, _periodic) {
  double d_zeroTol = 1.0e-13;

  if ( abs(_cell(0,1)) > d_zeroTol || abs(_cell(0,2)) > d_zeroTol ||
       abs(_cell(1,0)) > d_zeroTol || abs(_cell(1,2)) > d_zeroTol ||
       abs(_cell(2,0)) > d_zeroTol || abs(_cell(2,1)) > d_zeroTol  ) {
    throw InvalidState("Error:  Attempt to instantiate an orthorhombic coordinate system with a non-orthorhombic unit cell",
                               __FILE__,
                               __LINE__);
  }
  if (_cell(0,0) < d_zeroTol || _cell(1,1) < d_zeroTol || _cell(2,2) < d_zeroTol) {
    throw InvalidState("Error:  Attempt to instantiate a coordinate system with a principle vector of length zero",
                               __FILE__,
                               __LINE__);
  }

  d_Lengths[0]=_cell(0,0);
  d_Lengths[1]=_cell(1,1);
  d_Lengths[2]=_cell(2,2);

  calculateCellVolume();
  isVolumeCurrent = true;
  calculateInverse();
  isInverseCurrent = true;
  markCellChanged();

}

orthorhombicCoordinates::orthorhombicCoordinates(const SCIRun::IntVector& _extent,
                                                 const SCIRun::IntVector& _periodic,
                                                 const SCIRun::Vector& _lengths,
                                                 const SCIRun::Vector& _angles)
                                                :coordinateSystem(_extent, _periodic) {

  double d_zeroTol = 1.0e-13;
  SCIRun::Vector angleDiff = _angles*MDConstants::degToRad-SCIRun::Vector(MDConstants::PI_Over_2);
  if (abs(angleDiff[0]) > d_zeroTol || abs(angleDiff[1]) > d_zeroTol || abs(angleDiff[2]) > d_zeroTol) {
    throw InvalidState("Error:  Attempt to instantiate an orthorhombic coordinate system without all cell angles equal to 90 degrees",
                               __FILE__,
                               __LINE__);
  }

  if ((_lengths[0] < d_zeroTol) || (_lengths[1] < d_zeroTol) || (_lengths[2] < d_zeroTol)) {
    throw InvalidState("Error:  Attempt to instantiate a coordinate system with a principle vector of length zero",
                               __FILE__,
                               __LINE__);
  }

  d_Lengths = _lengths;

  calculateCellVolume();
  isVolumeCurrent = true;
  calculateInverse();
  isInverseCurrent = true;
  markCellChanged();

}

orthorhombicCoordinates::~orthorhombicCoordinates()
{

}

void orthorhombicCoordinates::getInverseCell(Uintah::Matrix3& In) {

  if (!isInverseCurrent) {
    calculateInverse();
  }

  In(0,0) = d_InverseLengths[0];
  In(1,1) = d_InverseLengths[1];
  In(2,2) = d_InverseLengths[2];
  In(0,1) = 0.0;
  In(0,2) = 0.0;
  In(1,0) = 0.0;
  In(1,2) = 0.0;
  In(2,1) = 0.0;
  In(2,2) = 0.0;

}

void orthorhombicCoordinates::updateUnitCell(const Uintah::Matrix3& In) {
  double d_zeroTol = 1.0e-13;

  if ( abs(In(0,1)) > d_zeroTol || abs(In(0,2)) > d_zeroTol ||
       abs(In(1,0)) > d_zeroTol || abs(In(1,2)) > d_zeroTol ||
       abs(In(2,0)) > d_zeroTol || abs(In(2,1)) > d_zeroTol  ) {
    throw InvalidState("Error:  Attempt to instantiate an orthorhombic coordinate system with a non-orthorhombic unit cell",
                               __FILE__,
                               __LINE__);
  }
  if (In(0,0) < d_zeroTol || In(1,1) < d_zeroTol || In(2,2) < d_zeroTol) {
    throw InvalidState("Error:  Attempt to instantiate a coordinate system with a principle vector of length zero",
                               __FILE__,
                               __LINE__);
  }

  d_Lengths[0]=In(0,0);
  d_Lengths[1]=In(1,1);
  d_Lengths[2]=In(2,2);

  calculateCellVolume();
  isVolumeCurrent = true;
  calculateInverse();
  isInverseCurrent = true;
  markCellChanged();
}

void orthorhombicCoordinates::updateUnitCell(const SCIRun::Vector& inLengths,
                                             const SCIRun::Vector& inAngles) {

  // Update the angles first in case they're not valid so we don't screw the Lengths
  updateBasisAngles(inAngles);
  updateBasisLengths(inLengths);

}

void orthorhombicCoordinates::minimumImageDistance(const SCIRun::Point& P1,
                                                   const SCIRun::Point& P2,
                                                         SCIRun::Vector& offset) const {

  offset = (P2 - P1);

  // Convert to reduced coordinates
  offset[0] *= d_InverseLengths[0];
  offset[1] *= d_InverseLengths[1];
  offset[2] *= d_InverseLengths[2];

  // Only need minimum image projection for directions in which we are periodic
  // Minimum image
  offset[0] = offset[0] - trunc(2.0*offset[0]*this->periodicX());
  offset[1] = offset[1] - trunc(2.0*offset[1]*this->periodicY());
  offset[2] = offset[2] - trunc(2.0*offset[2]*this->periodicZ());

  // Recast into cartesian
  offset[0] *= d_Lengths[0];
  offset[1] *= d_Lengths[1];
  offset[2] *= d_Lengths[2];

}
