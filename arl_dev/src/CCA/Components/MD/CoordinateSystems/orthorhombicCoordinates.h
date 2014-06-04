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
 * orthorhombicCoordinates.h
 *
 *  Created on: May 9, 2014
 *      Author: jbhooper
 */

#ifndef ORTHORHOMBICCOORDINATES_H_
#define ORTHORHOMBICCOORDINATES_H_

#include <CCA/Components/MD/MDUtil.h>
#include <CCA/Components/MD/CoordinateSystems/coordinateSystem.h>

#include <Core/Exceptions/InvalidState.h>

namespace Uintah {
  class orthorhombicCoordinates: public coordinateSystem {
    public:
       orthorhombicCoordinates(const SCIRun::IntVector&,
                               const SCIRun::IntVector&,
                               const Uintah::Matrix3&);

       orthorhombicCoordinates(const SCIRun::IntVector&,
                               const SCIRun::IntVector&,
                               const SCIRun::Vector&,
                               const SCIRun::Vector&);

      ~orthorhombicCoordinates();

      virtual inline void toReduced(const SCIRun::Vector& _In,
                                   SCIRun::Vector& _Out) const {
        _Out[0] = _In[0]*d_InverseLengths[0];
        _Out[1] = _In[1]*d_InverseLengths[1];
        _Out[2] = _In[2]*d_InverseLengths[2];
      }

      virtual inline void toCartesian(const SCIRun::Vector& _In,
                                     SCIRun::Vector& _Out) const {

        _Out[0] = _In[0]*d_Lengths[0];
        _Out[1] = _In[1]*d_Lengths[1];
        _Out[2] = _In[2]*d_Lengths[2];

      }

      virtual inline void getUnitCell(Uintah::Matrix3& _In) {
        _In(0,0) = d_Lengths[0];
        _In(1,1) = d_Lengths[1];
        _In(2,2) = d_Lengths[2];
        _In(0,1) = 0.0;
        _In(0,2) = 0.0;
        _In(1,0) = 0.0;
        _In(1,2) = 0.0;
        _In(2,0) = 0.0;
        _In(2,1) = 0.0;
      }

      virtual void getInverseCell(Uintah::Matrix3& _In);

      inline virtual void getBasisLengths(SCIRun::Vector& _In) {
        _In = d_Lengths;
      }

      inline virtual void getBasisAngles(SCIRun::Vector& _In) {

        // Orthorhombic, all angles are 90 degrees
        _In[0] = MDConstants::PI_Over_2;
        _In[1] = MDConstants::PI_Over_2;
        _In[2] = MDConstants::PI_Over_2;
      }

      inline virtual double getCellVolume() {
        if (!isVolumeCurrent) {
          calculateCellVolume();
        }
        return d_cellVolume;
      }

      virtual void updateUnitCell(const Uintah::Matrix3&);

      virtual void updateUnitCell(const SCIRun::Vector&,
                                  const SCIRun::Vector&);

      inline virtual void updateBasisLengths(const SCIRun::Vector& _lengthsIn) {

        d_Lengths = _lengthsIn;
        calculateCellVolume();
        isVolumeCurrent = true;
        calculateInverse();
        isInverseCurrent = true;
        markCellChanged();
      }

      inline virtual void updateBasisAngles(const SCIRun::Vector& _anglesIn) {
        double d_zeroTol = 1.0e-13;
        SCIRun::Vector angleDiff = _anglesIn*MDConstants::degToRad-SCIRun::Vector(MDConstants::PI_Over_2);
        if (abs(angleDiff[0]) > d_zeroTol || abs(angleDiff[1]) > d_zeroTol || abs(angleDiff[2]) > d_zeroTol) {
          throw InvalidState("Error:  Attempt to update an orthorhombic coordinate system with cell angles not equal to 90 degrees",
                                     __FILE__,
                                     __LINE__);
        }
      }

      virtual void minimumImageDistance(const SCIRun::Point&,
                                        const SCIRun::Point&,
                                              SCIRun::Vector&) const;

      inline virtual void distance(const SCIRun::Point& _P1,
                                   const SCIRun::Point& _P2,
                                         SCIRun::Vector& _Distance) const {
        _Distance = _P2 - _P1;

      }

    private:
      SCIRun::Vector d_Lengths;
      SCIRun::Vector d_InverseLengths;
      double d_cellVolume;
      bool isVolumeCurrent;
      bool isInverseCurrent;

      inline virtual void calculateCellVolume() {
        d_cellVolume = d_Lengths[0]*d_Lengths[1]*d_Lengths[2];
        isVolumeCurrent = true;
        markCellChanged();
      }

      virtual inline void calculateInverse() {
        d_InverseLengths[0] = 1.0/d_Lengths[0];
        d_InverseLengths[1] = 1.0/d_Lengths[1];
        d_InverseLengths[2] = 1.0/d_Lengths[2];
        markCellChanged();
      }
  };
}



#endif /* ORTHORHOMBICCOORDINATES_H_ */
