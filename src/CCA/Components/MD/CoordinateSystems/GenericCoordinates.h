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
 * GenericCoordinates.h
 *
 *  Created on: May 12, 2014
 *      Author: jbhooper
 */

#ifndef GENERICCOORDINATES_H_
#define GENERICCOORDINATES_H_

#include <CCA/Components/MD/CoordinateSystems/CoordinateSystem.h>

#include <Core/Exceptions/InternalError.h>

namespace Uintah {
  class GenericCoordinates: public CoordinateSystem {
    public:
       GenericCoordinates(const SCIRun::IntVector&,
                          const SCIRun::IntVector&,
                          const Uintah::Matrix3&);

       GenericCoordinates(const SCIRun::IntVector&,
                          const SCIRun::IntVector&,
                          const SCIRun::Vector&,
                          const SCIRun::Vector&);

      ~GenericCoordinates();

      virtual inline void toReduced(const SCIRun::Vector& _In,
                                          SCIRun::Vector& _Out) const {

        _Out = d_inverseCell * _In;

      }

      virtual inline void toCartesian(const SCIRun::Vector& _In,
                                            SCIRun::Vector& _Out) const {

        _Out = d_unitCell * _In;

      }

      virtual inline void getUnitCell(Uintah::Matrix3& _In) {
        _In = d_unitCell;
      }

      virtual void getInverseCell(Uintah::Matrix3& _In) {
        if (!isInverseCurrent) {
          calculateInverse();
        }
        _In = d_inverseCell;
      }

      inline virtual void getBasisLengths(SCIRun::Vector& _In) {
        if (!areBasisLengthsCurrent) {
          calculateBasisLengths();
        }
        _In = d_basisLengths;
      }

      inline virtual void getBasisAngles(SCIRun::Vector& _In) {
        if (!areBasisAnglesCurrent) {
          calculateBasisAngles();
        }
        _In = d_basisAngles;
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
        d_basisLengths = _lengthsIn;
        areBasisLengthsCurrent = true;
        isUnitCellCurrent = false;
        isInverseCurrent = false;
        isVolumeCurrent = false;
      }

      inline virtual void updateBasisAngles(const SCIRun::Vector& _anglesIn) {
        d_basisAngles = _anglesIn;
        areBasisAnglesCurrent = true;
        isUnitCellCurrent = false;
        isInverseCurrent = false;
        isVolumeCurrent = false;
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
      SCIRun::Vector d_basisLengths;
      SCIRun::Vector d_basisAngles;

      Uintah::Matrix3 d_unitCell;
      Uintah::Matrix3 d_inverseCell;

      double d_cellVolume;

      bool isVolumeCurrent;
      bool isUnitCellCurrent;
      bool isInverseCurrent;

      bool areBasisLengthsCurrent;
      bool areBasisAnglesCurrent;

      inline void calculateBasisLengths() {
        SCIRun::Vector X[3];
        if (!isUnitCellCurrent) { // Nothing to calculate basis lengths from
          throw SCIRun::InternalError("Update to unit cell basis lengths requested from non-updated unit cell.",
                                      __FILE__,
                                      __LINE__);
        }
        else {
            for (size_t Index=0; Index < 3; ++Index) {
              d_unitCell.getColumn(Index,X[Index]);
              d_basisLengths[Index] = X[Index].length();
            }
            areBasisLengthsCurrent = true;
          }
      }

      inline void calculateBasisAngles() {
        SCIRun::Vector A, B, C;
        if (!isUnitCellCurrent) { // Nothing to calculate basis lengths from
          throw SCIRun::InternalError("Update to unit cell basis angles requested from non-updated unit cell.",
                                      __FILE__,
                                      __LINE__);
        }
        else {
          d_unitCell.getColumn(0,A);
          d_unitCell.getColumn(0,B);
          d_unitCell.getColumn(0,C);
          if (!areBasisLengthsCurrent) {
            calculateBasisLengths();
          }
          d_basisAngles[0]=acos(Dot(B,C)/(d_basisLengths[1]*d_basisLengths[2]));
          d_basisAngles[1]=acos(Dot(A,C)/(d_basisLengths[0]*d_basisLengths[2]));
          d_basisAngles[2]=acos(Dot(A,B)/(d_basisLengths[0]*d_basisLengths[1]));

          areBasisAnglesCurrent=true;
        }
      }

      inline void cellFromLengthsAndAngles() {
        if (areBasisLengthsCurrent && areBasisAnglesCurrent) {
          double a = d_basisLengths[0];
          double b = d_basisLengths[1];
          double c = d_basisLengths[2];

          double alpha = d_basisAngles[0];
          double beta  = d_basisAngles[1];
          double gamma = d_basisAngles[2];

          d_unitCell.set(0.0);

          d_unitCell(0,0) = a;
          d_unitCell(0,1) = b*cos(gamma);
          d_unitCell(1,1) = b*sin(gamma);
          d_unitCell(0,2) = c*cos(beta);
          d_unitCell(1,2) = c*(cos(alpha) - cos(beta)*cos(gamma))/sin(gamma);
          d_unitCell(2,2) = sqrt(c*c - d_unitCell(0,2)*d_unitCell(0,2) - d_unitCell(1,2)*d_unitCell(1,2));

          isUnitCellCurrent = true;
        }
        else {
          throw SCIRun::InternalError("Cannot update unit cell from non-current lengths and angles",
                                        __FILE__,
                                        __LINE__);
        }
      }

      inline virtual void calculateCellVolume() {
        if (isUnitCellCurrent) {
          SCIRun::Vector A, B, C;
          d_unitCell.getColumn(0,A);
          d_unitCell.getColumn(1,B);
          d_unitCell.getColumn(2,C);

          d_cellVolume = abs(Dot(A,Cross(B,C)));
          isVolumeCurrent = true;
        }
        else {
          throw SCIRun::InternalError("Cannot update cell volume from non-current unit cell",
                                        __FILE__,
                                        __LINE__);
        }
      }

      virtual inline void calculateInverse() {
        if (isUnitCellCurrent) {
          d_inverseCell = d_unitCell.Inverse();
          isInverseCurrent = true;
        }
        else if (areBasisLengthsCurrent && areBasisAnglesCurrent) {
          cellFromLengthsAndAngles();
          d_inverseCell = d_unitCell.Inverse();
          isInverseCurrent = true;
        }
        else {
          throw SCIRun::InternalError("Cannot calculate cell inverse:  No current unit cell data",
                                        __FILE__,
                                        __LINE__);
        }
      }
  };
}




#endif /* GENERICCOORDINATES_H_ */
