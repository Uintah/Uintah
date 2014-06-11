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
 * coordinateSystemInterface.h
 *
 *  Created on: May 8, 2014
 *      Author: jbhooper
 */

#ifndef COORDINATESYSTEM_H_
#define COORDINATESYSTEM_H_

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Math/Matrix3.h>

#include<CCA/Components/MD/MDUtil.h>

namespace Uintah {


  class CoordinateSystem {
    public:

      enum principleDirection {X=1, Y=2, XY = 3, Z=4, XZ = 5, YZ = 6, XYZ = 7};
      static const double PI_Over_2;

               CoordinateSystem(const SCIRun::IntVector&,
                                const SCIRun::IntVector&);

      virtual ~CoordinateSystem() { };

      inline bool queryCellChanged() const {
        return d_cellChanged;
      }

      inline void clearCellChanged() {
        d_cellChanged = false;
      }

      inline void markCellChanged() {
        d_cellChanged = true;
      }

      inline SCIRun::IntVector getCellExtent() const {
        return d_totalCellExtent;
      }

      inline double periodicX() const {
        int result = 0.0;
        if (f_periodicX) result = 1.0;
        return result;
      }

      inline double periodicY() const {
        int result = 0.0;
        if (f_periodicY) result = 1.0;
        return result;
      }

      inline double periodicZ() const {
        int result = 0.0;
        if (f_periodicZ) result = 1.0;
        return result;
      }

      inline std::string periodic() {
        std::string direction = "";
        if (f_periodicX) direction += "X";
        if (f_periodicY) direction += "Y";
        if (f_periodicZ) direction += "Z";
        return direction;
      }

      inline bool periodic(const std::string& boundaryString) {
        bool result = true;
        if (boundaryString.find("x") || boundaryString.find("X")) result = result && f_periodicX;
        if (boundaryString.find("y") || boundaryString.find("Y")) result = result && f_periodicY;
        if (boundaryString.find("z") || boundaryString.find("Z")) result = result && f_periodicZ;
        return result;
      }

      virtual void toReduced(const SCIRun::Vector&,
                                   SCIRun::Vector&) const = 0;


      virtual void toReduced(const double radius, SCIRun::Vector& _out) const {
        SCIRun::Vector sphericalVector(radius);
        toReduced(sphericalVector, _out);
      }

      inline virtual SCIRun::Vector toReduced(const SCIRun::Vector& _in) const {
        SCIRun::Vector temp;
        toReduced(_in, temp);
        return temp;
      }



      virtual void toCartesian(const SCIRun::Vector&,
                                     SCIRun::Vector&) const = 0;

      inline virtual SCIRun::Vector toCartesian(const SCIRun::Vector& _in) const {
        SCIRun::Vector temp;
        toCartesian(_in, temp);
        return temp;
      }

      virtual void            getUnitCell(Uintah::Matrix3&) = 0;

      virtual Uintah::Matrix3 getUnitCell() {
        Uintah::Matrix3 temp;
        getUnitCell(temp);
        return temp;
      }

      virtual void            getInverseCell(Uintah::Matrix3&) = 0;

      inline Uintah::Matrix3 getInverseCell() {
        Uintah::Matrix3 temp;
        getInverseCell(temp);
        return temp;
      }

      virtual void            getBasisLengths(SCIRun::Vector&) = 0;

      virtual SCIRun::Vector  getBasisLengths() {
        SCIRun::Vector temp;
        getBasisLengths(temp);
        return temp;
      }

      virtual void            getBasisAngles(SCIRun::Vector&) = 0;

      virtual SCIRun::Vector  getBasisAngles() {
        SCIRun::Vector temp;
        getBasisAngles(temp);
        return temp;
      }

      virtual double getCellVolume() = 0;

      virtual void updateUnitCell(const Uintah::Matrix3& matrixIn) = 0;
      virtual void updateUnitCell(const SCIRun::Vector& lengthsIn,
                                  const SCIRun::Vector& anglesIn = SCIRun::Vector(PI_Over_2)) = 0;
      virtual void updateBasisLengths(const SCIRun::Vector& lengthsIn) = 0;
      virtual void updateBasisAngles(const SCIRun::Vector& anglesIn) = 0;



      virtual void           minimumImageDistance(const SCIRun::Point&,
                                                  const SCIRun::Point&,
                                                        SCIRun::Vector&) const = 0;

      virtual SCIRun::Vector minimumImageDistance(const SCIRun::Point& _P1,
                                                  const SCIRun::Point& _P2) const {
        SCIRun::Vector temp;
        minimumImageDistance(_P1, _P2, temp);
        return temp;
      }

      virtual void           distance(const SCIRun::Point&,
                                      const SCIRun::Point&,
                                            SCIRun::Vector&) const = 0;

      virtual SCIRun::Vector distance(const SCIRun::Point& _P1,
                                      const SCIRun::Point& _P2) const {
        SCIRun::Vector temp;
        distance(_P1, _P2, temp);
        return temp;
      }

    private:
      bool d_cellChanged;

      // Flag values
      bool f_periodicX;
      bool f_periodicY;
      bool f_periodicZ;

      SCIRun::IntVector d_totalCellExtent;
      SCIRun::IntVector d_periodic;

      virtual void calculateCellVolume() = 0;
      virtual void calculateInverse() = 0;

  };
}



#endif /* COORDINATESYSTEM_H_ */
