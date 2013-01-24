/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef UINTAH_MD_MDSYSTEM_H
#define UINTAH_MD_MDSYSTEM_H

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Variables/Array3.h>

#include <vector>

namespace Uintah {

class ProcessorGroup;
class MD;

using SCIRun::Vector;
using SCIRun::IntVector;

/**
 *  @class MDSystem
 *  @ingroup MD
 *  @author Alan Humphrey and Justin Hooper
 *  @date   December, 2012
 *
 *  @brief
 *
 *  @param
 */
class MDSystem {

  public:

    /**
     * @brief Default constructor
     * @param
     */
    MDSystem();

    /**
     * @brief Default destructor
     * @param
     */
    ~MDSystem();

    /**
     * @brief 4 argument constructor
     * @param _volume The initial MD system volume.
     * @param _pressure The initial MD system pressure.
     * @param _temperature The initial MD system temperature.
     * @param _orthorhombic Whether of not the MD system is using orthorhombic coordinates.
     */
    MDSystem(const ProcessorGroup* pg);

    /**
     * @brief
     * @param
     * @return
     */
    inline Matrix3 getCellInverse() const
    {
      return this->d_cellInverse;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline double getVolume() const
    {
      return this->d_unitCell;
    }

    /**
     * @brief
     * @param  None
     * @return
     */
    inline bool getPressure() const
    {
      return this->d_pressure;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline bool getTemperature() const
    {
      return this->d_temperature;
    }

    /**
     * @brief
     * @param None
     * @return
     */
    inline bool isOrthorhombic() const
    {
      return this->d_orthorhombic;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline bool newBox() const
    {
      return this->d_changeBox;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline void changeBox(bool value)
    {
      this->d_changeBox = value;
    }

    friend class MD;

  private:

    double d_unitCell;      //!< Total MD system unit cell volume
    double d_pressure;      //!< Total MD system pressure
    double d_temperature;   //!< Total MD system temperature
    bool   d_orthorhombic;  //!< Whether or not the MD system is using orthorhombic coordinates
    bool   d_changeBox;     //!<

    Matrix3 d_cellInverse;  //!<
    const ProcessorGroup* d_myworld;  //!< Uintah processor group

    MDSystem(const MDSystem& system);
    MDSystem& operator=(const MDSystem& system);

};

}  // End namespace Uintah

#endif
