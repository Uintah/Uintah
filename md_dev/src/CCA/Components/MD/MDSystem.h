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
     * @brief Default constructor
     * @param _ewaldBeta The Ewald damping coefficient.
     * @param _volume The initial MD system volume.
     * @param _pressure The initial MD system pressure.
     * @param _temperature The initial MD system temperature.
     * @param _orthorhombic Whether of not the MD system is using orthorhombic coordinates.
     */
    MDSystem(double _ewaldBeta, double _volume, double _pressure, double _temperature, bool _orthorhombic);

    /**
     * @brief
     * @param
     * @return
     */
    inline Matrix3 getCellInverse() const
    {
      return this->cellInverse;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline void setCellInverse(Matrix3 _cellInverse)
    {
      this->cellInverse = _cellInverse;
    }

    /**
     * @brief Returns the damping coefficient for this MD system.
     * @param None
     * @return double The damping coefficient for this MD system.
     */
    inline double getEwaldBeta() const
    {
      return this->ewaldBeta;
    }

    /**
     * @brief Sets the damping coefficient for this MD system.
     * @param _ewaldBeta The new damping coefficient.
     * @return None
     */
    inline void setEwaldBeta(double _ewaldBeta)
    {
      this->ewaldBeta = _ewaldBeta;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline double getVolume() const
    {
      return this->volume;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline void setVolume(double _volume)
    {
      this->volume = _volume;
    }

    /**
     * @brief
     * @param  None
     * @return
     */
    inline bool getPressure() const
    {
      return this->pressure;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline void setPressure(double _pressure)
    {
      this->pressure = _pressure;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline bool getTemperature() const
    {
      return this->temperature;
    }

    /**
     * @brief Sets the temperature for this MD system.
     * @param
     * @return
     */
    inline void setTemperature(double _temperature)
    {
      this->temperature = _temperature;
    }


    /**
     * @brief
     * @param None
     * @return
     */
    inline bool isOrthorhombic() const
    {
      return this->orthorhombic;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline void setOrthorhombic(bool _value)
    {
      this->orthorhombic = _value;
    }

  private:

    Matrix3 cellInverse;  //!<
    double ewaldBeta;     //!< The Ewald damping coefficient
    double volume;        //!< Total MD system volume
    double pressure;      //!< Total MD system pressure
    double temperature;   //!< Total MD system temperature
    bool orthorhombic;    //!< Whether or not the MD system is using orthorhombic coordinates

};

}  // End namespace Uintah

#endif
