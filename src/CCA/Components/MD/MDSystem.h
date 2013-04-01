/*
 * The MIT License
 *
 * Copyright (c) 1997-2013 The University of Utah
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

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Math/Matrix3.h>

namespace Uintah {

using namespace SCIRun;

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
     * @brief
     * @param
     */
    MDSystem(ProblemSpecP& ps);

    /**
     * @brief
     * @param
     * @return
     */
    inline double getVolume() const
    {
      return d_volume;
    }

    /**
     * @brief
     * @param  None
     * @return
     */
    inline Vector getPressure() const
    {
      return d_pressure;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline double getTemperature() const
    {
      return d_temperature;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline int getNumGhostCells() const
    {
      return d_numGhostCells;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline int getMaxIterations() const
    {
      return d_maxIterations;
    }

    /**
     * @brief
     * @param None
     * @return
     */
    inline bool isOrthorhombic() const
    {
      return d_orthorhombic;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline bool newBox() const
    {
      return d_newBox;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline void setNewBox(bool value)
    {
      d_newBox = value;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline Matrix3 getUnitCell() const
    {
      return d_unitCell;
    }

    /**
     * @brief
     * @param
     * @return
     */
    inline Matrix3 getInverseCell() const
    {
      return d_inverseCell;
    }

    /**
     * @brief
     * @param
     * @return
     */
    double getCellVolume() const
    {
      return d_cellVolume;
    }

  private:

    double d_volume;            //!< Total MD system unit cell volume
    Vector d_pressure;          //!< Total MD system pressure
    double d_temperature;       //!< Total MD system temperature
    int d_numGhostCells;        //!< Number of ghost cells used, a function of cutoffRadius and cell size
    int d_maxIterations;        //!<
    bool d_orthorhombic;        //!< Whether or not the MD system is using orthorhombic coordinates
    bool d_newBox;           //!< Whether or not the system size has changed... create a new box

    Matrix3 d_unitCell;         //!< MD system unit cell
    Matrix3 d_inverseCell;      //!< MD system inverse unit cell
    double d_cellVolume;       //!< Cell volume; calculate internally, return at request for efficiency

    MDSystem(const MDSystem& system);
    MDSystem& operator=(const MDSystem& system);

    void calcCellVolume();
};

}  // End namespace Uintah

#endif
