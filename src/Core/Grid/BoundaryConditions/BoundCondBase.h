/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#ifndef UINTAH_GRID_BoundCondBase_H
#define UINTAH_GRID_BoundCondBase_H

#include <string>

namespace Uintah {
   
/**************************************

CLASS
   BoundCondBase
   
   
GENERAL INFORMATION

   BoundCondBase.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   BoundCondBase

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class BoundCondBase  {
  public:

    /**
     *  \enum   BoundCondValueTypeEnum
     *  \author Tony Saad
     *  \date   August 29, 2013
     *
     *  \brief An enum that lists the datatypes associated with the values specified for boundary conditions.
     */
    enum BoundCondValueTypeEnum
    {
      INT_TYPE,
      DOUBLE_TYPE,
      VECTOR_TYPE,
      STRING_TYPE,
      UNKNOWN_TYPE
    };

    BoundCondBase() {};
    virtual ~BoundCondBase() {};
    virtual BoundCondBase* clone() = 0;
    const std::string getBCVariable() const { return d_variable; };
    const std::string getBCType() const { return d_type; };
    const std::string getBCFaceName() const { return d_face_label; };
    
    /**
     *  \author Tony Saad
     *  \date   August 29, 2013
     *  \brief  Returns a BoundCondValueTypeEnum that designates the data type associated with the 
     *          value of this boundary condition.
     */
    BoundCondValueTypeEnum getValueType() const { return d_value_type; }
    
  protected:
    std::string d_variable;              // Pressure, Density, etc
    std::string d_type;                  // Dirichlet, Neumann, etc
    std::string d_face_label;            // holds the user specified name of the bc face: left-wall, ox-inlet,...
    BoundCondValueTypeEnum d_value_type; // int, double, string, vector, unknown
  };
} // End namespace Uintah

#endif
