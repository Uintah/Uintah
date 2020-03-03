/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef UINTAH_GRID_BCData_H
#define UINTAH_GRID_BCData_H

#include <Core/Grid/BoundaryConditions/BoundCondBase.h>
#include <Core/Util/Handle.h>

#include <string>
#include <vector>

namespace Uintah {

/**************************************

CLASS
   BCData
   
   
GENERAL INFORMATION

   BCData.h

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

  class BCData  {
  public:
    BCData();
    ~BCData();
    BCData(const BCData&);
    BCData& operator=(const BCData&);
    bool operator==(const BCData&);
    bool operator<(const BCData&) const;
    void setBCValues(BoundCondBase* bc);

    /*
     *  \brief Clones the boundary conditions associated with this boundary. This is a deep copy
     and the user is responsible for deleting the newly created variable.
     */
    const BoundCondBase* cloneBCValues(const std::string& type) const;
    
    /*
     *  \author Tony Saad
     *  \date   August 2014
     *  \brief Returns a const pointer to the boundary conditions associated with this boundary.
     This is a lightweight access to the BCValues and does not perform a deep copy.
     */
    const BoundCondBase* getBCValues(const std::string& type) const;
    
    /**
     *  \author Tony Saad
     *  \date   August 29, 2013
     *  \brief  Returns a reference to the vector of boundary conditions (BoundCondBase) attached 
     to this BCData. Recall that BCDataArray stores a collection of BCGeometries. Each BCGeomBase
     has BCData associated with it. The BCData stores all the boundary specification via BoundCond
     and BoundCondBase. This boundary specification corresponds to all the spec within a given
     <Face> xml tag in the input file.
     */
    const std::vector<BoundCondBase*>& getBCData() const;
    
    void print() const;
    bool find(const std::string& type) const;
    bool find(const std::string& bc_type,const std::string& bc_variable) const;
    void combine(BCData& from);
    
   private:
      // The map is for the name of the
    // bc type and then the actual bc data, i.e. 
    // "Velocity", VelocityBoundCond
    std::vector<BoundCondBase*>  d_BCData;
  };
} // End namespace Uintah

#endif


