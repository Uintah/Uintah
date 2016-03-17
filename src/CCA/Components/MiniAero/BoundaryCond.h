/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef __MiniAero_SM_BOUNDARYCOND_H__
#define __MiniAero_SM_BOUNDARYCOND_H__

#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Util/DebugStream.h>



namespace Uintah {
namespace MiniAeroNS {

  static DebugStream BC_dbg("BC_dbg", false);
  static DebugStream BC_CC("BC_CC", false);
  static DebugStream BC_FC("BC_FC", false);
  
  class DataWarehouse;
  
  //__________________________________
  // Main driver method for CCVariables
  template<class T> 
  void setBC(CCVariable<T>& variable, 
             const std::string& desc,
             const Patch* patch,    
             const int mat_id);

  void setGradientBC(
    CCVariable<Vector> & gradient_var_CC,
    constCCVariable<double> & var_CC,
    const std::string & desc,
    const Patch * patch,
    const int mat_id);

  void setGradientBC(
    CCVariable<Matrix3> & gradient_var_CC,
    constCCVariable<Vector> & var_CC,
    const std::string & desc,
    const Patch * patch,
    const int mat_id);

  int setSymmetryBC_CC( const Patch* patch,
                        const Patch::FaceType face,
                        CCVariable<double>& var_CC,               
                        Iterator& bound_ptr);

  int setSymmetryBC_CC( const Patch* patch,
                        const Patch::FaceType face,
                        CCVariable<Vector>& var_CC,               
                        Iterator& bound_ptr);
  
  int numFaceCells(const Patch* patch, 
                   const Patch::FaceIteratorType type,
                   const Patch::FaceType face);

} // End namespace MiniAeroNS
} // End namespace Uintah
#endif
