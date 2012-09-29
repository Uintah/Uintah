/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <sci_defs/hypre_defs.h>

#include <CCA/Components/Solvers/AMR/HypreDriver.h>
#include <CCA/Components/Solvers/AMR/HypreDriverStruct.h>
#include <CCA/Components/Solvers/AMR/HypreDriverSStruct.h>
#include <CCA/Components/Solvers/MatrixUtil.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/ConvergenceFailure.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/LoadBalancer.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>

using namespace Uintah;
//__________________________________
//  To turn on normal output
//  setenv SCI_DEBUG "HYPRE_DOING_COUT:+"

static DebugStream cout_doing("HYPRE_DOING_COUT", false);
static DebugStream cout_dbg("HYPRE_DBG", false);


void
HypreDriver::makeLinearSystem_CC(const int matl)
{
  throw InternalError("makeLinearSystem is not implemented for CC variables",
                      __FILE__, __LINE__);
}

void
HypreDriver::getSolution_CC(const int matl)
{
  throw InternalError("getSolution is not implemented for CC variables",
                      __FILE__, __LINE__);
}

//__________________________________
bool
HypreDriver::isConvertable(const HypreInterface& to)
  // Is it possible to convert from the current HypreInterface
  // "_interface" to "to" ?
{
  // Add here any known rules of conversion.

  // It does not seem possible to convert from Struct to ParCSR.
  //  if ((_interface == HypreStruct ) && (to == HypreParCSR)) return true;

  if ((_interface == HypreSStruct) && (to == HypreParCSR)) return true;

  return false;
}

//__________________________________

namespace Uintah {
  HypreDriver*
  newHypreDriver(const HypreInterface& interface,
                 const Level* level,
                 const MaterialSet* matlset,
                 const VarLabel* A, Task::WhichDW which_A_dw,
                 const VarLabel* x, bool modifies_x,
                 const VarLabel* b, Task::WhichDW which_b_dw,
                 const VarLabel* guess,
                 Task::WhichDW which_guess_dw,
                 const HypreSolverParams* params,
                 const PatchSet* perProcPatches)
  {
    switch (interface) {
    case HypreStruct: 
      {
        return scinew HypreDriverStruct
          (level, matlset, A, which_A_dw,
           x, modifies_x, b, which_b_dw, guess, 
           which_guess_dw, params, perProcPatches, interface);
      }
    case HypreSStruct:
      {
        return scinew HypreDriverSStruct
          (level, matlset, A, which_A_dw,
           x, modifies_x, b, which_b_dw, guess, 
           which_guess_dw, params, perProcPatches, interface);
      }
    default:
      throw InternalError("Unsupported Hypre Interface: "+interface,
                          __FILE__, __LINE__);
    } // end switch (interface)
  }

 //__________________________________
 // Correspondence between Uintah patch face and hypre BoxSide
 // (left or right).
  BoxSide
  patchFaceSide(const Patch::FaceType& patchFace)

  {
    if (patchFace == Patch::xminus || 
        patchFace == Patch::yminus || 
        patchFace == Patch::zminus) {
      return LeftSide;
    } else if (patchFace == Patch::xplus || 
               patchFace == Patch::yplus || 
               patchFace == Patch::zplus){
      return RightSide;
    } else {
      ostringstream msg;
      msg << "patchFaceSide() called with invalid Patch::FaceType "
          << patchFace;
      throw InternalError(msg.str(),__FILE__, __LINE__);
    }
  }

} // end namespace Uintah
