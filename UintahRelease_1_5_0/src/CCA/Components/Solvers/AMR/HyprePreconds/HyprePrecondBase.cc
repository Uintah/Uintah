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

//--------------------------------------------------------------------------
// File: HyprePrecondBase.cc
// 
// A generic Hypre preconditioner driver that checks whether the precond
// can work with the input interface. The actual precond setup/destroy is
// done in the classes derived from HyprePrecondBase.
//--------------------------------------------------------------------------
#include <CCA/Components/Solvers/AMR/HyprePreconds/HyprePrecondBase.h>
#include <CCA/Components/Solvers/HypreSolverParams.h>
#include <CCA/Components/Solvers/AMR/HyprePreconds/HyprePrecondSMG.h>
#include <CCA/Components/Solvers/AMR/HyprePreconds/HyprePrecondPFMG.h>
#include <CCA/Components/Solvers/AMR/HyprePreconds/HyprePrecondSparseMSG.h>
#include <CCA/Components/Solvers/AMR/HyprePreconds/HyprePrecondJacobi.h>
#include <CCA/Components/Solvers/AMR/HyprePreconds/HyprePrecondDiagonal.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/Time.h>
#include <Core/Util/DebugStream.h>
#include <iomanip>

using namespace Uintah;

HyprePrecondBase::HyprePrecondBase(const Priorities& priority) :
  _priority(priority)
{
  if (_priority.size() < 1) {
    throw InternalError("Preconditioner created without interface "
                        "priorities",
                        __FILE__, __LINE__);
  }
}

namespace Uintah {

  HyprePrecondBase*
  newHyprePrecond(const PrecondType& precondType)
    // Create a new preconditioner object of specific precond type
    // "precondType" but a generic preconditioner pointer type.
  {
    switch (precondType) {
    case PrecondNA:        return 0;  // no preconditioner
    case PrecondSMG:       return scinew HyprePrecondSMG();
    case PrecondPFMG:      return scinew HyprePrecondPFMG();
    case PrecondSparseMSG: return scinew HyprePrecondSparseMSG();
    case PrecondJacobi:    return scinew HyprePrecondJacobi();
    case PrecondDiagonal:  return scinew HyprePrecondDiagonal();
    case PrecondAMG:       break; // Not implemented yet
    case PrecondFAC:       break; // Not implemented yet
    default:
      throw InternalError("Unknown preconditionertype in newHyprePrecond: "
                          +precondType, __FILE__, __LINE__);
    }
    throw InternalError("Preconditioner not yet implemented in newHyprePrecond: "
                        +precondType, __FILE__, __LINE__);
  } 
  
/* Determine preconditioner type from title */
  PrecondType
  getPrecondType(const string& precondTitle)
  {
    if ((precondTitle == "None") ||
        (precondTitle == "none")) {
      return PrecondNA;
    } else if ((precondTitle == "SMG") ||
               (precondTitle == "smg")) {
      return PrecondSMG;
    } else if ((precondTitle == "PFMG") ||
               (precondTitle == "pfmg")) {
      return PrecondPFMG;
    } else if ((precondTitle == "SparseMSG") ||
               (precondTitle == "sparsemsg")) {
      return PrecondSparseMSG;
    } else if ((precondTitle == "Jacobi") ||
               (precondTitle == "jacobi")) {
      return PrecondJacobi;
    } else if ((precondTitle == "Diagonal") ||
               (precondTitle == "diagonal")) {
      return PrecondDiagonal;
    } else if ((precondTitle == "AMG") ||
               (precondTitle == "amg") ||
               (precondTitle == "BoomerAMG") ||
               (precondTitle == "boomeramg")) {
      return PrecondAMG;
    } else if ((precondTitle == "FAC") ||
               (precondTitle == "fac")) {
      return PrecondFAC;
    } else {
      throw InternalError("Unknown preconditionertype: "+precondTitle,
                          __FILE__, __LINE__);
    } 
  }
}
