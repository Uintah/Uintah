/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <CCA/Components/Schedulers/DetailedTask.h>

#include <Core/Grid/DbgOutput.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/DOUT.hpp>

namespace Uintah {

//______________________________________________________________________
//
void
printSchedule( const PatchSet    * patches
             ,       DebugStream & dbg
             , const std::string & where
             )
{
  if (dbg.active()){
    dbg << Uintah::Parallel::getMPIRank() << " ";
    dbg << std::left;
    dbg.width(50);
    dbg << where << "L-" << getLevel(patches)->getIndex() << std::endl;
  }
}

//______________________________________________________________________
//
void
printSchedule( const LevelP      & level
             ,       DebugStream & dbg
             , const std::string & where
             )
{
  if (dbg.active()){
    dbg << Uintah::Parallel::getMPIRank() << " ";
    dbg << std::left;
    dbg.width(50);
    dbg << where << "L-" << level->getIndex() << std::endl;
  }
}

//______________________________________________________________________
//
void
printTask( const PatchSubset * patches
         , const Patch       * patch
         ,       DebugStream & dbg
         , const std::string & where
         )
{
  if (dbg.active()){
    dbg << Uintah::Parallel::getMPIRank() << " ";
    dbg << std::left;
    dbg.width(50);
    dbg << where << "  \tL-"
        << getLevel(patches)->getIndex()
        << " patch " << patch->getGridIndex()<< std::endl;
  }
}

//______________________________________________________________________
//
void
printTask( const PatchSubset * patches
         ,       DebugStream & dbg
         , const std::string & where
         )
{
  if (dbg.active()){
    dbg << Uintah::Parallel::getMPIRank() << " ";
    dbg << std::left;
    dbg.width(50);
    dbg << where << "  \tL-"
        << getLevel(patches)->getIndex()
        << " patches " << *patches << std::endl;
  }
}

//______________________________________________________________________
//
void
printTask( const Patch       * patch
         ,       DebugStream & dbg
         , const std::string & where
         )
{
  if (dbg.active()){
    dbg << Uintah::Parallel::getMPIRank()  << " ";
    dbg << std::left;
    dbg.width(50);
    dbg << where << " \tL-"
        << patch->getLevel()->getIndex()
        << " patch " << patch->getGridIndex()<< std::endl;
  }
}

//______________________________________________________________________
//
void
printTask(       DebugStream & dbg
         , const std::string & where
         )
{
  if (dbg.active()){
    dbg << Uintah::Parallel::getMPIRank()  << " ";
    dbg << std::left;
    dbg.width(50);
    dbg << where << " \tAll Patches" << std::endl;
  }
}



// ------------------------------------------------------------------------------------------------
// APH - 07/14/17
// ------------------------------------------------------------------------------------------------
// Dout (class) versions of the above (moving away from DebugStream)
//
// Dout is an extremely lightweight way to provide the same functionality as DebugStream,
// but in a fully lock-free way for both multiple threads and MPI ranks. DOut also does not
// inherit from from the standard library (std::ostream specifically) as DebugStream does.
// The DOUT variadic macro is then used, which is printf-based. By the POSIX standard, printf
// must "behave" like it acquired a lock. The Dout class also defines an explicit bool() operator
// for checking state, e.g. active or inactive.
// ------------------------------------------------------------------------------------------------

//______________________________________________________________________
// Output the task name and the level it's executing on and each of the patches
void
printTask( Dout         & out
         , DetailedTask * dtask
         )
{
  if (out) {
    std::ostringstream msg;
    msg << std::left;
    msg.width(70);
    msg << dtask->getTask()->getName();
    if (dtask->getPatches()) {

      msg << " \t on patches ";
      const PatchSubset* patches = dtask->getPatches();
      for (int p = 0; p < patches->size(); p++) {
        if (p != 0) {
          msg << ", ";
        }
        msg << patches->get(p)->getID();
      }

      if (dtask->getTask()->getType() != Task::OncePerProc && dtask->getTask()->getType() != Task::Hypre) {
        const Level* level = getLevel(patches);
        msg << "\t  L-" << level->getIndex();
      }
    }
    DOUT(out, msg.str());
  }
}

//______________________________________________________________________
// Dout version (moving away from DebugStream)
void
printTask( const Patch       * patch
         ,       Dout        & out
         , const std::string & where
         )
{
  if (out) {
    std::ostringstream msg;
    msg << Uintah::Parallel::getMPIRank()  << " ";
    msg << std::left;
    msg.width(50);
    msg << where << " \tL-"
        << patch->getLevel()->getIndex()
        << " patch " << patch->getGridIndex();
    DOUT(out, msg.str());
  }
}

//______________________________________________________________________
// Dout version (moving away from DebugStream)
void
printTask( const PatchSubset * patches
         , const Patch       * patch
         ,       Dout        & out
         , const std::string & where
         )
{
  if (out) {
    std::ostringstream msg;
    msg << Uintah::Parallel::getMPIRank() << " ";
    msg << std::left;
    msg.width(50);
    msg << where << "  \tL-"
        << getLevel(patches)->getIndex()
        << " patch " << patch->getGridIndex();
    DOUT(out, msg.str());
  }
}

//______________________________________________________________________
//  Output the task name and the level it's executing on only first patch of that level
void
printTaskLevels( const ProcessorGroup * d_myworld
               ,       Dout           & out
               ,       DetailedTask   * dtask
               )
{
  if (out) {
    if (dtask->getPatches()) {
      const PatchSubset* taskPatches = dtask->getPatches();
      const Level* level = getLevel(taskPatches);
      const Patch* firstPatch = level->getPatch(0);
      if (taskPatches->contains(firstPatch)) {
        std::ostringstream msg;
        msg << "Rank-" << d_myworld->myRank() << "   ";
        msg << std::left;
        msg.width(10);
        msg << dtask->getTask()->getName();
        msg << "\t Patch-" << firstPatch->getGridIndex();
        msg << "\t L-" << level->getIndex();
        DOUT(out, msg.str());
      }
    }
  }
}

//______________________________________________________________________
//
void
printSchedule( const PatchSet    * patches
             ,       Dout        & out
             , const std::string & where
             )
{
  if (out) {
    std::ostringstream msg;
    msg << Uintah::Parallel::getMPIRank() << " ";
    msg << std::left;
    msg.width(50);
    msg << where << " L-" << getLevel(patches)->getIndex();
    DOUT(out, msg.str());
  }
}

//______________________________________________________________________
//
void
printSchedule( const LevelP      & level
             ,       Dout        & out
             , const std::string & where
             )
{
  if (out) {
    std::ostringstream msg;
    msg << Uintah::Parallel::getMPIRank() << " ";
    msg << std::left;
    msg.width(50);
    msg << where << " L-" << level->getIndex();
    DOUT(out, msg.str());
  }
}

//______________________________________________________________________
// 

void printTask( const Patch * patch,       
                Dout & out,
                const std::string & where,
                const int timestep,
                const int material,
                const std::string varName)
{
  std::ostringstream msg;
  msg << std::left;
  msg.width(50);
  msg << where << std::right << " timestep: " << timestep 
      << " matl: " << material << " Var: " << varName;
  
  printTask( patch, out, msg.str());
}

} // End namespace Uintah
