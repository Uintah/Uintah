/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

      if (dtask->getTask()->getType() != Task::OncePerProc) {
        const Level* level = getLevel(patches);
        msg << "\t  L-" << level->getIndex();
      }
    }
    printf("%s\n", msg.str().c_str());
  }
}

//______________________________________________________________________
// Dout version (moving away from DebugStream)
void
printTask( const Patch       * patch
         ,       Dout        & dbg
         , const std::string & where
         )
{
  if (dbg){
    std::ostringstream msg;
    msg << Uintah::Parallel::getMPIRank()  << " ";
    msg << std::left;
    msg.width(50);
    msg << where << " \tL-"
        << patch->getLevel()->getIndex()
        << " patch " << patch->getGridIndex();
    printf("%s\n", msg.str().c_str());
  }
}

//______________________________________________________________________
// Dout version (moving away from DebugStream)
void
printTask( const PatchSubset * patches
         , const Patch       * patch
         ,       Dout        & dbg
         , const std::string & where
         )
{
  if (dbg) {
    std::ostringstream msg;
    msg << Uintah::Parallel::getMPIRank() << " ";
    msg << std::left;
    msg.width(50);
    msg << where << "  \tL-"
        << getLevel(patches)->getIndex()
        << " patch " << patch->getGridIndex();
    printf("%s\n", msg.str().c_str());
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
        msg << "Rank-" << d_myworld->myrank() << "   ";
        msg << std::left;
        msg.width(10);
        msg << dtask->getTask()->getName();
        msg << "\t Patch-" << firstPatch->getGridIndex();
        msg << "\t L-" << level->getIndex();
        printf("%s\n", msg.str().c_str());
      }
    }
  }
}
//______________________________________________________________________
//
void
printSchedule( const PatchSet    * patches
             ,       Dout        & dbg
             , const std::string & where
             )
{
  if (dbg){
    std::ostringstream mesg;
    mesg << Uintah::Parallel::getMPIRank() << " ";
    mesg << std::left;
    mesg.width(50);
    mesg << where << " L-" << getLevel(patches)->getIndex() << std::endl;
    printf("%s\n", mesg.str().c_str());
  }
}

//______________________________________________________________________
//
void
printSchedule( const LevelP      & level
             ,       Dout        & dbg
             , const std::string & where
             )
{
  if (dbg){
    std::ostringstream mesg;
    mesg << Uintah::Parallel::getMPIRank() << " ";
    mesg << std::left;
    mesg.width(50);
    mesg << where << " L-" << level->getIndex();
    printf("%s\n", mesg.str().c_str());
  }
}

} // End namespace Uintah
