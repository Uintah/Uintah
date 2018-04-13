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

#ifndef Uintah_CORE_GRID_DBGOUTPUT_H
#define Uintah_CORE_GRID_DBGOUTPUT_H

#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/DOUT.hpp>

#include <string>

namespace Uintah {

void printSchedule( const PatchSet    * patches
                  ,       DebugStream & dbg
                  , const std::string & where
                  );

void printSchedule( const LevelP      & level
                  ,       DebugStream & dbg
                  , const std::string & where
                  );

void printTask( const PatchSubset * patches
              , const Patch       * patch
              ,       DebugStream & dbg
              , const std::string & where
              );

void printTask( const PatchSubset * patches
              ,       DebugStream & dbg
              , const std::string & where
              );

void printTask( const Patch       * patch
              ,       DebugStream & dbg
              , const std::string & where
              );

void printTask(       DebugStream & dbg
              , const std::string & where
              );


void printTask( const Patch * patch,       
                Dout & out,
                const std::string & where,
                const int timestep,
                const int material,
                const std::string var);

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

// Output the task name and the level it's executing on and each of the patches
void printTask( Dout         & out
              , DetailedTask * dtask
              );

void printTask( const Patch       * patch
              ,       Dout        & out
              , const std::string & where
              );

void printTask( const PatchSubset * patches
              , const Patch       * patch
              ,       Dout        & out
              , const std::string & where
              );

// Output the task name and the level it's executing on only first patch of that level
void printTaskLevels( const ProcessorGroup * d_myworld
                    ,       Dout           & out
                    ,       DetailedTask   * dtask
                    );

void printSchedule( const LevelP      & level
                  ,       Dout        & out
                  , const std::string & where
                  );

void printSchedule( const PatchSet    * patches
                  ,       Dout        & out
                  , const std::string & where
                  );

} // End namespace Uintah

#endif
