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

#ifndef Uintah_CORE_GRID_DBGOUTPUT_H
#define Uintah_CORE_GRID_DBGOUTPUT_H

#include <Core/Util/DebugStream.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <string>

namespace Uintah{

void printSchedule( const PatchSet       * patches,
                    SCIRun::DebugStream  & dbg,
                    const std::string    & where );

void printSchedule( const LevelP        & level,
                    SCIRun::DebugStream & dbg,
                    const string        & where );

void printTask( const PatchSubset   * patches,
                const Patch         * patch,
                SCIRun::DebugStream & dbg,
                const string        & where );
                
void printTask( const PatchSubset   * patches,
                SCIRun::DebugStream & dbg,
                const string        & where );

void printTask( const Patch         * patch,
                SCIRun::DebugStream & dbg,
                const string        & where );
                
void printTask( SCIRun::DebugStream & dbg,
                const string        & where );

} // End namespace Uintah

#endif
