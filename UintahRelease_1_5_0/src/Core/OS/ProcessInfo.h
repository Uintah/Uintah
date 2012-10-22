/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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


/*
 *  ProcessInfo.h:
 *
 *  Written by:
 *   Author: Randy Jones
 *   Department of Computer Science
 *   University of Utah
 *   Date: 2004/02/05
 *
 */

#ifndef Core_OS_ProcessInfo_h
#define Core_OS_ProcessInfo_h 1

#include <Core/OS/share.h>
namespace SCIRun {


  class SCISHARE ProcessInfo {

  public:

    enum /*info_type*/ {
      MEM_SIZE,
      MEM_RSS
    };

    static bool          IsSupported       ( int info_type );
    static unsigned long GetInfo           ( int info_type );

    static unsigned long GetMemoryUsed     ( void ) { return GetInfo( MEM_SIZE ); }
    static unsigned long GetMemoryResident ( void ) { return GetInfo( MEM_RSS  ); }

  private:

    ProcessInfo  ( void ) {}
    ~ProcessInfo ( void ) {}

  }; // class ProcessInfo {


} // namespace SCIRun {


#endif // #ifndef Core_OS_ProcessInfo_h
