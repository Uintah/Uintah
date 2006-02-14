/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

/*
 *  DTMessageTag.h
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */


#ifndef CORE_CCA_COMM_DT_DTMESSAGETAG_H
#define CORE_CCA_COMM_DT_DTMESSAGETAG_H


#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <Core/CCA/Comm/DT/DTAddress.h>
#include <Core/Thread/Mutex.h>

////////////////////////////////////////////
//  This class defines a message tag, which
//  essentially 8-byte unsigned integer. It 
//  will not does not overflow in thousands
//  of years assuming 1 message passing per 
//  micro second.
/////////////////////////////////////////////


namespace SCIRun {
  class DTMessageTag{
  public:
    /////////////////////////////////////////
    // constructor: the message counter is 
    // initialized
    DTMessageTag();

    /////////////////////////////////////////
    // constructor: the message counter is 
    // initialized with the given (hi, lo) pair.
    DTMessageTag(unsigned int hi, unsigned int lo);

    /////////////////////////////////////////
    // destructor
    ~DTMessageTag();

    bool operator<(const DTMessageTag &tag) const;
    bool operator==(const DTMessageTag &tag) const;

    DTMessageTag defaultTag();

    ////////////////////////////////////////
    // this method increment the current tag
    // and return a copy of itself. This is
    // method is thread-safe.
    DTMessageTag nextTag();
    unsigned int hi, lo;

  private:
    static Mutex counter_mutex;
  };
}//namespace SCIRun

#endif

