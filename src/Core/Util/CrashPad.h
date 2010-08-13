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
 *  CrashPad: Utility class for outputing debug messages after crashing
 *
 *  Written by:
 *   Author: Justin Lutijens
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef Core_Util_CrashPad_h
#define Core_Util_CrashPad_h

#include <ostream>
#include <string>
#include <vector>

namespace Uintah {
/**************************************

 CLASS
 CrashPad

 KEYWORDS
 Debug, CrashPad

 DESCRIPTION
 Utility class to output error messages after
 a crash.  This is useful when a debugger is
 unavailable or when using a large number of 
 processors.

****************************************/
class CrashPad {
public:
	    static void addMessage(std::string msg) {d_messages.push_back(msg);}
      static void clearMessages() {d_messages.clear();}
      static void printMessages(std::ostream &out)
      {
        for( std::vector<std::string>::iterator iter=d_messages.begin();iter!=d_messages.end();++iter)
          out << *iter;
      }
private:

  static std::vector<std::string> d_messages;
};
} // End namespace SCIRun

#endif


