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
 *  ProgressiveWarning.h
 *
 *  Written by:
 *   Bryan Worthen
 *   SCI Institute
 *   University of Utah
 *   Aug. 2004
 *
 *  Copyright (C) 2000 SCI Group
 * 
 */

#ifndef SCI_Util_ProgressiveWarning_h
#define SCI_Util_ProgressiveWarning_h

#include <iostream>
#include <string>

namespace SCIRun {

/**************************************

CLASS 
   ProgressiveWarning
   
DESCRIPTION
   
   Holds info to display a warning message NOT every time it occurs.
 
  
****************************************/

  class ProgressiveWarning {
  public:
    //! Pass the message to output as a warning.  The multiplier is the amount to multiply the
    //! next occurence by when we output the warning.  -1 will mean to only output once.
    //! Output to stream.
    ProgressiveWarning(std::string message, int multiplier = -1, std::ostream& stream = std::cerr);

    //! Invoke the warning numTimes times.  If we've hit this enough times, output the warning message.
    void invoke(int numTimes = 1);
    void changeMessage(std::string message);
    void showWarning();
  private:
    void doMessage();
    std::string d_message;
    int d_numOccurences;
    int d_multiplier;
    int d_nextOccurence;

    bool d_warned;

    std::ostream* out;
  };

} // ends namespace SCIRun

#endif
