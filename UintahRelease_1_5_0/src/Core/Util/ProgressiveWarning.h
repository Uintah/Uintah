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

/*
 *  ProgressiveWarning.h
 *
 *    To the best of my understanding, this class will print out a given
 *    error message based on the number of times the warning is invoked
 *    (occurs).  This depends on the settings given to the  warning object.
 *    For example:
 *
 *       ProgressiveWarning warn( msg, 10 );
 *       warn.invoke();
 *    
 *    Would print out the 1st, 10th, 100th, 1000th... etc time the warning
 *    is invoked.  A 'warn( msg, 2)' would display the 1st, 2nd, 4th, 8th,
 *    16th, etc.
 *
 *  Written by:
 *   Bryan Worthen
 *   SCI Institute
 *   University of Utah
 *   Aug. 2004
 *
 * 
 */

#ifndef SCI_Util_ProgressiveWarning_h
#define SCI_Util_ProgressiveWarning_h

#include <iostream>
#include <string>

#include <Core/Util/share.h>

namespace SCIRun {

/**************************************

CLASS 
   ProgressiveWarning
   
DESCRIPTION
   
   Holds info to display a warning message NOT every time it occurs.
 
  
****************************************/

  class SCISHARE ProgressiveWarning {
  public:
    //! Pass the message to output as a warning.  The multiplier is the amount to multiply the
    //! next occurence by when we output the warning.  -1 will mean to only output once.
    //! Output to stream.  'Multiplier' should not be set to '1'... and will be updated to '2' if
    //! '1' is specified.
    ProgressiveWarning(std::string message, int multiplier = -1, std::ostream& stream = std::cerr);

    //! Invoke the warning numTimes times.  If we've hit this enough times, output the warning message.
    //! Return true if we output the message.
    bool invoke(int numTimes = 1);
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
