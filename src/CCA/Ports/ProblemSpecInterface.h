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


#ifndef UINTAH_HOMEBREW_ProblemSpecINTERFACE_H
#define UINTAH_HOMEBREW_ProblemSpecINTERFACE_H

#include <Core/Parallel/UintahParallelPort.h>
#include <Core/ProblemSpec/ProblemSpecP.h>


#include <string>

namespace Uintah {

/**************************************

CLASS
   ProblemSpecInterface
   
   Short description...

GENERAL INFORMATION

   ProblemSpecInterface.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   ProblemSpec_Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class ProblemSpecInterface : public UintahParallelPort {
   public:
      ProblemSpecInterface();
      virtual ~ProblemSpecInterface();

      virtual ProblemSpecP readInputFile( const std::string & filename, bool validate = false ) = 0;
      virtual std::string getInputFile() = 0;
      
   private:
      ProblemSpecInterface( const ProblemSpecInterface & );
      ProblemSpecInterface & operator=( const ProblemSpecInterface & );
   };
} // End namespace Uintah

#endif

