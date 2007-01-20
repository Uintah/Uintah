//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : Color.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:06:40 2006

#include <math.h>
#include <stdlib.h>
#include <Core/Skinner/Color.h>

#ifdef _WIN32
#include <Core/OS/Rand.h>
#endif

#include <iostream>

namespace SCIRun {
  namespace Skinner {
    Color::Color() :
      r(1.0),
      g(0.0),
      b(0.0),
      a(1.0)
    {
    }

    Color::Color(double R, double G, double B, double A) :
      r(R),
      g(G),
      b(B),
      a(A)
    {
    }

    void
    Color::print(const string &prefix) {
      std::cerr << prefix << " color Red: " << r 
                << " Green: " << g 
                << " Blue: " << b 
                << " Alpha: " << a << std::endl; 
    }
    
    bool
    Color::operator==(const Color &rhs) {
      return this->r == rhs.r && this->g == rhs.g && this->b == rhs.b && this->a == rhs.a;
    }

    void 
    Color::random() {
      // Generates a bright random color
      while (fabs(r - g) + 
	     fabs(r - b) + 
	     fabs(g - b) < 1.0)
	{
	r = 1.0 - sqrt(1.0 - drand48());
	g = 1.0 - sqrt(1.0 - drand48());
	b = 1.0 - sqrt(1.0 - drand48());
      }
      a = 1.0;
    }

  }
}
