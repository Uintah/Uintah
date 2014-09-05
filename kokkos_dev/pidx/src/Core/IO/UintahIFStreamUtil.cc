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

#include "UintahIFStreamUtil.h"
#include <iostream>
#include <limits>

namespace Uintah {
  
//------------------------------------------------------------------------------
  void
  skip_comments( std::ifstream &inputFStream )
  {
    char c;
    const int c_MAX_LINESIZE=std::numeric_limits<int>::max();
    
    if(!inputFStream) return;
    
    while (true) {
      
      // values are assumed to be separated by a space, comma, or semi-colon.
      while(isspace(c=inputFStream.get())||c==','||c==';');
      
      if(c=='#'||c=='%'||(c=='/' && inputFStream.peek()=='/'))
      {
        //skip the rest of the line
        inputFStream.ignore(c_MAX_LINESIZE, '\n');
      }
      
      else if (c=='/' && inputFStream.peek()=='*')
      {
        //skip everything in the comment block
        c=inputFStream.get();          //skip the first '*'
        char last='\0';
        while(!(inputFStream.eof()) && inputFStream.good())
        {
          c=inputFStream.get();
          if(c=='/' && last=='*')break;
          else last=c;
        }
      }
      
      else if(c!=EOF)
      {
        inputFStream.putback(c);
        break;
      }
    }
  }

//------------------------------------------------------------------------------
  static
  const std::string
  getToken( std::ifstream &inputFStream )
  {
    std::string token;
    skip_comments(inputFStream);
    inputFStream >> token;             
    return token;
  }

//------------------------------------------------------------------------------  
  const std::string
  getString( std::ifstream &inputFStream )
  {
    std::string token;
    skip_comments(inputFStream);
    inputFStream >> token;             
    return token;
  }

//------------------------------------------------------------------------------  
  double
  getDouble( std::ifstream &inputFStream )
  {
    double out;    
    getValue(inputFStream, out);
    return out;
  }

//------------------------------------------------------------------------------
  int
  getInt( std::ifstream &inputFStream )
  {
    int out;
    getValue(inputFStream, out);
    return out;
  }

//------------------------------------------------------------------------------  
  template <typename Type>
  void
  getValue( std::ifstream &inputFStream, Type& value )
  {
    skip_comments(inputFStream);
    inputFStream >> value;              
  }

//------------------------------------------------------------------------------
  const std::string
  getLine( std::ifstream &inputFStream, bool skipComments )
  {
    std::string line;
    if (skipComments) skip_comments(inputFStream);
    std::getline(inputFStream,line);
    return line;
  }

//------------------------------------------------------------------------------  
  #define INSTANTIATE_GET_VALUE( TYPE ) \
    template void getValue<TYPE>(std::ifstream&, TYPE&);
  
  INSTANTIATE_GET_VALUE(int);
  INSTANTIATE_GET_VALUE(float);
  INSTANTIATE_GET_VALUE(double);
  INSTANTIATE_GET_VALUE(std::string);
} // end namespace Uintah
