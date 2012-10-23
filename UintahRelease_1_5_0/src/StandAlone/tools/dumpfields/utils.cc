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

#include "utils.h"

using namespace std;

vector<string> 
split(const string & s, char sep, bool skipmult)
{
  vector<string> res;
  
  string::size_type ls(s.size());
  string::size_type p0(0), p1;
  
  while( (p1=s.find(sep,p0))!=string::npos )
    {
      // Assert(p1>=0 && p1<ls, "string separator pointer in range");
      // Assert(p1>=p0,         "non-negative sub-string length");
      
      res.push_back(s.substr(p0, p1-p0));
      p0 = p1+1;
      if(skipmult)
	{
	  while(p0<ls && s[p0]==sep)
	    p0++;
	}
    }
  
  if(p0<s.size() || (p0>0&&p0==s.size()) ) // end case of splitting '1,2,' into three
    {
      res.push_back(s.substr(p0));
    }
  
  return res;
}
