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
 *  StringUtil.c: Some useful string functions
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   April 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Util/Assert.h>
#include <Core/Containers/StringUtil.h>
#include <stdio.h>
#include <stdlib.h>

namespace SCIRun {

bool
string_to_int(const string &str, int &result)
{
  return sscanf(str.c_str(), "%d", &result) == 1;
}

bool
string_to_double(const string &str, double &result)
{
  return sscanf(str.c_str(), "%lf", &result) == 1;
}

string
to_string(int val)
{
  char s[50];
  sprintf(s, "%d", val);
  return string(s);
}

string
to_string(unsigned int val)
{
  char s[50];
  sprintf(s, "%u", val);
  return string(s);
}

string
to_string(unsigned long val)
{
  char s[50];
  sprintf(s, "%lu", val);
  return string(s);
}

string
to_string(double val)
{
  char s[50];
  sprintf(s, "%g", val);
  return string(s);
}

string
basename(const string &path)
{
  return path.substr(path.rfind('/')+1);
}

string
pathname(const string &path)
{
  return path.substr(0, path.rfind('/')+1);
}


char *
ccast_unsafe(const string &str)
{
  char *result = const_cast<char *>(str.c_str());
  ASSERT(result);
  return result;
}


vector<string>
split_string(const std::string& str, char sep)
{
  vector<string> result;
  string s(str);
  while(s != ""){
    unsigned long first = s.find(sep);
    if(first < s.size()){
      result.push_back(s.substr(0, first));
      s = s.substr(first+1);
    } else {
      result.push_back(s);
      break;
    }
  }
  return result;
}


/////////
// C++ify a string, turn newlines into \n, use \t, \r, \\ \", etc.
string
string_Cify(const string &str)
{
  string result("");
  for (string::size_type i = 0; i < str.size(); i++)
  {
    switch(str[i])
    {
    case '\n':
      result.push_back('\\');
      result.push_back('n');
      break;

    case '\r':
      result.push_back('\\');
      result.push_back('r');
      break;

    case '\t':
      result.push_back('\\');
      result.push_back('t');
      break;

    case '"':
      result.push_back('\\');
      result.push_back('"');
      break;

    case '\\':
      result.push_back('\\');
      result.push_back('\\');
      break;

    default:
      result.push_back(str[i]);
    }
  }
  return result;
}


// replaces all occurances of 'substr' in 'str' with 'replacement'
string 
replace_substring(string str, 
		  string substr, 
		  string replacement) 
{
  string::size_type pos;
  do {
    pos = str.find(substr);
    if (pos != string::npos)
      str = str.replace(str.begin()+pos, str.begin()+pos+substr.length(), replacement);
  } while (pos != string::npos);
  return str;
}


bool
ends_with(string str, string substr)
{
  return str.rfind(substr) == str.size()-substr.size();
}  


} // End namespace SCIRun

