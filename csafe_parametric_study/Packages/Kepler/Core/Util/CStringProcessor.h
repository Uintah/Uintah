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

//this file will include some helper functions that will enable us to
//process the strings that are sent to the SCIRun server by the
//SPA workflow.
//by oscar barney


#include <Core/Containers/StringUtil.h>

/**
 * Trim newlines from end of string.
 */
void trimCString(const char* input, std::string& output)
{
  // similar to string_Cify in Core/Containers/StringUtil.cc, but with dir. separator fixes...
  const std::string inputString(input);
  output.clear();
  for (string::size_type i = 0; i < inputString.size(); i++) {
    switch(inputString[i]) {
    case '\n':
      // skip this character
      continue;
      break;

    case '\r':
      // skip this character
      continue;
      break;

    case '\t':
      output.push_back('\\');
      output.push_back('t');
      break;

    case '\v':
      output.push_back('\\');
      output.push_back('v');
      break;

    case '\b':
      output.push_back('\\');
      output.push_back('b');
      break;

    case '\f':
      output.push_back('\\');
      output.push_back('f');
      break;

    case '\a':
      output.push_back('\\');
      output.push_back('a');
      break;

    case '"':
      output.push_back('\\');
      output.push_back('"');
      break;

    case '\\':
     // we're transmitting paths, so assume that any control characters
     // have been dealt with and any remaining '\\' are directory path
     // separators
      output.push_back('/');
      break;

    default:
      output.push_back(inputString[i]);
    }
  }
}

/**
 * Turns input characters into a vector of strings
 * and returns the vector. sets size to be the number
 * of things that are in the vector.
 *
 * Vector must be empty.
 */
bool processCString(const char* input, std::vector<std::string>& v, int& size)
{
  if (! v.empty()) {
    return false;
  }
  // empty input is not an error
  if (input == 0) {
    size = 0;
    return true;
  }

  // change the input into a string then vector
  std::string temp;
  trimCString(input, temp);
  // cut \n off the end of the last string
  v = SCIRun::split_string(temp,';');
  if (temp.empty()) {
    size = 0;  //case where we do not do anything
  } else {
    size = (int) v.size();
    //v[size - 1] = v[size - 1].substr(0, v[size - 1].size() - 1);
  }
  return true;
}
