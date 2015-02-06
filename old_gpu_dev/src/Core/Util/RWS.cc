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


#include <Core/Util/RWS.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <Core/Util/notset.h>

namespace SCIRun {
using std::string;

const char _NOTSET_[] = "(null string)";

bool remove_lt_white_space(string &str)
{
  string::iterator iter = str.begin();

  int idx1 = 0;
  while ((iter < str.end())) 
  {
    if (*iter == ' '  || *iter == '\t' || 
	*iter == '\n' || *iter == '\r') 
    {
      ++iter; ++idx1;
    } 
    else break;
  }
  str.erase(0, idx1);

  string::reverse_iterator riter = str.rbegin();
  int idx2 = str.size() - 1;
  while ((riter < str.rend())) 
  {
    if (*riter == ' '  || *riter == '\t' || 
	*riter == '\n' || *riter == '\r') 
    {
      ++riter; --idx2;
    } 
    else break;
  }

  str.erase(idx2 + 1, str.size());
  return true;
}

char* removeLTWhiteSpace(char* str)
{
  char* newstr = 0;
  int index1 = 0, index2 = 0;

  index2 = strlen(str)-1;

  newstr = new char[index2+2];
  newstr[0] = '\0';

  while ((index1<=index2) &&
         (str[index1]==' '||
          str[index1]=='\t' ||
          str[index1]=='\n' ||
          str[index1]=='\r')) index1++;

  while ((index2>=index1) &&
         (str[index2]==' '||
          str[index2]=='\t' ||
          str[index2]=='\n' ||
          str[index2]=='\r')) index2--;

  if (index1>index2) 
    return NOT_SET;

  str[index2+1]='\0';
  sprintf(newstr,"%s",&str[index1]);

  sprintf(str,"%s",newstr);

  delete[] newstr;

  return str;
}

}
