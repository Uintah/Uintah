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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Core/Util/notset.h>

namespace SCIRun {

const char _NOTSET_[] = "(null string)";

char* removeLTWhiteSpace(char* string)
{
  char* newstring = 0;
  int index1 = 0, index2 = 0;

  index2 = strlen(string)-1;

  newstring = new char[index2+2];
  newstring[0] = '\0';

  while ((index1<=index2) &&
         (string[index1]==' '||
          string[index1]=='\t' ||
          string[index1]=='\n' ||
          string[index1]=='\r')) index1++;

  while ((index2>=index1) &&
         (string[index2]==' '||
          string[index2]=='\t' ||
          string[index2]=='\n' ||
          string[index2]=='\r')) index2--;

  if (index1>index2) 
    return NOT_SET;

  string[index2+1]='\0';
  sprintf(newstring,"%s",&string[index1]);

  sprintf(string,"%s",newstring);

  delete[] newstring;

  return string;
}

}
