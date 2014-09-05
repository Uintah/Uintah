/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
