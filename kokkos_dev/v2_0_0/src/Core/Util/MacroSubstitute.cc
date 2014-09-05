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

#include <Core/Util/MacroSubstitute.h>
#include <stdio.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;
using std::cerr;
using std::endl;

char* MacroSubstitute(char* var_val, env_map& env)
{
  int cur = 0;
  int start = 0;
  int end = start;
  string newstring("");
  char* macro = 0;
  
  if (var_val==0)
    return 0;

  int length = (int)strlen(var_val);

  while (cur < length-1) {
    if (var_val[cur] == '$' && var_val[cur+1]=='(') {
      cur+=2;
      start = cur;
      while (cur < length) {
	if (var_val[cur]==')') {
	  end = cur-1;
	  var_val[cur]='\0';
	  macro = new char[end-start+1];
	  sprintf(macro,"%s",&var_val[start]);
	  env_iter i = env.find(string(macro));
	  delete[] macro;
	  if (i!=env.end()) 
	    newstring += (*i).second;
	  var_val[cur]=')';
	  cur++;
	  break;
	} else
	  cur++;
      }
    } else {
      newstring += var_val[cur];
      cur++;
    }
  }

  newstring += var_val[cur]; // don't forget the last character!

  unsigned long newlength = strlen(newstring.c_str());
  char* retval = new char[newlength+1];
  sprintf(retval,"%s",newstring.c_str());
  
  return retval;
}

}
