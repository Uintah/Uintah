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

#include <stdio.h>
#include <Core/Util/RCParse.h>
#include <Core/Util/RWS.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <Core/Util/scirun_env.h>
#include <Core/Util/MacroSubstitute.h>

namespace SCIRun {

bool RCParse(const char* rcfile, env_map& env)
{
  FILE* filein = fopen(rcfile,"r");
  char var[0xff];
  char var_val[0xffff];

  if (!filein)
    return false;

  int i=1; // prevent warning

  while (i) {
    var[0]='\0';
    var_val[0]='\0';
    if (fscanf(filein,"%[^=]=%s",var,var_val)!=EOF){
      if (var[0]!='\0' && var_val[0]!='\0') {
	removeLTWhiteSpace(var);
	removeLTWhiteSpace(var_val);
	char* sub = MacroSubstitute(var_val,env);
	env.insert(env_entry(string(var),string(sub)));
	//std::cerr << "inserted : ]" << var << "=" << sub
	//  << "[" << std::endl;
	delete[] sub;
      }
    } else
      break;
  }

  fclose(filein);

  return true;
}

} // namespace SCIRun 
