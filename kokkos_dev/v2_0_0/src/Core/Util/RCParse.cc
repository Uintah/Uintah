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

// emptryOrComment returns true if the 'line' passed in is a comment
// ie: the first non-whitespace character is a '#'
// or if the entire line is empty or white space.
bool
emptyOrComment( const char * line )
{
  const char A_TAB = '	';
  int   length = strlen( line );

  for( int pos = 0; pos < length; pos++ ) {
    if( line[pos] == '#' ) {
      return true;
    } else if( ( line[pos] != ' ' ) && ( line[pos] != A_TAB ) ) {
      return false;
    }
  }
  return true;
}

// parse_scirunrc reads the .scirunrc file 'rcfile' into the program's enviroment
// It uses sci_putenv to set variables in the environment. 
// Returns true if the file was opened and parsed.  False otherwise.

bool RCParse(const char* rcfile, env_map& env)
{
  FILE* filein = fopen(rcfile,"r");
  if (!filein) return false;

  char var[0xff];
  char var_val[0xffff];
  bool done = false;
  int linenum = 0;

  while( !done ) {
    linenum++;
    var[0]='\0';
    var_val[0]='\0';

    char line[1024];
    // If we get to the EOF:
    if( !fgets( line, 1024, filein ) ) break;

    int length = strlen(line);
    if( length > 0 ) {
      // Replace CR with EOL.
      line[length-1] = 0;
    }
      
    // Skip commented out lines or blank lines
    if( emptyOrComment( line ) ) continue;

    // Get the environment variable and its value
    if( sscanf( line, "%[^=]=%s", var, var_val ) == 2 ){
      if (var[0]!='\0' && var_val[0]!='\0') {
	removeLTWhiteSpace(var);
	removeLTWhiteSpace(var_val);
	char* sub = MacroSubstitute(var_val,env);
	env.insert(env_entry(string(var),string(sub)));
	delete[] sub;
      }
    } else { // Couldn't find a string of the format var=var_val
      // Print out the offending line
      std::cerr << "Error parsing " << rcfile << " file on line: " 
		<< linenum << std::endl << "--> " << line << std::endl;
    }
  }
  fclose(filein);
  return true;
}

} // namespace SCIRun 
