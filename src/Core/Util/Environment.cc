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

// Core SCIRun Includes
#include <sci_defs.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/RWS.h>

// STL Includes
#include <sgi_stl_warnings_off.h>
#include <Core/Util/Environment.h> // includes <string>
#include <iostream>
#include <stdio.h>
#include <set>
#include <sgi_stl_warnings_on.h>


namespace SCIRun {

using namespace std;

// MacroSubstitute takes var_val returns a string with the environment
// variables expanded out.  Performs one level of substitution
char*
MacroSubstitute( char * var_val )
{
  int    cur = 0;
  int    start = 0;
  int    end = start;
  string newstring("");
  char * macro = 0;
  
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
	  char *env = getenv(macro);
	  delete[] macro;
	  if (env) 
	    newstring += string(env);
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

// This set stores all of the environemnt keys that were set when scirun was
// started.  Its checked by sci_putenv to ensure we don't overwrite env variables
set<string> existing_env;

// get_existing_env() will fill up the SCIRun::existing_env string set
// with all the currently set environment variable keys, but not their values
void store_existing_environment(char **environ) 
{  
  char **environment = environ;
  existing_env.clear();
  while (*environment) {
    const string str(*environment);
    environment++;
    existing_env.insert(str.substr(0,str.find("=")));
  }
}

// sci_putenv will check the existing_env set string (filled in store_existing_env)
// If the environment key existed when the program was run, then  it will not
// be overwritten.  This follows the Unix convention of using environment 
// variables to supercede default program settings and .rc files.
void
sci_putenv( const string &key, const string &val,  bool force )
{
  // Check the environment as it existed when SCIRun was started,
  // return without owerwriting the varabile if the key is in our set
  if (!force && (existing_env.find(key) != existing_env.end())) return;

  // Use the old way of setting environment variables.
  // Because we need to remain compatible with old Unix's (namely SGI)
  const string envarstr = key+"="+val;
  char *envar = scinew char[envarstr.size()+1];
  memcpy(envar, envarstr.c_str(), envarstr.size());
  envar[envarstr.size()] = '\0';
  putenv(envar);
}  

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
bool
parse_scirunrc( const string rcfile )
{
  FILE* filein = fopen(rcfile.c_str(),"r");
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
	char* sub = MacroSubstitute(var_val);
	sci_putenv(var,sub);
	delete[] sub;
      }
    } else { // Couldn't find a string of the format var=var_val
      // Print out the offending line
      cerr << "Error parsing " << rcfile << " file on line: " 
	   << linenum << std::endl << "--> " << line << std::endl;
    }
  }
  fclose(filein);
  return true;
}

// find_and_parse_scirunrc will search for the users .scirunrc file in 
// default locations and read it into the environemnt if possible.
bool
find_and_parse_scirunrc()
{
  // Tell the user that we are searching for the .scirunrc file...
  std::cout << "Parsing .scirunrc... ";
  bool foundrc=false;

  // 1. check the local directory
  string filename(".scirunrc");
  foundrc = parse_scirunrc(filename);

  // 2. check the BUILD_DIR
  if (!foundrc) {
    filename = SCIRUN_OBJDIR+string("/.scirunrc");
    foundrc = parse_scirunrc(filename);
  }
  
  // 3. check the user's home directory
  char *HOME;
  if (!foundrc && (HOME = getenv("HOME"))) {
      filename = HOME+string("/.scirunrc");
      foundrc = parse_scirunrc(filename);
  }
  
  // 4. check the source code directory
  if (!foundrc) {
    filename = SCIRUN_SRCDIR+string("/.scirunrc");
    foundrc = parse_scirunrc(filename);
  }

  // The .scirunrc file wasn't found.
  if(!foundrc) filename = string("not found.");
  
  // print location of .scirunrc
  cout << filename << std::endl;

  // return if found
  return foundrc;
}


// sci_getenv_p will lookup the value of the environment variable 'key' and 
// returns false if the variable is equal to 'false', 'no', 'off', or '0'
// returns true otherwise.  Case insensitive.
bool
sci_getenv_p(const string &key) {
  char *value = getenv(key.c_str());
  if (!value || !(*value)) return false;
  string str;
  while (*value) {
    str += toupper(*value);
    value++;
  }
  // Only return false if value is zero (or equivalant)
  if (str == "FALSE" || str == "NO" || str == "OFF" || str == "0")
    return false;
  // Following C convention where any non-zero value is assumed true
  return true;
}


} // namespace SCIRun 
