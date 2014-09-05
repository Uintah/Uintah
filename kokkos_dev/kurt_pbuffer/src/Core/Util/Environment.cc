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

// Core SCIRun Includes
#include <Core/Malloc/Allocator.h>
#include <Core/Util/RWS.h>
#include <Core/Util/Assert.h>

// STL Includes
#include <sgi_stl_warnings_off.h>
#include <Core/Util/Environment.h> // includes <string>
#include <iostream>
#include <map>
#include <sgi_stl_warnings_on.h>

#define SCI_OK_TO_INCLUDE_SCI_ENVIRONMENT_DEFS_H
#include <sci_defs/environment_defs.h>

#include <stdio.h>
#include <unistd.h>
#include <sys/param.h>


#ifndef LOAD_PACKAGE
#  error You must set a LOAD_PACKAGE or life is pretty dull
#endif

#ifndef ITCL_WIDGETS
#  error You must set ITCL_WIDGETS to the iwidgets/scripts path
#endif

using namespace SCIRun;
using namespace std;

// This set stores all of the environemnt keys that were set when scirun was
// started. Its checked by sci_putenv to ensure we don't overwrite variables
static map<string,string> scirun_env;

// MacroSubstitute takes var_value returns a string with the environment
// variables expanded out.  Performs one level of substitution
//   Note: Must delete the returned string when you are done with it.
char*
MacroSubstitute( const char * var_value )
{
  int    cur = 0;
  int    start = 0;
  int    end = start;
  string newstring("");
  char * macro = 0;
  
  if (var_value==0)
    return 0;

  char * var_val = strdup(var_value);

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
	  const char *env = sci_getenv(macro);
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
  
  free(var_val);
  return retval;
}

// WARNING: According to other software (specifically: tcl) you should
// lock before messing with the environment.

// Have to append 'SCIRun::' to these function names so that the
// compiler believes that they are in the SCIRun namespace (even
// though they are declared in the SCIRun namespace in the .h file...)
const char *
SCIRun::sci_getenv( const string & key )
{
  if (scirun_env.find(key) == scirun_env.end()) return 0;
  return scirun_env[key].c_str();
}

void
SCIRun::sci_putenv( const string &key, const string &val )
{
  scirun_env[key] = val;
}  

// get_existing_env() will fill up the SCIRun::existing_env string set
// with all the currently set environment variable keys, but not their values
void
SCIRun::create_sci_environment(char **env, char *execname)
{
  if (env) {
    char **environment = env;
    scirun_env.clear();
    while (*environment) {
      const string str(*environment);
      const size_t pos = str.find("=");
      scirun_env[str.substr(0,pos)] = str.substr(pos+1, str.length());
      environment++;
    }
  }

  if (!sci_getenv("SCIRUN_OBJDIR")) 
  {
    if (!execname)
      sci_putenv("SCIRUN_OBJDIR", SCIRUN_OBJDIR);
    else {
      string objdir(execname);
      if (execname[0] != '/') {
	char cwd[MAXPATHLEN];
	getcwd(cwd,MAXPATHLEN);
	objdir = cwd+string("/")+objdir;
      }
      int pos = objdir.length()-1;
      while (pos >= 0 && objdir[pos] != '/') --pos;
      ASSERT(pos >= 0);
      objdir.erase(objdir.begin()+pos+1, objdir.end());
      sci_putenv("SCIRUN_OBJDIR", objdir);
    }
  }

  if (!sci_getenv("SCIRUN_SRCDIR"))
      sci_putenv("SCIRUN_SRCDIR", SCIRUN_SRCDIR);
  if (!sci_getenv("SCIRUN_THIRDPARTY_DIR"))
      sci_putenv("SCIRUN_THIRDPARTY_DIR", SCIRUN_THIRDPARTY_DIR);
  if (!sci_getenv("SCIRUN_LOAD_PACKAGE"))
    sci_putenv("SCIRUN_LOAD_PACKAGE", LOAD_PACKAGE);
  if (!sci_getenv("SCIRUN_ITCL_WIDGETS"))
    sci_putenv("SCIRUN_ITCL_WIDGETS", ITCL_WIDGETS);
  sci_putenv("SCIRUN_ITCL_WIDGETS", 
	     MacroSubstitute(sci_getenv("SCIRUN_ITCL_WIDGETS")));

}

// emptryOrComment returns true if the 'line' passed in is a comment
// ie: the first non-whitespace character is a '#'
// or if the entire line is empty or white space.
bool
emptyOrComment( const char * line )
{
  const char A_TAB = '	';
  int   length = (int)strlen( line );

  for( int pos = 0; pos < length; pos++ ) {
    if( line[pos] == '#' ) {
      return true;
    } else if( ( line[pos] != ' ' ) && ( line[pos] != A_TAB ) ) {
      return false;
    }
  }
  return true;
}

// parse_scirunrc reads the .scirunrc file 'rcfile' into the SCIRuns enviroment
// It uses sci_putenv to set variables in the environment. 
// Returns true if the file was opened and parsed.  False otherwise.
bool
SCIRun::parse_scirunrc( const string &rcfile )
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

    int length = (int)strlen(line);
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



	// Only put the var into the environment if it is not already there.
	if(!SCIRun::sci_getenv( var ) || 
	   // Except the .scirunrc version, it should always come from the file
	   string(var) == string("SCIRUN_RCFILE_VERSION")) {
	  sci_putenv(var,sub);
	} 

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
SCIRun::find_and_parse_scirunrc()
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
SCIRun::sci_getenv_p(const string &key) 
{
  const char *value = sci_getenv( key );

  // If the environment variable does NOT EXIST OR is EMPTY then return FASE
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

