/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
// Core SCIRun Includes
#include <Core/Malloc/Allocator.h>
#include <Core/Util/RWS.h>
#include <Core/Util/Assert.h>
#include <Core/Util/FileUtils.h>
#include <Core/Util/sci_system.h>

// STL Includes
#include   <Core/Util/Environment.h> // includes <string>
#include   <iostream>
#include   <map>
#include   <cstring>
#include   <cstdlib>
#include   <cstdio>

#define SCI_OK_TO_INCLUDE_SCI_ENVIRONMENT_DEFS_H
#include <sci_defs/environment_defs.h>

#ifndef _WIN32
#  include <unistd.h>
#  include <sys/param.h>
#else
#  define MAXPATHLEN 256
#  include <direct.h>
#  include <windows.h>
#endif

using namespace SCIRun;
using namespace std;

static bool sci_environment_created = false;

namespace SCIRun {
  void find_and_parse_scirunrc( bool beSilent = false );
  bool parse_scirunrc(const std::string &);
}

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

  char* var_val = strdup(var_value);
  int length = strlen(var_val);
  while (cur < length-1) {
    if (var_val[cur] == '$' && var_val[cur+1]=='(') {
      cur+=2;
      start = cur;
      while (cur < length) {
	if (var_val[cur]==')') {
	  end = cur-1;
	  var_val[cur]='\0';
	  macro = new char[end-start+2];
	  sprintf(macro,"%s",&var_val[start]);
	  const char *env = sci_getenv(macro);
	  delete [] macro;
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
  free(var_val);

  unsigned long newlength = strlen(newstring.c_str());
  char* retval = new char[newlength+1];
  sprintf(retval,"%s",newstring.c_str());
  
  return retval;
}

void
SCIRun::show_env()
{
  printf( "\n" );
  printf("Environment:\n" );

  map<string,string>::const_iterator iter = scirun_env.begin();
  while( iter != scirun_env.end() ) {
    printf( "  %s : %s\n", iter->first.c_str(), iter->second.c_str() );
    iter++;
  }
}

// WARNING: According to other software (specifically: tcl) you should
// lock before messing with the environment.

// Have to append 'SCIRun::' to these function names so that the
// compiler believes that they are in the SCIRun namespace (even
// though they are declared in the SCIRun namespace in the .h file...)
const char *
SCIRun::sci_getenv( const string & key )
{
  if( !sci_environment_created ) {
    cout << "\n!!!WARNING!!! Core/Util/Environment.cc::sci_getenv() called before create_sci_environment()!\n";
    cout << "                Segfault probably coming soon...\n\n";
  }
  if (scirun_env.find(key) == scirun_env.end()) return 0;
  return scirun_env[key].c_str();
}

void
SCIRun::sci_putenv( const string &key, const string &val )
{
  scirun_env[key] = val;
}  


void
SCIRun::create_sci_environment(char **env, char *execname, bool beSilent /* = false */ )
{
  if( sci_environment_created ) {
    cout << "\n!!!WARNING!!! Core/Util/Environment.cc::create_sci_environment() called twice!  Skipping 2nd+ call.\n\n";
    return;
  }
  sci_environment_created = true;

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

  string objdir = SCIRUN_OBJDIR;
  string srcdir = SCIRUN_SRCDIR;

  if (!sci_getenv("SCIRUN_OBJDIR")) 
  {
    if (!execname)
      sci_putenv("SCIRUN_OBJDIR", objdir);
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
    sci_putenv("SCIRUN_SRCDIR", srcdir);

  find_and_parse_scirunrc( beSilent );

} // end create_sci_environment()

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
	   << linenum << "\n" << "--> " << line << "\n";
    }
  }
  fclose(filein);
  sci_putenv("SCIRUN_RC_PARSED","1");
  return true;
}

// find_and_parse_scirunrc will search for the users .scirunrc file in 
// default locations and read it into the environemnt if possible.
void
SCIRun::find_and_parse_scirunrc( bool beSilent /* = false */ )
{
  // Tell the user that we are searching for the .scirunrc file...
  if( !beSilent ) {
    cout << "Parsing .scirunrc... ";
  }

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
  const char *HOME;
  if (!foundrc && (HOME = sci_getenv("HOME"))) {
      filename = HOME+string("/.scirunrc");
      foundrc = parse_scirunrc(filename);
  }
  
  // 4. check the source code directory
  if (!foundrc) {
    filename = SCIRUN_SRCDIR+string("/.scirunrc");
    foundrc = parse_scirunrc(filename);
  }

  if( !beSilent ) {
    // The .scirunrc file wasn't found.
    if(!foundrc) filename = string("not found.");
    
    // print location of .scirunrc
    cout << filename << "\n";
  }
}


void
SCIRun::copy_and_parse_scirunrc()
{
#ifdef _MSC_VER
  // native windows doesn't have "HOME"
  // point to OBJTOP instead
  sci_putenv("HOME", sci_getenv("SCIRUN_OBJDIR"));
#endif
  const char* home = sci_getenv("HOME");
  const char* srcdir = sci_getenv("SCIRUN_SRCDIR");
  ASSERT(home && srcdir);  
  if (!home || !srcdir) return;

  string cmd;
  string homerc = string(home)+"/.scirunrc";
  if (validFile(homerc))
  {
    const char* env_rcfile_version = sci_getenv("SCIRUN_RCFILE_VERSION");
    string backup_extension =(env_rcfile_version ? env_rcfile_version:"bak");
    string backuprc = homerc + "." + backup_extension;
    cmd = "cp -f "+homerc+" "+backuprc;
    cout << "Backing up " << homerc << " to " << backuprc << "\n";
    if (sci_system(cmd.c_str()))
    {
      cerr << "Error executing: " << cmd << "\n";
    }
  }
  
  cmd = string("cp -f ")+srcdir+"/scirunrc "+homerc;
  cout << "Copying " << srcdir << "/scirunrc to " << homerc << "...\n";
  if (sci_system(cmd.c_str())) {
    cerr << "Error executing: " << cmd << "\n";
  }
  else { 
    // If the scirunrc file was copied, then parse it.
    parse_scirunrc(homerc);
  }
}




// sci_getenv_p will lookup the value of the environment variable 'key' and 
// returns false if the variable is equal to 'false', 'no', 'off', or '0'
// returns true otherwise.  Case insensitive.
bool
SCIRun::sci_getenv_p(const string &key) 
{
  const char *value = sci_getenv(key);

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

