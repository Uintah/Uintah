/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Malloc/Allocator.h>
#include <Core/Util/RWS.h>
#include <Core/Util/Assert.h>
#include <Core/Util/FileUtils.h>
#include <Core/Util/Environment.h> // includes <string>

#include   <cstdlib>
#include   <cstring>
#include   <cstdio>
#include   <iostream>
#include   <map>

#include   <unistd.h>
#include   <sys/param.h>


#define SCI_OK_TO_INCLUDE_SCI_ENVIRONMENT_DEFS_H
#include <sci_defs/environment_defs.h>


static bool sci_environment_created = false;


// This set stores all of the environment keys that were set when scirun was
// started. Its checked by sci_putenv to ensure we don't overwrite variables
static std::map<std::string, std::string> scirun_env;


//______________________________________________________________________________
//
void
Uintah::show_env()
{
  printf("\n");
  printf("Environment:\n");

  std::map<std::string, std::string>::const_iterator iter = scirun_env.begin();
  while (iter != scirun_env.end()) {
    printf("  %s : %s\n", iter->first.c_str(), iter->second.c_str());
    ++iter;
  }
  printf("\n");
}


//______________________________________________________________________________
// WARNING: According to other software (specifically: tcl) you should
// lock before messing with the environment.

// Have to append 'Uintah::' to these function names so that the
// compiler believes that they are in the Uintah namespace (even
// though they are declared in the Uintah namespace in the .h file...)
const char *
Uintah::sci_getenv( const std::string & key )
{
  if (!sci_environment_created) {
    std::cout << "\n!!!WARNING!!! Core/Util/Environment.cc::sci_getenv() called before create_sci_environment()!\n";
    std::cout << "                Segfault probably coming soon...\n\n";
  }
  if (scirun_env.find(key) == scirun_env.end()) {
    return 0;
  }
  return scirun_env[key].c_str();
}


//______________________________________________________________________________
//
void
Uintah::sci_putenv( const std::string &key, const std::string &val )
{
  scirun_env[key] = val;
}  


//______________________________________________________________________________
// Used by sus.cc and restart_merger.cc
void
Uintah::create_sci_environment(char **env, char *execname, bool beSilent /* = false */ )
{
  if (sci_environment_created) {
    std::cout << "\n!!!WARNING!!! Core/Util/Environment.cc::create_sci_environment() called twice!  Skipping 2nd+ call.\n\n";
    return;
  }
  sci_environment_created = true;

  if (env) {
    char **environment = env;
    scirun_env.clear();
    while (*environment) {
      const std::string str(*environment);
      const size_t pos = str.find("=");
      scirun_env[str.substr(0, pos)] = str.substr(pos + 1, str.length());
      environment++;
    }
  }

  std::string objdir = SCIRUN_OBJDIR;
  std::string srcdir = SCIRUN_SRCDIR;

  if (!sci_getenv("SCIRUN_OBJDIR")) {
    if (!execname) {
      sci_putenv("SCIRUN_OBJDIR", objdir);
    }
    else {
      std::string objdir(execname);
      if (execname[0] != '/') {
        char cwd[MAXPATHLEN];
        getcwd(cwd, MAXPATHLEN);
        objdir = cwd + std::string("/") + objdir;
      }
      int pos = objdir.length() - 1;
      while (pos >= 0 && objdir[pos] != '/') {
        --pos;
      }
      ASSERT(pos >= 0);
      objdir.erase(objdir.begin() + pos + 1, objdir.end());
      sci_putenv("SCIRUN_OBJDIR", objdir);
    }
  }

  if (!sci_getenv("SCIRUN_SRCDIR")) {
    sci_putenv("SCIRUN_SRCDIR", srcdir);
  }
} // end create_sci_environment()


//______________________________________________________________________________
// sci_getenv_p will lookup the value of the environment variable 'key' and 
// returns false if the variable is equal to 'false', 'no', 'off', or '0'
// returns true otherwise.  Case insensitive.
bool
Uintah::sci_getenv_p(const std::string &key)
{
  const char *value = sci_getenv(key);

  // If the environment variable does NOT EXIST OR is EMPTY then return FASE
  if (!value || !(*value)) {
    return false;
  }
  std::string str;
  while (*value) {
    str += toupper(*value);
    value++;
  }

  // Only return false if value is zero (or equivalent)
  if (str == "FALSE" || str == "NO" || str == "OFF" || str == "0") {
    return false;
  }

  // Following C convention where any non-zero value is assumed true
  return true;
}

