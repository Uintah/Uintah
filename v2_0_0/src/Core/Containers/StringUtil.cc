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


/*
 *  StringUtil.c: Some useful string functions
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   April 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Util/Assert.h>
#include <Core/Containers/StringUtil.h>
#include <stdio.h>
#include <stdlib.h>

namespace SCIRun {

bool
string_to_int(const string &str, int &result)
{
  return sscanf(str.c_str(), "%d", &result) == 1;
}

bool
string_to_double(const string &str, double &result)
{
  return sscanf(str.c_str(), "%lf", &result) == 1;
}

string
to_string(int val)
{
  char s[50];
  sprintf(s, "%d", val);
  return string(s);
}

string
to_string(unsigned int val)
{
  char s[50];
  sprintf(s, "%u", val);
  return string(s);
}

string
to_string(double val)
{
  char s[50];
  sprintf(s, "%g", val);
  return string(s);
}

string
basename(const string &path)
{
  return path.substr(path.rfind('/')+1);
}

string
pathname(const string &path)
{
  return path.substr(0, path.rfind('/')+1);
}


char *
ccast_unsafe(const string &str)
{
  char *result = const_cast<char *>(str.c_str());
  ASSERT(result);
  return result;
}


vector<string> split_string(const std::string& str, char sep)
{
  vector<string> result;
  string s(str);
  while(s != ""){
    unsigned long first = s.find(sep);
    if(first < s.size()){
      result.push_back(s.substr(0, first));
      s = s.substr(first+1);
    } else {
      result.push_back(s);
      break;
    }
  }
  return result;
}

} // End namespace SCIRun

