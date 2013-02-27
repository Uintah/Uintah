/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


/*
 *  StringUtil.c: Some useful string functions
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   April 2001
 *
 */

#include <Core/Util/Assert.h>
#include <Core/Containers/StringUtil.h>
#include <cstdio>
#include <cstdlib>
#include <ctype.h> // for toupper() (at least for linux RH8)

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

bool
string_to_unsigned_long(const string &str, unsigned long &result)
{
  return sscanf(str.c_str(), "%lu", &result) == 1;
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
to_string(unsigned long val)
{
  char s[50];
  sprintf(s, "%lu", val);
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

static
bool
is_separator( char ch, vector<char> separators )
{
  for( unsigned int pos = 0; pos < separators.size(); pos++ ) {
    if( ch == separators[ pos ] ) {
      return true;
    }
  }
  return false;
}

string
concatStrings( const vector<string> strings )
{
  string result;
  for( unsigned int pos = 0; pos < strings.size(); pos++ ) {
    result += strings[pos];
    if( pos != (strings.size()-1) ) {
      result += ", ";
    }
  } 
  return result;
}

vector<string>
split_string(const std::string& str, const vector<char> & separators)
{
  vector<string> result;
  unsigned int begin = 0;

  bool validDataFound = false;

  for( unsigned int pos = 0; pos < str.length(); pos++ ) {
    if( is_separator( str[pos], separators ) ) {
      validDataFound = false;
      if( pos > begin ) {
        result.push_back( str.substr( begin, (pos-begin) ) );
        begin = pos + 1;
      }
      else if( !validDataFound ) {
        begin++;
      }
    } 
    else {
      validDataFound = true;
    }
  }
  if( begin != str.length() ) {
    int size = str.length() - begin + 1;
    result.push_back( str.substr( begin, size ) );
  }
  return result;
}


/////////
// C++ify a string, turn newlines into \n, use \t, \r, \\ \", etc.
string
string_Cify(const string &str)
{
  string result("");
  for (string::size_type i = 0; i < str.size(); i++)
  {
    switch(str[i])
    {
    case '\n':
      result.push_back('\\');
      result.push_back('n');
      break;

    case '\r':
      result.push_back('\\');
      result.push_back('r');
      break;

    case '\t':
      result.push_back('\\');
      result.push_back('t');
      break;

    case '"':
      result.push_back('\\');
      result.push_back('"');
      break;

    case '\\':
      result.push_back('\\');
      result.push_back('\\');
      break;

    default:
      result.push_back(str[i]);
    }
  }
  return result;
}

// Remove leading and trailing white space (blanks, tabs, \n, \r) from string.
void
collapse( string & str )
{
  string orig = str;

  str = "";
  
  unsigned int start = 0;
  for( ; start < orig.length(); start++ ) {
    char ch = orig[ start ];

    if( ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r' ) {
      break;
    }
  }

  unsigned int end = orig.length();
  for( ; end > start; end-- ) {
    char ch = orig[ end-1 ];

    if( ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r' ) {
      break;
    }
  }
  
  if( start != (orig.length() ) ) {
    str = orig.substr( start, end-start );
  }
}


// replaces all occurances of 'substr' in 'str' with 'replacement'
void
replace_substring( string & str,
                   const string & substr, 
                   const string & replacement )
{
  string::size_type pos;
  do {
    pos = str.find(substr);
    if (pos != string::npos)
      str = str.replace(str.begin()+pos, 
                        str.begin()+pos+substr.length(), 
                        replacement);
  } while (pos != string::npos);
}


bool
ends_with(const string &str, const string &substr)
{
  return str.rfind(substr) == str.size()-substr.size();
}  


string
string_toupper( const string & str ) 
{
  string temp = str;
  for (unsigned int i = 0; i < temp.length(); ++i) {
    temp[i] = toupper(str[i]);
  }
  return temp;
}

string
string_tolower( const string & str ) 
{
  string temp = str;
  for (unsigned int i = 0; i < temp.length(); ++i) {
    temp[i] = tolower(str[i]);
  }
  return temp;
}

unsigned int
count_substrs( const string & str, const string & substr )
{
  unsigned int num = 0;
  size_t       loc = 0;

  while( (loc = str.find(substr, loc)) != string::npos ) {
    loc++;
    num++;
  }
  return num;
}

} // End namespace SCIRun

