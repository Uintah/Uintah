/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#include <StandAlone/tools/dumpfields/Args.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <cfloat>
#include <cstring>
#include <climits>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;
  
Args::Args(int argc, char ** argv)
{
  pname = argv[0];
  for(int iarg=1;iarg<argc;iarg++) {
    args.push_back(argv[iarg]);
    argused.push_back(false);
  }
}
  
string
Args::trailing()
{
  return this->trailing(1)[0];
}
  
vector<string>
Args::trailing(int num)
{
  vector<string> res;
  int nargs = args.size();
  if(nargs<num)
    throw ProblemSetupException("Not enough arguments",__FILE__,__LINE__);
  for(int iarg=nargs-num;iarg<nargs;iarg++) {
    if(argused[iarg])
      throw ProblemSetupException("Not enough arguments",__FILE__,__LINE__);
    res.push_back(args[iarg]);
    argused[iarg] = true;
  }
  return res;
}
  
vector<string> 
Args::allTrailing(unsigned int minnumber)
{
  vector<string> res;
  unsigned int nargs = args.size();
  if(nargs<minnumber)
    throw ProblemSetupException("Not enough arguments",__FILE__,__LINE__);
  for(int iarg=nargs-1;iarg>=0;iarg--) {
    if(!argused[iarg]) {
      res.push_back(args[iarg]);
      argused[iarg] = true;
    } else {
      break;
    }
  }
  if(res.size()<minnumber)
    throw ProblemSetupException("Not enough arguments",__FILE__,__LINE__);
  return res;
}
  
bool
Args::hasUnused() const
{
  for(vector<bool>::const_iterator ait(argused.begin());ait!=argused.end();ait++)
    if (!*ait) return true;
  return false;
}
  
vector<string> 
Args::unusedArgs() const
{
  vector<string> res;
  for(int iarg=0;iarg<(int)args.size();iarg++)
    if (!argused[iarg]) res.push_back( args[iarg] );
  return res;
}
  
bool
Args::getLogical(string name)
{
  int nargs = args.size();
  for(int iarg=0;iarg<nargs;iarg++) {
    if(args[iarg]==string("-")+name) {
      argused[iarg] = true;
      return true;
    }
  }
  return false;
}
  
string 
Args::getString(string name, string defval)
{
  int nargs = args.size();
  string res(defval);
  for(int iarg=0;iarg<nargs-1;iarg++) {
    if(args[iarg]=="-"+name) {
      if(argused[iarg] || argused[iarg+1])
        throw ProblemSetupException("Not enough arguments",__FILE__,__LINE__);
      res = args[iarg+1];
      argused[iarg  ] = true;
      argused[iarg+1] = true;
      break;
    }
  }
    
  return res;
}
  
string
Args::getString (string name) {
  string res = this->getString(name, "__nodef__");
  if(res=="__nodef__") {
    throw ProblemSetupException("Missing required argument '"+name+"'",__FILE__,__LINE__);
  }
  return res;
}
  
double
Args::getReal(string name) {
  string sval = this->getString(name);
  char * endptr;
  double val = strtod(sval.c_str(), &endptr);
  if(val==DBL_MAX || val==-DBL_MAX || (endptr && *endptr!='\0'))
    throw ProblemSetupException("Failed to convert value '"+sval+"' to double for argument '"+
                                name+"'",__FILE__,__LINE__);
  return val;
}
  
int 
Args::getInteger(string name) {
  string sv = this->getString(name);
  char * endptr;
  long tval = strtol(sv.c_str(), &endptr, 10);
  if(tval==LONG_MIN || (endptr && *endptr!='\0'))
    throw ProblemSetupException("Failed to convert value '"+sv+"' to integer for argument '"+
                                name+"'",__FILE__,__LINE__);
  return tval;
}
 
double
Args::getReal(std::string name, double defval)
{
  string sval = this->getString(name, "__nodefault__");
  if(sval=="__nodefault__") {
    return defval;
  } else {
    char * endptr;
    double val = strtod(sval.c_str(), &endptr);
    if(val==DBL_MAX || val==-DBL_MAX || (endptr && *endptr!='\0'))
      throw ProblemSetupException("Failed to convert value '"+sval+"' to double for argument '"+
                                  name+"'",__FILE__,__LINE__);
    return val;
  }
}
  
int 
Args::getInteger(std::string name, int defval) {
  string sv = this->getString(name, "__nodefault__");
  if(sv=="__nodefault__") {
    return defval;
  } else {
    char * endptr;
    long tval = strtol(sv.c_str(), &endptr, 10);
    if(tval==LONG_MIN || (endptr && *endptr!='\0'))
      throw ProblemSetupException("Failed to convert value '"+sv+"' to integer for argument '"+
                                  name+"'",__FILE__,__LINE__);
    return tval;
  }
}
  
