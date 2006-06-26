#include "utils.h"

using namespace std;

vector<string> 
split(const string & s, char sep, bool skipmult)
{
  vector<string> res;
  
  string::size_type ls(s.size());
  string::size_type p0(0), p1;
  
  while( (p1=s.find(sep,p0))!=string::npos )
    {
      // Assert(p1>=0 && p1<ls, "string separator pointer in range");
      // Assert(p1>=p0,         "non-negative sub-string length");
      
      res.push_back(s.substr(p0, p1-p0));
      p0 = p1+1;
      if(skipmult)
	{
	  while(p0<ls && s[p0]==sep)
	    p0++;
	}
    }
  
  if(p0<s.size() || (p0>0&&p0==s.size()) ) // end case of splitting '1,2,' into three
    {
      res.push_back(s.substr(p0));
    }
  
  return res;
}
