//#include <string>
#include <iostream>

using namespace std;

void
outputs(const string &s,
	const string &s2,
	const string &s3,
	const string &time){
  if ( time == "8")
    
  cout << s2 << endl;
  else
    cout << s3 << endl;
}


int main()
{
  string s, s2, s3, time;
  s = "hello paula";
  s2 = "morning";
  s3 = "have a nice day";
  time = "hohoho";
  
  outputs(s, s2, s3, time);  
  
  return 0;
}
