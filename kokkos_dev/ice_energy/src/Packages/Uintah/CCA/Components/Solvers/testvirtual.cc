#include <string>
#include <iostream>
#include <sstream>

// Here's a convenient & general utility function
//  You could also use boost::lexical_cast instead...
template<typename T>
std::string
to_string(const T& t)
{
  std::ostringstream strm;
  strm << t;
  return strm.str();
}

class Cillegal
{
public:
  virtual void do_it(const std::string& msg)=0;
 
  template<class T> void do_it(const T& t)
  {
    this->do_it( to_string(t) ); // pass-on to virtual fun
  }
};

class CillegalA : public Cillegal
{
public:
  virtual void do_it(const std::string& msg)
  { std::cout<<"I am class A "<<msg<<std::endl; }
};

class CillegalB : public Cillegal
{
public:
  virtual void do_it(const std::string& msg)
  { std::cout<<"I am class B "<<msg<<std::endl; }
};

template<class T>
void print_it(Cillegal& ill,const T& t)
{
  ill.do_it(t);
}

int main()
{
  int i = 1;
  double d = 3.0;
  to_string(i);
  to_string(d);
  CillegalA A;
  print_it(A,i);
  print_it(A,d);
  CillegalB B;
  print_it(B,i);
  return 0;
}
