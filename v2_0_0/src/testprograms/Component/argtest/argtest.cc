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
 *  argtest.cc
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 U of U
 */

#include <Core/CCA/PIDL/PIDL.h>

#include <testprograms/Component/argtest/argtest_sidl.h>

#include <Core/Thread/Time.h>
#include <Core/Thread/Thread.h>

#include <iostream>
#include <vector>
#include <sstream>

#include <unistd.h>

using namespace std;

using argtest::ref;
using argtest::Server;
using argtest::TestEnum;
using argtest::Foo;
using argtest::Bar;

using SSIDL::array1;
using std::istringstream;
using std::ostringstream;

using namespace SCIRun;

static void init(array1<int>& a, int s)
{
  a.resize(s);
  for(int i=0;i<s;i++)
    a[i]=s+i;
}

static bool test(const array1<int>& a, int s)
{
  if((int)a.size() != s)
    return false;
  for(int i=0;i<s;i++){
    if(a[i] != s+i)
      return false;
  }
  return true;
}

static void init(array1<array1<int> >& a, int s)
{
  a.resize(s);
  for(int i=0;i<s;i++)
    init(a[i], i+s);
}

static bool test(const array1<array1<int> >& a, int s)
{
  if(int(a.size()) != s)
    return false;
  for(int i=0;i<s;i++){
    if(!test(a[i], i+s))
      return false;
  }
  return true;
}

static void init(array1<string>& a, int s)
{
  a.resize(s);
  for(int i=0;i<s;i++){
    ostringstream o;
    o << s+i;
    a[i]=o.str();
  }
}

static bool test(const array1<string>& a, int s)
{
  if(int(a.size()) != s)
    return false;
  for(int i=0;i<s;i++){
    string tmp = a[i];
    istringstream in(tmp);
    int t;
    in >> t;
    if(t != s+i)
      return false;
  }
  return true;
}

static void init(array1<bool>& a, int s)
{
  a.resize(s);
  for(int i=0;i<s;i++)
    a[i]=((s+i)&1)?true:false;
}

static bool test(const array1<bool>& a, int s)
{
  if(int(a.size()) != s)
    return false;
  for(int i=0;i<s;i++){
    if(a[i] != (((s+i)&1)?true:false))
      return false;
  }
  return true;
}

class ref_impl : public argtest::ref {
  int myvalue;
public:
  ref_impl(int myvalue);
  virtual ~ref_impl();
  virtual int test();
};

ref_impl::ref_impl(int myvalue)
  : myvalue(myvalue)
{
}

ref_impl::~ref_impl()
{
}

int ref_impl::test()
{
  return myvalue;
}

class Server_impl : public argtest::Server {
  bool success;
  ref::pointer return_reference;
  ref::pointer out_reference;
  ref::pointer inout_reference_out;
public:
  Server_impl();
  virtual ~Server_impl();

  virtual int return_int();
  virtual void in_int(int a);
  virtual void out_int(int& a);
  virtual void inout_int(int& a);

  virtual TestEnum return_enum();
  virtual void in_enum(TestEnum a);
  virtual void out_enum(TestEnum& a);
  virtual void inout_enum(TestEnum& a);

  virtual string return_string();
  virtual void in_string(const string& a);
  virtual void out_string(string& a);
  virtual void inout_string(string& a);

  virtual ref::pointer return_ref();
  virtual void in_ref(const ref::pointer& a);
  virtual void out_ref(ref::pointer& a);
  virtual void inout_ref(ref::pointer& a);

  virtual array1<int> return_array();
  virtual void in_array(const array1<int>& a);
  virtual void out_array(array1<int>& a);
  virtual void inout_array(array1<int>& a);

  virtual array1<bool> return_arraybool();
  virtual void in_arraybool(const array1<bool>& a);
  virtual void out_arraybool(array1<bool>& a);
  virtual void inout_arraybool(array1<bool>& a);

  virtual array1<string> return_arraystring();
  virtual void in_arraystring(const array1<string>& a);
  virtual void out_arraystring(array1<string>& a);
  virtual void inout_arraystring(array1<string>& a);

  virtual array1<array1<int> > return_arrayarray();
  virtual void in_arrayarray(const array1<array1<int> >& a);
  virtual void out_arrayarray(array1<array1<int> >& a);
  virtual void inout_arrayarray(array1<array1<int> >& a);

  virtual array1<ref::pointer> return_arrayref();
  virtual void in_arrayref(const array1<ref::pointer>& a);
  virtual void out_arrayref(array1<ref::pointer>& a);
  virtual void inout_arrayref(array1<ref::pointer>& a);

  bool getSuccess();
};

Server_impl::Server_impl()
{
  success=true;
  return_reference=new ref_impl(10);
  out_reference=new ref_impl(12);
  inout_reference_out=new ref_impl(14);
}

Server_impl::~Server_impl()
{
}

int Server_impl::return_int()
{
  return 5;
}

void Server_impl::in_int(int a)
{
  if(a != 6)
    success=false;
}

void Server_impl::out_int(int& a)
{
  a=7;
}

void Server_impl::inout_int(int& a)
{
  if(a != 8)
    success=false;
  a=9;
}

TestEnum Server_impl::return_enum()
{
  return Foo;
}

void Server_impl::in_enum(TestEnum a)
{
  if(a != Foo)
    success=false;
}

void Server_impl::out_enum(TestEnum& a)
{
  a=Bar;
}

void Server_impl::inout_enum(TestEnum& a)
{
  if(a != Foo)
    success=false;
  a=Bar;
}


string Server_impl::return_string()
{
  return "return string";
}

void Server_impl::in_string(const string& a)
{
  if(a != "in string")
    success=false;
}

void Server_impl::out_string(string& a)
{
  a="out string";
}

void Server_impl::inout_string(string& a)
{
  if(a != "inout string in")
    success=false;
  a="inout string out";
}

ref::pointer Server_impl::return_ref()
{
  return return_reference;
}

void Server_impl::in_ref(const ref::pointer& a)
{
  cerr << "Server_impl::in_ref - 11111\n"; 
  if(a->test() != 11)
    success=false;
  cerr << "Server_impl::in_ref - 22222\n";
}

void Server_impl::out_ref(ref::pointer& a)
{
  a=out_reference;
}

void Server_impl::inout_ref(ref::pointer& a)
{
  if(a->test() != 13)
    success=false;
  a=inout_reference_out;
}

array1<int> Server_impl::return_array()
{
  array1<int> ret;
  init(ret, 10);
  return ret;
}

void Server_impl::in_array(const array1<int>& a)
{
  if(!test(a, 11))
    success=false;
}

void Server_impl::out_array(array1<int>& a)
{
  init(a, 12);
}

void Server_impl::inout_array(array1<int>& a)
{
  if(!test(a, 13))
    success=false;
  init(a, 14);
}

array1<string> Server_impl::return_arraystring()
{
  array1<string> ret;
  init(ret, 10);
  return ret;
}

void Server_impl::in_arraystring(const array1<string>& a)
{
  if(!test(a, 11))
    success=false;
}

void Server_impl::out_arraystring(array1<string>& a)
{
  init(a, 12);
}

void Server_impl::inout_arraystring(array1<string>& a)
{
  if(!test(a, 13))
    success=false;
  init(a, 14);
}

array1<bool> Server_impl::return_arraybool()
{
  array1<bool> ret;
  init(ret, 10);
  return ret;
}

void Server_impl::in_arraybool(const array1<bool>& a)
{
  if(!test(a, 11))
    success=false;
}

void Server_impl::out_arraybool(array1<bool>& a)
{
  init(a, 12);
}

void Server_impl::inout_arraybool(array1<bool>& a)
{
  if(!test(a, 13))
    success=false;
  init(a, 14);
}

array1<array1<int> > Server_impl::return_arrayarray()
{
  array1<array1<int> > ret;
  init(ret, 10);
  return ret;
}

void Server_impl::in_arrayarray(const array1<array1<int> >& a)
{
  if(!test(a, 11))
    success=false;
}

void Server_impl::out_arrayarray(array1<array1<int> >& a)
{
  init(a, 12);
}

void Server_impl::inout_arrayarray(array1<array1<int> >& a)
{
  if(!test(a, 13))
    success=false;
  init(a, 14);
}

static void init(array1<ref::pointer>& a, int s)
{
  a.resize(s);
  for(int i=0;i<s;i++)
    a[i]=new ref_impl(s+i);
}

static bool test(const array1<ref::pointer>& a, int s)
{
  if(int(a.size()) != s)
    return false;
  for(int i=0;i<s;i++){
    if(a[i]->test() != s+i)
      return false;
  }
  return true;
}

array1<ref::pointer> Server_impl::return_arrayref()
{
  array1<ref::pointer> ret;
  init(ret, 10);
  return ret;
}

void Server_impl::in_arrayref(const array1<ref::pointer>& a)
{
  if(!test(a, 11))
    success=false;
}

void Server_impl::out_arrayref(array1<ref::pointer>& a)
{
  init(a, 12);
}

void Server_impl::inout_arrayref(array1<ref::pointer>& a)
{
  if(!test(a, 13))
    success=false;
  init(a, 14);
}

bool Server_impl::getSuccess()
{
  return success;
}

static void fail(char* why)
{
  cerr << "Failure: " << why << '\n';
  Thread::exitAll(1);
}

static void usage(char* progname)
{
  cerr << "usage: " << progname << " [options]\n";
  cerr << "valid options are:\n";
  cerr << "  -server  - server process\n";
  cerr << "  -client URL  - client process\n";
  cerr << "  -reps N - do test N times\n";
  cerr << "\n";
  Thread::exitAll(1);
}

int main(int argc, char* argv[])
{
  using std::string;

  //  usleep( 900000 );

  bool client=false;
  bool server=false;
  string client_url;
  int reps=1;

  for(int i=1;i<argc;i++){
    string arg(argv[i]);
    if(arg == "-server"){
      if(client)
	usage(argv[0]);
      server=true;
    } else if(arg == "-client"){
      if(server)
	usage(argv[0]);
      if(++i>=argc)
	usage(argv[0]);
      client_url=argv[i];
      client=true;
    } else if(arg == "-reps"){
      if(++i>=argc)
	usage(argv[0]);
      reps=atoi(argv[i]);
    } else {
      usage(argv[0]);
    }
  }
  if(!client && !server)
    usage(argv[0]);

  try {
    PIDL::initialize();
    sleep( 1 ); // Give threads enough time to come up.

    Server::pointer pp;
    if(server) {
      pp=Server::pointer(new Server_impl);
      cerr << "Waiting for argtest connections...\n";
      cerr << pp->getURL().getString() << '\n';
    } else {
      double stime=Time::currentSeconds();
      Object::pointer obj=PIDL::objectFrom(client_url);
      Server::pointer rm=pidl_cast<Server::pointer>(obj);
      for(int i=0;i<reps;i++){
	if(rm->return_int() != 5)
	  fail("return_int");
	rm->in_int(6);
	int test_int;
	rm->out_int(test_int);
	if(test_int != 7)
	  fail("out_int");
	test_int=8;
	rm->inout_int(test_int);
	if(test_int != 9)
	  fail("inout_int");
	if(!rm->getSuccess())
	  fail("int failure on remote side");
	  
	if(rm->return_enum() != Foo)
	  fail("return_enum");
	rm->in_enum(Foo);
	TestEnum test_enum;
	rm->out_enum(test_enum);
	if(test_enum != Bar)
	  fail("out_enum");
	test_enum=Foo;
	rm->inout_enum(test_enum);
	if(test_enum != Bar)
	  fail("inout_enum");
	if(!rm->getSuccess())
	  fail("enum failure on remote side");
	  
	if(rm->return_string() != "return string")
	  fail("return_string");
	rm->in_string("in string");
	string test_string;
	rm->out_string(test_string);
	if(test_string != "out string")
	  fail("out_string");
	test_string="inout string in";
	rm->inout_string(test_string);
	if(test_string != "inout string out")
	  fail("inout_string");
	if(!rm->getSuccess())
	  fail("string failure on remote side");
	  
	if(rm->return_ref()->test() != 10)
	  fail("return_ref");
	rm->in_ref(ref::pointer(new ref_impl(11)));
	ref::pointer test_ref;
	rm->out_ref(test_ref);
	if(test_ref->test() != 12)
	  fail("out_ref");
	test_ref=new ref_impl(13);
	rm->inout_ref(test_ref);
	if(test_ref->test() != 14)
	  fail("inout_ref");
	if(!rm->getSuccess())
	  fail("ref failure on remote side");
	  
	array1<int> test_array=rm->return_array();
	if(!test(test_array, 10))
	  fail("return_array");
	init(test_array, 11);
	rm->in_array(test_array);
	rm->out_array(test_array);
	if(!test(test_array, 12))
	  fail("out_array");
	init(test_array, 13);
	rm->inout_array(test_array);
	if(!test(test_array, 14))
	  fail("inout_array");
	if(!rm->getSuccess())
	  fail("array failure on remote side");
	  
	array1<bool> test_arraybool=rm->return_arraybool();
	if(!test(test_arraybool, 10))
	  fail("return_arraybool");
	init(test_arraybool, 11);
	rm->in_arraybool(test_arraybool);
	rm->out_arraybool(test_arraybool);
	if(!test(test_arraybool, 12))
	  fail("out_arraybool");
	init(test_arraybool, 13);
	rm->inout_arraybool(test_arraybool);
	if(!test(test_arraybool, 14))
	  fail("inout_arraybool");
	if(!rm->getSuccess())
	  fail("arraybool failure on remote side");
	  
	array1<string> test_arraystring=rm->return_arraystring();
	if(!test(test_arraystring, 10))
	  fail("return_arraystring");
	init(test_arraystring, 11);
	rm->in_arraystring(test_arraystring);
	rm->out_arraystring(test_arraystring);
	if(!test(test_arraystring, 12))
	  fail("out_arraystring");
	init(test_arraystring, 13);
	rm->inout_arraystring(test_arraystring);
	if(!test(test_arraystring, 14))
	  fail("inout_arraystring");
	if(!rm->getSuccess())
	  fail("arraystring failure on remote side");
	  
	array1<array1<int> > test_arrayarray=rm->return_arrayarray();
	if(!test(test_arrayarray, 10))
	  fail("return_arrayarray");
	init(test_arrayarray, 11);
	rm->in_arrayarray(test_arrayarray);
	rm->out_arrayarray(test_arrayarray);
	if(!test(test_arrayarray, 12))
	  fail("out_arrayarray");
	init(test_arrayarray, 13);
	rm->inout_arrayarray(test_arrayarray);
	if(!test(test_arrayarray, 14))
	  fail("inout_arrayarray");
	if(!rm->getSuccess())
	  fail("arrayarray failure on remote side");
	  
	array1<ref::pointer> test_arrayref=rm->return_arrayref();
	if(!test(test_arrayref, 10))
	  fail("return_arrayref");
	init(test_arrayref, 11);
	rm->in_arrayref(test_arrayref);
	rm->out_arrayref(test_arrayref);
	if(!test(test_arrayref, 12))
	  fail("out_arrayref");
	init(test_arrayref, 13);
	rm->inout_arrayref(test_arrayref);
	if(!test(test_arrayref, 14))
	  fail("inout_arrayref");
	if(!rm->getSuccess())
	  fail("arrayref failure on remote side");
      }
      double dt=Time::currentSeconds()-stime;
      cerr << "argtest: " << reps << " reps in " << dt << " seconds\n";
      double us=dt/reps*1000*1000;
      cerr << "argtest: " << us << " us/rep\n";
    }
    PIDL::serveObjects();
    PIDL::finalize();
    cerr << "Argtest successful\n";
  } catch(const Exception& e) {
    cerr << "argtest caught exception:\n";
    cerr << e.message() << '\n';
    //Thread::exitAll(1);
  } catch(...) {
    cerr << "Caught unexpected exception!\n";
    //Thread::exitAll(1);
  }
  return 0;
}

