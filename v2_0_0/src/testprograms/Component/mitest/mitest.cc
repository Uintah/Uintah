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
 *  mitest.cc
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 U of U
 */

#include <iostream>
#include <Core/CCA/PIDL/PIDL.h>
#include <testprograms/Component/mitest/mitest_sidl.h>
#include <Core/Thread/Time.h>

using std::cerr;
using std::cout;

using namespace mitest;
using namespace SCIRun;

class mitest_impl : public D {
public:
  mitest_impl();
  virtual ~mitest_impl();
  virtual int a();
  virtual int b();
  virtual int c();
  virtual int d();

  virtual bool isa_mitestA(const A::pointer&);
  virtual bool isa_mitestB(const B::pointer&);
  virtual bool isa_mitestC(const C::pointer&);
  virtual bool isa_mitestD(const D::pointer&);

};

mitest_impl::mitest_impl()
{
}

mitest_impl::~mitest_impl()
{
}

int mitest_impl::a()
{
  return 1;
}

int mitest_impl::b()
{
  return 2;
}

int mitest_impl::c()
{
  return 3;
}

int mitest_impl::d()
{
  return 4;
}

bool mitest_impl::isa_mitestA(const A::pointer& p)
{
  if(dynamic_cast<mitest_impl*>(p.getPointer()))
    return true;
  else
    return false;
}

bool mitest_impl::isa_mitestB(const B::pointer& p)
{
  if(dynamic_cast<mitest_impl*>(p.getPointer()))
    return true;
  else
    return false;
}

bool mitest_impl::isa_mitestC(const C::pointer& p)
{
  if(dynamic_cast<mitest_impl*>(p.getPointer()))
    return true;
  else
    return false;
}

bool mitest_impl::isa_mitestD(const D::pointer& p)
{
  if(dynamic_cast<mitest_impl*>(p.getPointer()))
    return true;
  else
    return false;
}

void usage(char* progname)
{
  cerr << "usage: " << progname << " [options]\n";
  cerr << "valid options are:\n";
  cerr << "  -server  - server process\n";
  cerr << "  -client URL  - client process\n";
  cerr << "\n";
  exit(1);
}

static void test_d(D::pointer d, bool& failed)
{
  if(d.isNull()){
    cerr << "Wrong object type!\n";
    abort();
  }
  int aa=d->a();
  int bb=d->b();
  int cc=d->c();
  int dd=d->d();
  if(aa != 1 || bb != 2 || cc != 3 || dd != 4){
    cerr << "Wrong answer for interface D:\n";
    cerr << "a=" << aa << '\n';
    cerr << "b=" << bb << '\n';
    cerr << "c=" << cc << '\n';
    cerr << "d=" << dd << '\n';
    failed=true;
  }
  if(!d->isa_mitestD(d)){
    cerr << "isa_mitest(D) failed!\n";
    failed=true;
  }
}

static void test_c(C::pointer c, bool& failed)
{
  if(c.isNull()){
    cerr << "Wrong object type!\n";
    abort();
  }
  int aa=c->a();
  int cc=c->c();
  if(aa != 1 || cc != 3){
    cerr << "Wrong answer for interface C:\n";
    cerr << "a=" << aa << '\n';
    cerr << "c=" << cc << '\n';
    failed=true;
  }
  if(!c->isa_mitestC(c)){
    cerr << "isa_mitest(C) failed!\n";
    failed=true;
  }
}

static void test_b(B::pointer b, bool& failed)
{
  if(b.isNull()){
    cerr << "Wrong object type!\n";
    abort();
  }
  int aa=b->a();
  int bb=b->b();
  if(aa != 1 || bb != 2){
    cerr << "Wrong answer for interface B:\n";
    cerr << "a=" << aa << '\n';
    cerr << "b=" << bb << '\n';
    failed=true;
  }
  if(!b->isa_mitestB(b)){
    cerr << "isa_mitest(B) failed!\n";
    failed=true;
  }
}

static void test_a(A::pointer a, bool& failed)
{
  if(a.isNull()){
    cerr << "Wrong object type!\n";
    abort();
  }
  int aa=a->a();
  if(aa != 1){
    cerr << "Wrong answer for interface A:\n";
    cerr << "a=" << aa << '\n';
    failed=true;
  }
  if(!a->isa_mitestA(a)){
    cerr << "isa_mitest(A) failed!\n";
    failed=true;
  }
}

int main(int argc, char* argv[])
{
  using std::string;

  try {
    PIDL::initialize();

    bool client=false;
    bool server=false;
    string client_url;

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
      } else {
	usage(argv[0]);
      }
    }
    if(!client && !server)
      usage(argv[0]);

    if(server) {
      cerr << "Creating mitest object\n";
      mitest_impl* pp=new mitest_impl;
      cerr << "Waiting for mitest connections...\n";
      cerr << pp->getURL().getString() << '\n';
    } else {
      Object::pointer obj=PIDL::objectFrom(client_url);
      bool failed=false;

      // From base Object
      D::pointer d=pidl_cast<D::pointer>(obj);
      test_d(d, failed);
      test_b(d, failed);
      test_c(d, failed);
      test_a(d, failed);

      C::pointer c=pidl_cast<C::pointer>(obj);
      test_c(c, failed);
      test_a(c, failed);

      B::pointer b=pidl_cast<B::pointer>(obj);
      test_b(b, failed);
      test_a(b, failed);

      A::pointer a=pidl_cast<A::pointer>(obj);
      test_a(a, failed);

      // From A
      A::pointer a_from_a=pidl_cast<A::pointer>(a);
      test_a(a_from_a, failed);

      B::pointer b_from_a=pidl_cast<B::pointer>(a);
      test_b(b_from_a, failed);
      test_a(b_from_a, failed);

      C::pointer c_from_a=pidl_cast<C::pointer>(a);
      test_c(c_from_a, failed);
      test_a(c_from_a, failed);

      D::pointer d_from_a=pidl_cast<D::pointer>(a);
      test_d(d_from_a, failed);
      test_c(d_from_a, failed);
      test_b(d_from_a, failed);
      test_a(d_from_a, failed);

      // From B
      A::pointer a_from_b=pidl_cast<A::pointer>(b);
      test_a(a_from_b, failed);

      B::pointer b_from_b=pidl_cast<B::pointer>(b);
      test_b(b_from_b, failed);
      test_a(b_from_b, failed);

      C::pointer c_from_b=pidl_cast<C::pointer>(b);
      test_c(c_from_b, failed);
      test_a(c_from_b, failed);

      D::pointer d_from_b=pidl_cast<D::pointer>(b);
      test_d(d_from_b, failed);
      test_c(c_from_b, failed);
      test_b(b_from_b, failed);
      test_a(a_from_b, failed);

      // From C
      A::pointer a_from_c=pidl_cast<A::pointer>(c);
      test_a(a_from_c, failed);

      B::pointer b_from_c=pidl_cast<B::pointer>(c);
      test_b(b_from_c, failed);
      test_a(b_from_c, failed);

      C::pointer c_from_c=pidl_cast<C::pointer>(c);
      test_c(c_from_c, failed);
      test_a(c_from_c, failed);

      D::pointer d_from_c=pidl_cast<D::pointer>(c);
      test_d(d_from_c, failed);
      test_c(c_from_c, failed);
      test_b(b_from_c, failed);
      test_a(a_from_c, failed);
	    
      // From D
      A::pointer a_from_d=pidl_cast<A::pointer>(d);
      test_a(a_from_d, failed);

      B::pointer b_from_d=pidl_cast<B::pointer>(d);
      test_b(b_from_d, failed);
      test_a(b_from_d, failed);

      C::pointer c_from_d=pidl_cast<C::pointer>(d);
      test_c(c_from_d, failed);
      test_a(c_from_d, failed);

      D::pointer d_from_d=pidl_cast<D::pointer>(d);
      test_d(d_from_d, failed);
      test_c(c_from_d, failed);
      test_b(b_from_d, failed);
      test_a(a_from_d, failed);

      if(failed){
	cout << "tests failed!\n";
	exit(1);
      } else {
	cout << "tests successful!\n";
      }
    }
    PIDL::serveObjects();
    PIDL::finalize();
  } catch(const Exception& e) {
    cerr << "Caught exception:\n";
    cerr << e.message() << '\n';
    abort();
  } catch(...) {
    cerr << "Caught unexpected exception!\n";
    abort();
  }
  return 0;
}
