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
#include <Core/CCA/Component/PIDL/PIDL.h>
#include <testprograms/Component/mitest/mitest_sidl.h>
#include <Core/Thread/Time.h>

using std::cerr;
using std::cout;

using namespace mitest;
using namespace SCIRun;

class mitest_impl : public D_interface {
public:
    mitest_impl();
    virtual ~mitest_impl();
    virtual int a();
    virtual int b();
    virtual int c();
    virtual int d();

    virtual bool isa_mitest(const A&);
    virtual bool isa_mitest(const B&);
    virtual bool isa_mitest(const C&);
    virtual bool isa_mitest(const D&);

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

bool mitest_impl::isa_mitest(const A& p)
{
    if(dynamic_cast<mitest_impl*>(p.getPointer()))
	return true;
    else
	return false;
}

bool mitest_impl::isa_mitest(const B& p)
{
    if(dynamic_cast<mitest_impl*>(p.getPointer()))
	return true;
    else
	return false;
}

bool mitest_impl::isa_mitest(const C& p)
{
    if(dynamic_cast<mitest_impl*>(p.getPointer()))
	return true;
    else
	return false;
}

bool mitest_impl::isa_mitest(const D& p)
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

static void test_d(D d, bool& failed)
{
    if(!d){
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
    if(!d->isa_mitest(d)){
	cerr << "isa_mitest(D) failed!\n";
	failed=true;
    }
}

static void test_c(C c, bool& failed)
{
    if(!c){
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
    if(!c->isa_mitest(c)){
	cerr << "isa_mitest(C) failed!\n";
	failed=true;
    }
}

static void test_b(B b, bool& failed)
{
    if(!b){
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
    if(!b->isa_mitest(b)){
	cerr << "isa_mitest(B) failed!\n";
	failed=true;
    }
}

static void test_a(A a, bool& failed)
{
    if(!a){
	cerr << "Wrong object type!\n";
	abort();
    }
    int aa=a->a();
    if(aa != 1){
	cerr << "Wrong answer for interface A:\n";
	cerr << "a=" << aa << '\n';
	failed=true;
    }
    if(!a->isa_mitest(a)){
	cerr << "isa_mitest(A) failed!\n";
	failed=true;
    }
}

int main(int argc, char* argv[])
{
    using std::string;

    try {
	PIDL::PIDL::initialize(argc, argv);

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
	    PIDL::Object obj=PIDL::PIDL::objectFrom(client_url);
	    bool failed=false;

	    // From base Object
	    D d=pidl_cast<D>(obj);
	    test_d(d, failed);
	    test_b(d, failed);
	    test_c(d, failed);
	    test_a(d, failed);

	    C c=pidl_cast<C>(obj);
	    test_c(c, failed);
	    test_a(c, failed);

	    B b=pidl_cast<B>(obj);
	    test_b(b, failed);
	    test_a(b, failed);

	    A a=pidl_cast<A>(obj);
	    test_a(a, failed);

	    // From A
	    A a_from_a=pidl_cast<A>(a);
	    test_a(a_from_a, failed);

	    B b_from_a=pidl_cast<B>(a);
	    test_b(b_from_a, failed);
	    test_a(b_from_a, failed);

	    C c_from_a=pidl_cast<C>(a);
	    test_c(c_from_a, failed);
	    test_a(c_from_a, failed);

	    D d_from_a=pidl_cast<D>(a);
	    test_d(d_from_a, failed);
	    test_c(d_from_a, failed);
	    test_b(d_from_a, failed);
	    test_a(d_from_a, failed);

	    // From B
	    A a_from_b=pidl_cast<A>(b);
	    test_a(a_from_b, failed);

	    B b_from_b=pidl_cast<B>(b);
	    test_b(b_from_b, failed);
	    test_a(b_from_b, failed);

	    C c_from_b=pidl_cast<C>(b);
	    test_c(c_from_b, failed);
	    test_a(c_from_b, failed);

	    D d_from_b=pidl_cast<D>(b);
	    test_d(d_from_b, failed);
	    test_c(c_from_b, failed);
	    test_b(b_from_b, failed);
	    test_a(a_from_b, failed);

	    // From C
	    A a_from_c=pidl_cast<A>(c);
	    test_a(a_from_c, failed);

	    B b_from_c=pidl_cast<B>(c);
	    test_b(b_from_c, failed);
	    test_a(b_from_c, failed);

	    C c_from_c=pidl_cast<C>(c);
	    test_c(c_from_c, failed);
	    test_a(c_from_c, failed);

	    D d_from_c=pidl_cast<D>(c);
	    test_d(d_from_c, failed);
	    test_c(c_from_c, failed);
	    test_b(b_from_c, failed);
	    test_a(a_from_c, failed);
	    
	    // From D
	    A a_from_d=pidl_cast<A>(d);
	    test_a(a_from_d, failed);

	    B b_from_d=pidl_cast<B>(d);
	    test_b(b_from_d, failed);
	    test_a(b_from_d, failed);

	    C c_from_d=pidl_cast<C>(d);
	    test_c(c_from_d, failed);
	    test_a(c_from_d, failed);

	    D d_from_d=pidl_cast<D>(d);
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
	PIDL::PIDL::serveObjects();
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

