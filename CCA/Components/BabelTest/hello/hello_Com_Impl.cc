// 
// File:          hello_Com_Impl.cc
// Symbol:        hello.Com-v1.0
// Symbol Type:   class
// Babel Version: 0.7.4
// SIDL Created:  20030618 13:12:37 MDT
// Generated:     20030618 13:12:40 MDT
// Description:   Server-side implementation for hello.Com
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// babel-version = 0.7.4
// source-line   = 14
// source-url    = file:/home/sci/kzhang/SCIRun/src/CCA/Components/BabelTest/hello/hello.sidl
// 
#include "hello_Com_Impl.hh"

// DO-NOT-DELETE splicer.begin(hello.Com._includes)
// Put additional includes or other arbitrary code here...
#include "hello.hh"
// DO-NOT-DELETE splicer.end(hello.Com._includes)

// user defined constructor
void hello::Com_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(hello.Com._ctor)
  // add construction details here
  // DO-NOT-DELETE splicer.end(hello.Com._ctor)
}

// user defined destructor
void hello::Com_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(hello.Com._dtor)
  // add destruction details here
  // DO-NOT-DELETE splicer.end(hello.Com._dtor)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Obtain Services handle, through which the 
 * component communicates with the framework. 
 * This is the one method that every CCA Component
 * must implement. 
 */
void
hello::Com_impl::setServices (
  /*in*/ ::gov::cca::Services services ) 
throw () 
{
  // DO-NOT-DELETE splicer.begin(hello.Com.setServices)
  svc=services;
  hello::GoPort goPort=hello::GoPort::_create();
  goPort.setService(svc);
  svc.addProvidesPort(goPort,"go","gov.cca.ports.GoPort",0);
  svc.registerUsesPort("idport","gov.cca.ports.IDPort",0);
  // DO-NOT-DELETE splicer.end(hello.Com.setServices)
}


// DO-NOT-DELETE splicer.begin(hello.Com._misc)
// DO-NOT-DELETE splicer.end(hello.Com._misc)

