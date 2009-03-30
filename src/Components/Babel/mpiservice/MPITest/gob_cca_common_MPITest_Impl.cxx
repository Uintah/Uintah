// 
// File:          gob_cca_common_MPITest_Impl.cxx
// Symbol:        gob.cca.common.MPITest-v0.0
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for gob.cca.common.MPITest
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "gob_cca_common_MPITest_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_CCAException_hxx
#include "gov_cca_CCAException.hxx"
#endif
#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_RuntimeException_hxx
#include "sidl_RuntimeException.hxx"
#endif
#ifndef included_sidl_NotImplementedException_hxx
#include "sidl_NotImplementedException.hxx"
#endif
// DO-NOT-DELETE splicer.begin(gob.cca.common.MPITest._includes)

#include <iostream>
#include <mpi.h>
#include <gob_cca_ports.hxx>

#define _BOCCA_STDERR 1
  // Insert-UserCode-Here {gob.cca.common.MPITest._includes:prolog} (additional includes or code)

  // Bocca generated code. bocca.protected.begin(gob.cca.common.MPITest._includes)

#define _BOCCA_CTOR_MESSAGES 0
  // If -D_BOCCA_STDERR is given to the compiler, diagnostics print to stderr.
  // In production use, probably want not to use -D_BOCCA_STDERR.
#ifdef _BOCCA_STDERR

#include <iostream>

#ifdef _BOCCA_CTOR_PRINT
#undef _BOCCA_CTOR_MESSAGES
#define _BOCCA_CTOR_MESSAGES 1
#endif // _BOCCA_CTOR_PRINT 
#else  // _BOCCA_STDERR


#endif // _BOCCA_STDERR



  // If -D_BOCCA_BOOST is given to the compiler, exceptions and diagnostics 
  // will include function names for boost-understood compilers.
  // If boost is not available (and therefore ccaffeine is not in use), 
  // -D_BOCCA_BOOST can be omitted and function names will not be included in 
  // messages.
#ifndef _BOCCA_BOOST
#define BOOST_CURRENT_FUNCTION ""
#else
#include <boost/current_function.hpp>
#endif

  // This is intended to simplify exception throwing as SIDL_THROW does for C.
#define BOCCA_THROW_CXX(EX_CLS, MSG) \
{ \
    EX_CLS ex = EX_CLS::_create(); \
    ex.setNote( MSG ); \
    ex.add(__FILE__, __LINE__, BOOST_CURRENT_FUNCTION); \
    throw ex; \
}

  // This simplifies exception extending and rethrowing in c++, like 
  // SIDL_CHECK in C. EX_OBJ must be the caught exception and is extended with 
  // msg and file/line/func added. Continuing the throw is up to the user.
#define BOCCA_EXTEND_THROW_CXX(EX_OBJ, MSG, LINEOFFSET) \
{ \
  std::string msg = std::string(MSG) + std::string(BOOST_CURRENT_FUNCTION); \
  EX_OBJ.add(__FILE__,__LINE__ + LINEOFFSET, msg); \
}


  // Bocca generated code. bocca.protected.end(gob.cca.common.MPITest._includes)

  // Insert-UserCode-Here {gob.cca.common.MPITest._includes:epilog} (additional includes or code)

// DO-NOT-DELETE splicer.end(gob.cca.common.MPITest._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
gob::cca::common::MPITest_impl::MPITest_impl() : StubBase(reinterpret_cast< 
  void*>(::gob::cca::common::MPITest::_wrapObj(reinterpret_cast< void*>(this))),
  false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPITest._ctor2)
  // Insert-Code-Here {gob.cca.common.MPITest._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPITest._ctor2)
}

// user defined constructor
void gob::cca::common::MPITest_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPITest._ctor)
    
  // Insert-UserCode-Here {gob.cca.common.MPITest._ctor:prolog} (constructor method) 

  // bocca-default-code. User may edit or delete.begin(gob.cca.common.MPITest._ctor)
   #if _BOCCA_CTOR_MESSAGES

     std::cerr << "CTOR gob.cca.common.MPITest: " << BOOST_CURRENT_FUNCTION 
               << " constructing " << this << std::endl;

   #endif // _BOCCA_CTOR_MESSAGES
  // bocca-default-code. User may edit or delete.end(gob.cca.common.MPITest._ctor)

  // Insert-UserCode-Here {gob.cca.common.MPITest._ctor:epilog} (constructor method)

  // DO-NOT-DELETE splicer.end(gob.cca.common.MPITest._ctor)
}

// user defined destructor
void gob::cca::common::MPITest_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPITest._dtor)
  // Insert-UserCode-Here {gob.cca.common.MPITest._dtor} (destructor method) 
    
  // bocca-default-code. User may edit or delete.begin(gob.cca.common.MPITest._dtor) 
   #if _BOCCA_CTOR_MESSAGES

     std::cerr << "DTOR gob.cca.common.MPITest: " << BOOST_CURRENT_FUNCTION 
               << " destructing " << this << std::endl;

   #endif // _BOCCA_CTOR_MESSAGES 
  // bocca-default-code. User may edit or delete.end(gob.cca.common.MPITest._dtor) 

  // DO-NOT-DELETE splicer.end(gob.cca.common.MPITest._dtor)
}

// static class initializer
void gob::cca::common::MPITest_impl::_load() {
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPITest._load)
  // Insert-Code-Here {gob.cca.common.MPITest._load} (class initialization)
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPITest._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  boccaSetServices[]
 */
void
gob::cca::common::MPITest_impl::boccaSetServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPITest.boccaSetServices)
  // DO-NOT-EDIT-BOCCA
  // Bocca generated code. bocca.protected.begin(gob.cca.common.MPITest.boccaSetServices)

  gov::cca::TypeMap typeMap;
  gov::cca::Port    port;

  this->d_services = services;

  typeMap = this->d_services.createTypeMap();

  port = ::babel_cast< gov::cca::Port>(*this);
  if (port._is_nil()) {
    BOCCA_THROW_CXX( ::sidl::SIDLException , 
                     "gob.cca.common.MPITest: Error casting self to gov::cca::Port");
  } 


  // Provide a gov.cca.ports.GoPort port with port name go 
  try{
    this->d_services.addProvidesPort(
                   port,              // implementing object
                   "go", // port instance name
                   "gov.cca.ports.GoPort",     // full sidl type of port
                   typeMap);          // properties for the port
  } catch ( ::gov::cca::CCAException ex )  {
    BOCCA_EXTEND_THROW_CXX(ex, 
        "gob.cca.common.MPITest: Error calling addProvidesPort(port,"
        "\"go\", \"gov.cca.ports.GoPort\", typeMap) ", -2);
    throw;
  }    

  // Use a gob.cca.ports.MPISetup port with port name commsetup 
  try{
    this->d_services.registerUsesPort(
                   "commsetup", // port instance name
                   "gob.cca.ports.MPISetup",     // full sidl type of port
                    typeMap);         // properties for the port
  } catch ( ::gov::cca::CCAException ex )  {
    BOCCA_EXTEND_THROW_CXX(ex,
       "gob.cca.common.MPITest: Error calling registerUsesPort(\"commsetup\", "
       "\"gob.cca.ports.MPISetup\", typeMap) ", -2);
    throw;
  }

  // Use a gob.cca.ports.MPIService port with port name commsource 
  try{
    this->d_services.registerUsesPort(
                   "commsource", // port instance name
                   "gob.cca.ports.MPIService",     // full sidl type of port
                    typeMap);         // properties for the port
  } catch ( ::gov::cca::CCAException ex )  {
    BOCCA_EXTEND_THROW_CXX(ex,
       "gob.cca.common.MPITest: Error calling registerUsesPort(\"commsource\", "
       "\"gob.cca.ports.MPIService\", typeMap) ", -2);
    throw;
  }


  gov::cca::ComponentRelease cr = 
        ::babel_cast< gov::cca::ComponentRelease>(*this);
  this->d_services.registerForRelease(cr);
  return;
  // Bocca generated code. bocca.protected.end(gob.cca.common.MPITest.boccaSetServices)
    
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPITest.boccaSetServices)
}

/**
 * Method:  boccaReleaseServices[]
 */
void
gob::cca::common::MPITest_impl::boccaReleaseServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPITest.boccaReleaseServices)
  // DO-NOT-EDIT-BOCCA
  // Bocca generated code. bocca.protected.begin(gob.cca.common.MPITest.boccaReleaseServices)
  this->d_services=0;


  // Un-provide gov.cca.ports.GoPort port with port name go 
  try{
    services.removeProvidesPort("go");
  } catch ( ::gov::cca::CCAException ex )  {

#ifdef _BOCCA_STDERR
    std::cerr << "gob.cca.common.MPITest: Error calling removeProvidesPort("
              << "\"go\") at " 
              << __FILE__ << ": " << __LINE__ -4 << ": " << ex.getNote() 
              << std::endl;
#endif // _BOCCA_STDERR

  }

  // Release gob.cca.ports.MPISetup port with port name commsetup 
  try{
    services.unregisterUsesPort("commsetup");
  } catch ( ::gov::cca::CCAException ex )  {

#ifdef _BOCCA_STDERR
    std::cerr << "gob.cca.common.MPITest: Error calling unregisterUsesPort("
              << "\"commsetup\") at " 
              << __FILE__ << ":" << __LINE__ -4 << ": " << ex.getNote() 
              << std::endl;
#endif // _BOCCA_STDERR

  }

  // Release gob.cca.ports.MPIService port with port name commsource 
  try{
    services.unregisterUsesPort("commsource");
  } catch ( ::gov::cca::CCAException ex )  {

#ifdef _BOCCA_STDERR
    std::cerr << "gob.cca.common.MPITest: Error calling unregisterUsesPort("
              << "\"commsource\") at " 
              << __FILE__ << ":" << __LINE__ -4 << ": " << ex.getNote() 
              << std::endl;
#endif // _BOCCA_STDERR

  }

  return;
  // Bocca generated code. bocca.protected.end(gob.cca.common.MPITest.boccaReleaseServices)
    
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPITest.boccaReleaseServices)
}

/**
 *  Starts up a component presence in the calling framework.
 * @param services the component instance's handle on the framework world.
 * Contracts concerning Svc and setServices:
 * 
 * The component interaction with the CCA framework
 * and Ports begins on the call to setServices by the framework.
 * 
 * This function is called exactly once for each instance created
 * by the framework.
 * 
 * The argument Svc will never be nil/null.
 * 
 * Those uses ports which are automatically connected by the framework
 * (so-called service-ports) may be obtained via getPort during
 * setServices.
 */
void
gob::cca::common::MPITest_impl::setServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPITest.setServices)

  // Insert-UserCode-Here{gob.cca.common.MPITest.setServices:prolog}

  // bocca-default-code. User may edit or delete.begin(gob.cca.common.MPITest.setServices)
     boccaSetServices(services); 
  // bocca-default-code. User may edit or delete.end(gob.cca.common.MPITest.setServices)
  
  // Insert-UserCode-Here{gob.cca.common.MPITest.setServices:epilog}

  // DO-NOT-DELETE splicer.end(gob.cca.common.MPITest.setServices)
}

/**
 * Shuts down a component presence in the calling framework.
 * @param services the component instance's handle on the framework world.
 * Contracts concerning Svc and setServices:
 * 
 * This function is called exactly once for each callback registered
 * through Services.
 * 
 * The argument Svc will never be nil/null.
 * The argument Svc will always be the same as that received in
 * setServices.
 * 
 * During this call the component should release any interfaces
 * acquired by getPort().
 * 
 * During this call the component should reset to nil any stored
 * reference to Svc.
 * 
 * After this call, the component instance will be removed from the
 * framework. If the component instance was created by the
 * framework, it will be destroyed, not recycled, The behavior of
 * any port references obtained from this component instance and
 * stored elsewhere becomes undefined.
 * 
 * Notes for the component implementor:
 * 1) The component writer may perform blocking activities
 * within releaseServices, such as waiting for remote computations
 * to shutdown.
 * 2) It is good practice during releaseServices for the component
 * writer to remove or unregister all the ports it defined.
 */
void
gob::cca::common::MPITest_impl::releaseServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPITest.releaseServices)

  // Insert-UserCode-Here {gob.cca.common.MPITest.releaseServices} 

  // bocca-default-code. User may edit or delete.begin(gob.cca.common.MPITest.releaseServices)
     boccaReleaseServices(services);
  // bocca-default-code. User may edit or delete.end(gob.cca.common.MPITest.releaseServices)
    
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPITest.releaseServices)
}

/**
 *  
 * Execute some encapsulated functionality on the component. 
 * Return 0 if ok, -1 if internal error but component may be 
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
gob::cca::common::MPITest_impl::go_impl () 

{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPITest.go)
// User editable portion is in the middle at the next Insert-UserCode-Here line.


// Bocca generated code. bocca.protected.begin(gob.cca.common.MPITest.go:boccaGoProlog)
  int bocca_status = 0;
  // The user's code should set bocca_status 0 if computation proceeded ok.
  // The user's code should set bocca_status -1 if computation failed but might
  // succeed on another call to go(), e.g. when a required port is not yet 
  // connected.
  // The user's code should set bocca_status -2 if the computation failed and 
  // can never succeed in a future call.
  // The user's code should NOT use return in this function.
  // Exceptions that are not caught in user code will be converted to 
  // status -2.

  gov::cca::Port port;

  // nil if not fetched and cast successfully:
  gob::cca::ports::MPISetup commsetup; 
  // True when releasePort is needed (even if cast fails):
  bool commsetup_fetched = false; 
  // nil if not fetched and cast successfully:
  gob::cca::ports::MPIService commsource; 
  // True when releasePort is needed (even if cast fails):
  bool commsource_fetched = false; 
  // Use a gob.cca.ports.MPISetup port with port name commsetup 
  try{
    port = this->d_services.getPort("commsetup");
  } catch ( ::gov::cca::CCAException ex )  {
    // we will continue with port nil (never successfully assigned) and 
    // set a flag.

#ifdef _BOCCA_STDERR
    std::cerr << "gob.cca.common.MPITest: Error calling getPort(\"commsetup\") "
              " at " << __FILE__ << ":" << __LINE__ -5 << ": " << ex.getNote() 
              << std::endl;
#endif // _BOCCA_STDERR

  }
  if ( port._not_nil() ) {
    // even if the next cast fails, must release.
    commsetup_fetched = true; 
    commsetup = ::babel_cast< gob::cca::ports::MPISetup >(port);
    if (commsetup._is_nil()) {

#ifdef _BOCCA_STDERR
      std::cerr << "gob.cca.common.MPITest: Error casting gov::cca::Port "
                << "commsetup to type "
                << "gob::cca::ports::MPISetup" << std::endl;
#endif //_BOCCA_STDERR

      goto BOCCAEXIT; // we cannot correctly continue. clean up and leave.
    } 
  } 

  // Use a gob.cca.ports.MPIService port with port name commsource 
  try{
    port = this->d_services.getPort("commsource");
  } catch ( ::gov::cca::CCAException ex )  {
    // we will continue with port nil (never successfully assigned) and 
    // set a flag.

#ifdef _BOCCA_STDERR
    std::cerr << "gob.cca.common.MPITest: Error calling getPort(\"commsource\") "
              " at " << __FILE__ << ":" << __LINE__ -5 << ": " << ex.getNote() 
              << std::endl;
#endif // _BOCCA_STDERR

  }
  if ( port._not_nil() ) {
    // even if the next cast fails, must release.
    commsource_fetched = true; 
    commsource = ::babel_cast< gob::cca::ports::MPIService >(port);
    if (commsource._is_nil()) {

#ifdef _BOCCA_STDERR
      std::cerr << "gob.cca.common.MPITest: Error casting gov::cca::Port "
                << "commsource to type "
                << "gob::cca::ports::MPIService" << std::endl;
#endif //_BOCCA_STDERR

      goto BOCCAEXIT; // we cannot correctly continue. clean up and leave.
    } 
  } 


// Bocca generated code. bocca.protected.end(gob.cca.common.MPITest.go:boccaGoProlog)



  // When this try/catch block is rewritten by the user, we will not change it.
  try {

    // All port instances should be rechecked for ._not_nil before calling in 
    // user code. Not all ports need be connected in arbitrary use.
    // The uses ports appear as local variables here named exactly as on the 
    // bocca commandline.


    // first we demo the mpisetup port, which the test script must have connected.
    // This assumes that we are serving mpi_comm_world. An alternate driver component
    // can do otherwise. We expect the bocca getport for service will have failed
    // since the server isn't running yet.

    if (commsetup._is_nil()) {
      bocca_status = -1;
    } else {
      int mi=0;
      int merror = MPI_SUCCESS;
      merror = MPI_Initialized(&mi);
      if (merror != MPI_SUCCESS) {
        bocca_status = -2;
        goto BOCCAEXIT; // we cannot correctly continue. clean up and leave.
      }
      if (! mi ) {
	std::cerr << "1111111111\n";
        merror = MPI_Init(NULL,NULL);
	std::cerr << "222222222222\n";
        if (merror != MPI_SUCCESS) {
          bocca_status = -2;
          goto BOCCAEXIT; // we cannot correctly continue. clean up and leave.
        }
      }

      MPI_Initialized(&mi);
      if (! mi) {
	std::cerr << "I am not initialized\n";
      }
      else {
	std::cerr << "I am initialized\n";
      }

      MPI_Fint server_fcomm = MPI_Comm_c2f(MPI_COMM_WORLD); // may be 4 or 8 bytes
      int64_t server_comm = (int64_t) server_fcomm; // always 8 bytes in sid
      commsetup.initAsService(server_comm);
      try{
        port = this->d_services.getPort("commsource");
      } catch ( ::gov::cca::CCAException ex )  {
        bocca_status = -2;
      }
      if ( port._not_nil() ) {
        // even if the next cast fails, must release.
        commsource_fetched = true; 
        commsource = ::babel_cast< gob::cca::ports::MPIService >(port);
        if (commsource._is_nil()) {
#ifdef _BOCCA_STDERR
          std::cerr << "gob.cca.common.MPITest: Error casting gov::cca::Port "
                << "commsource to type "
                << "gob::cca::ports::MPIService" << std::endl;
#endif //_BOCCA_STDERR
          bocca_status = -2;
        } else {
	  std::cerr << "4444444444\n";
          int64_t sComm = commsource.getComm();
	  std::cerr << "5555555555\n";
          if (sComm == 0) {
#ifdef _BOCCA_STDERR
            bocca_status = -2;
            std::cerr << "gob.cca.common.MPITest: Error fetching from "
                << "commsource" <<  std::endl;
#endif //_BOCCA_STDERR
            goto BOCCAEXIT; // we cannot correctly continue. clean up and leave.
          }
	  std::cerr << "66666666666\n";
          MPI_Fint fsComm = (MPI_Fint)sComm;
          MPI_Comm cComm = MPI_Comm_f2c(fsComm);
          if (cComm == MPI_COMM_NULL) {
            bocca_status = -2;
#ifdef _BOCCA_STDERR
            std::cerr << "gob.cca.common.MPITest: got MPI_COMM_NULL from "
                << "commsource" <<  std::endl;
#endif //_BOCCA_STDERR
            goto BOCCAEXIT; // we cannot correctly continue. clean up and leave.
          }

          int srank = -1;
	  std::cerr << "FORE RANK\n";
          merror = MPI_Comm_rank(cComm, &srank);
	  std::cerr << "AFT RANK\n";
          if (merror != 0) {
            bocca_status = -2;
          } else {
            bocca_status = 0;
          }
          std::cout << "found received communicator has rank = " << srank << std::endl;
          commsource.releaseComm(sComm);
        }
      }
    }
  } 
  // If unknown exceptions in the user code are tolerable and restart is ok, 
  // return -1 instead. -2 means the component is so confused that it and 
  // probably the application should be destroyed.
  // babel requires exact exception catching due to c++ binding of interfaces.
  catch (gov::cca::CCAException ex) {
    bocca_status = -2;
    std::string enote = ex.getNote();

#ifdef _BOCCA_STDERR
    std::cerr << "CCAException in user go code: " << enote << std::endl;
    std::cerr << "Returning -2 from go()" << std::endl;;
#endif

  }
  catch (sidl::RuntimeException ex) {
    bocca_status = -2;
    std::string enote = ex.getNote();

#ifdef _BOCCA_STDERR
    std::cerr << "RuntimeException in user go code: " << enote << std::endl;
    std::cerr << "Returning -2 from go()" << std::endl;;
#endif

  }
  catch (sidl::SIDLException ex) {
    bocca_status = -2;
    std::string enote = ex.getNote();

#ifdef _BOCCA_STDERR
    std::cerr << "SIDLException in user go code: " << enote << std::endl;
    std::cerr << "Returning -2 from go()" << std::endl;;
#endif

  }
  catch (sidl::BaseException ex) {
    bocca_status = -2;
    std::string enote = ex.getNote();

#ifdef _BOCCA_STDERR
    std::cerr << "BaseException in user go code: " << enote << std::endl;
    std::cerr << "Returning -2 from go()" << std::endl;;
#endif

  }
  catch (std::exception ex) {
    bocca_status = -2;

#ifdef _BOCCA_STDERR
    std::cerr << "C++ exception in user go code: " << ex.what() << std::endl;
    std::cerr << "Returning -2 from go()"  << std::endl;
#endif

  }
  catch (...) {
    bocca_status = -2;

#ifdef _BOCCA_STDERR
    std::cerr << "Odd exception in user go code " << std::endl;
    std::cerr << "Returning -2 from go()" << std::endl;
#endif

  }


  BOCCAEXIT:; // target point for error and regular cleanup. do not delete.
// Bocca generated code. bocca.protected.begin(gob.cca.common.MPITest.go:boccaGoEpilog)

  // release commsetup 
  if (commsetup_fetched) {
    commsetup_fetched = false;
    try{
      this->d_services.releasePort("commsetup");
    } catch ( ::gov::cca::CCAException ex )  {

#ifdef _BOCCA_STDERR
      std::cerr << "gob.cca.common.MPITest: Error calling releasePort("
                << "\"commsetup\") at " 
                << __FILE__ << ":" << __LINE__ -4 << ": " << ex.getNote() 
                << std::endl;
#endif // _BOCCA_STDERR

    }
    // c++ port reference will be dropped when function exits, but we 
    // must tell framework.
  }

  // release commsource 
  if (commsource_fetched) {
    commsource_fetched = false;
    try{
      this->d_services.releasePort("commsource");
    } catch ( ::gov::cca::CCAException ex )  {

#ifdef _BOCCA_STDERR
      std::cerr << "gob.cca.common.MPITest: Error calling releasePort("
                << "\"commsource\") at " 
                << __FILE__ << ":" << __LINE__ -4 << ": " << ex.getNote() 
                << std::endl;
#endif // _BOCCA_STDERR

    }
    // c++ port reference will be dropped when function exits, but we 
    // must tell framework.
  }


  return bocca_status;
// Bocca generated code. bocca.protected.end(gob.cca.common.MPITest.go:boccaGoEpilog)

  // DO-NOT-DELETE splicer.end(gob.cca.common.MPITest.go)
}


// DO-NOT-DELETE splicer.begin(gob.cca.common.MPITest._misc)
// Insert-Code-Here {gob.cca.common.MPITest._misc} (miscellaneous code)
// DO-NOT-DELETE splicer.end(gob.cca.common.MPITest._misc)

