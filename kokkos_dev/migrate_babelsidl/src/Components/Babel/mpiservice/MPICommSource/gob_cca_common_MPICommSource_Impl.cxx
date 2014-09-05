// 
// File:          gob_cca_common_MPICommSource_Impl.cxx
// Symbol:        gob.cca.common.MPICommSource-v0.0
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Server-side implementation for gob.cca.common.MPICommSource
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 
#include "gob_cca_common_MPICommSource_Impl.hxx"

// 
// Includes for all method dependencies.
// 
#ifndef included_gov_cca_AbstractFramework_hxx
#include "gov_cca_AbstractFramework.hxx"
#endif
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
// DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource._includes)

#include <iostream>
#include <mpi.h>
#include <gob_cca_ports.hxx>
#include <scijump_CCAException.hxx>

#define _BOCCA_STDERR 1

namespace {

void whine(std::string s)
{
        scijump::CCAException ex = scijump::CCAException::_create();
        ex.setNote(s);
	// fixme-feature traceback
        throw ex;
}
#define WHINE(s) whine(s)


/** a critter meeting the requirements of STL */
class CommHolder {
private:
        MPI_Comm com;
public:
        CommHolder(MPI_Comm c) : com(c) {}
        ~CommHolder(){}
        MPI_Comm getComm() { return com; }
        void destroy() {
                MPI_Comm_free(&com);
        }
}; // end class commholder

} // end anon namespace

namespace priv_gob_cca_common {

class MPIService_Impl
{

private:
	// darn things can be mighty heavy. we recycle.
        /** coms for recycle */
        std::vector< CommHolder *> rlist;
        /** coms in use */
        std::vector< CommHolder *> ulist;
        MPI_Comm prototype;

public:

        MPIService_Impl(MPI_Comm comm){
		prototype = comm;
	}
        virtual ~MPIService_Impl();
        virtual MPI_Comm getComm();
        virtual void releaseComm(MPI_Comm m);
        void reclaim();
	void sizeRank(int64_t & csize, int64_t & crank) {
		int merror1 = MPI_SUCCESS;
		int mrank;
		int msize;
		merror1 = MPI_Comm_rank(prototype, &mrank);
		if (merror1 != MPI_SUCCESS) {
			WHINE(std::string("error detected getting MPI_Comm_rank result"));
		}
		merror1 = MPI_Comm_size(prototype, &msize);
		if (merror1 != MPI_SUCCESS) {
			WHINE(std::string("error detected getting MPI_Comm_size result"));
		}
		csize = msize;
		crank = mrank;
	}

}; // end class MPIService_Impl


MPIService_Impl::~MPIService_Impl() {
        size_t i, cap;
        cap = rlist.size();
        for (i = 0; i < cap; i++) {
                CommHolder *c = rlist[i];
                c->destroy();
                delete c;
                rlist[i] = 0;
        }
        rlist.clear();
        cap = ulist.size();
        for (i = 0; i < cap; i++) {
                // this shouldn't occur if user is correct.
                CommHolder *c2 = ulist[i];
                delete c2;
                ulist[i] = 0;
        }
        ulist.clear();
}

void
MPIService_Impl::reclaim() {
        size_t i, cap;
        cap = rlist.size();
        for (i = 0; i < cap; i++) {
                CommHolder *c = rlist[i];
                c->destroy();
        }
        cap = ulist.size();
        for (i = 0; i < cap; i++) {
                CommHolder *c2 = ulist[i];
                c2->destroy();
        }
        ulist.clear();
}

MPI_Comm
MPIService_Impl::getComm() {
        // check for one recyclable
        if (rlist.size() > 0) {
                CommHolder *c = rlist[(rlist.size()-1)];
                rlist.erase(rlist.end()-1);
                ulist.push_back(c);
                MPI_Comm result = c->getComm();
                return result;
        }
        // make a new duplicate. save it and return it.
        MPI_Comm tmp;
	int mi = 0;
      MPI_Initialized(&mi);
      if (! mi) {
	std::cerr << "I am not initialized\n";
      }
      else {
	std::cerr << "I am initialized\n";
      }
        mi = MPI_Comm_dup(prototype, &tmp);
        CommHolder *c2 = new CommHolder(tmp);
        ulist.push_back(c2);
        return tmp;
}


void
MPIService_Impl::releaseComm(MPI_Comm m) {

        if (m == MPI_COMM_NULL) {
                return;
        }
        // if we don't find it, ignore it quietly.
        size_t i, cap;
        cap = ulist.size();
        for (i = 0; i < cap; i++) {
                int result;
                MPI_Comm tmp = ulist[i]->getComm();
                MPI_Comm_compare(m,tmp,&result);
                if (result == MPI_IDENT) {
                        CommHolder *c = ulist[i];
                        rlist.push_back(c);
                        ulist.erase(ulist.begin()+i);
                        return;
                }
        }
}

} // end namespace priv_gob_cca_common


#include <iostream>

  // Bocca generated code. bocca.protected.begin(gob.cca.common.MPICommSource._includes)

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
  // If boost is not available (and therefore gob::cca::common is not in use), 
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


  // Bocca generated code. bocca.protected.end(gob.cca.common.MPICommSource._includes)

  // Insert-UserCode-Here {gob.cca.common.MPICommSource._includes:epilog} (additional includes or code)

// DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource._includes)

// special constructor, used for data wrapping(required).  Do not put code here unless you really know what you're doing!
gob::cca::common::MPICommSource_impl::MPICommSource_impl() : StubBase(
  reinterpret_cast< void*>(::gob::cca::common::MPICommSource::_wrapObj(
  reinterpret_cast< void*>(this))),false) , _wrapped(true){ 
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource._ctor2)
  // Insert-Code-Here {gob.cca.common.MPICommSource._ctor2} (ctor2)
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource._ctor2)
}

// user defined constructor
void gob::cca::common::MPICommSource_impl::_ctor() {
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource._ctor)
    
  // Insert-UserCode-Here {gob.cca.common.MPICommSource._ctor:prolog} (constructor method) 

  // bocca-default-code. User may edit or delete.begin(gob.cca.common.MPICommSource._ctor)
   #if _BOCCA_CTOR_MESSAGES

     std::cerr << "CTOR gob.cca.common.MPICommSource: " << BOOST_CURRENT_FUNCTION 
               << " constructing " << this << std::endl;

   #endif // _BOCCA_CTOR_MESSAGES
  // bocca-default-code. User may edit or delete.end(gob.cca.common.MPICommSource._ctor)

        m1 = 0;
	initialized = false;
	finalized = false;
	isExternal = true;

  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource._ctor)
}

// user defined destructor
void gob::cca::common::MPICommSource_impl::_dtor() {
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource._dtor)
  // Insert-UserCode-Here {gob.cca.common.MPICommSource._dtor} (destructor method) 
    
  // bocca-default-code. User may edit or delete.begin(gob.cca.common.MPICommSource._dtor) 
   #if _BOCCA_CTOR_MESSAGES

     std::cerr << "DTOR gob.cca.common.MPICommSource: " << BOOST_CURRENT_FUNCTION 
               << " destructing " << this << std::endl;

   #endif // _BOCCA_CTOR_MESSAGES 
  // bocca-default-code. User may edit or delete.end(gob.cca.common.MPICommSource._dtor) 
        delete m1;
        m1 = 0;


  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource._dtor)
}

// static class initializer
void gob::cca::common::MPICommSource_impl::_load() {
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource._load)
  // Insert-Code-Here {gob.cca.common.MPICommSource._load} (class initialization)
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource._load)
}

// user defined static methods: (none)

// user defined non-static methods:
/**
 * Method:  boccaSetServices[]
 */
void
gob::cca::common::MPICommSource_impl::boccaSetServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource.boccaSetServices)
  // DO-NOT-EDIT-BOCCA
  // Bocca generated code. bocca.protected.begin(gob.cca.common.MPICommSource.boccaSetServices)

  gov::cca::TypeMap typeMap;
  gov::cca::Port    port;

  this->d_services = services;

  typeMap = this->d_services.createTypeMap();

  port = ::babel_cast< gov::cca::Port>(*this);
  if (port._is_nil()) {
    BOCCA_THROW_CXX( ::sidl::SIDLException , 
                     "gob.cca.common.MPICommSource: Error casting self to gov::cca::Port");
  } 


  // Provide a gob.cca.ports.MPIService port with port name mpiservice 
  try{
    this->d_services.addProvidesPort(
                   port,              // implementing object
                   "mpiservice", // port instance name
                   "gob.cca.ports.MPIService",     // full sidl type of port
                   typeMap);          // properties for the port
  } catch ( ::gov::cca::CCAException ex )  {
    BOCCA_EXTEND_THROW_CXX(ex, 
        "gob.cca.common.MPICommSource: Error calling addProvidesPort(port,"
        "\"mpiservice\", \"gob.cca.ports.MPIService\", typeMap) ", -2);
    throw;
  }    

  // Provide a gob.cca.ports.MPISetup port with port name MPISetup 
  try{
    this->d_services.addProvidesPort(
                   port,              // implementing object
                   "MPISetup", // port instance name
                   "gob.cca.ports.MPISetup",     // full sidl type of port
                   typeMap);          // properties for the port
  } catch ( ::gov::cca::CCAException ex )  {
    BOCCA_EXTEND_THROW_CXX(ex, 
        "gob.cca.common.MPICommSource: Error calling addProvidesPort(port,"
        "\"MPISetup\", \"gob.cca.ports.MPISetup\", typeMap) ", -2);
    throw;
  }    

  // Use a gov.cca.ports.ServiceRegistry port with port name sr 
  try{
    this->d_services.registerUsesPort(
                   "sr", // port instance name
                   "gov.cca.ports.ServiceRegistry",     // full sidl type of port
                    typeMap);         // properties for the port
  } catch ( ::gov::cca::CCAException ex )  {
    BOCCA_EXTEND_THROW_CXX(ex,
       "gob.cca.common.MPICommSource: Error calling registerUsesPort(\"sr\", "
       "\"gov.cca.ports.ServiceRegistry\", typeMap) ", -2);
    throw;
  }


  gov::cca::ComponentRelease cr = 
        ::babel_cast< gov::cca::ComponentRelease>(*this);
  this->d_services.registerForRelease(cr);
  return;
  // Bocca generated code. bocca.protected.end(gob.cca.common.MPICommSource.boccaSetServices)
    
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource.boccaSetServices)
}

/**
 * Method:  boccaReleaseServices[]
 */
void
gob::cca::common::MPICommSource_impl::boccaReleaseServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource.boccaReleaseServices)
  // DO-NOT-EDIT-BOCCA
  // Bocca generated code. bocca.protected.begin(gob.cca.common.MPICommSource.boccaReleaseServices)
  this->d_services=0;


  // Un-provide gob.cca.ports.MPIService port with port name mpiservice 
  try{
    services.removeProvidesPort("mpiservice");
  } catch ( ::gov::cca::CCAException ex )  {

#ifdef _BOCCA_STDERR
    std::cerr << "gob.cca.common.MPICommSource: Error calling removeProvidesPort("
              << "\"mpiservice\") at " 
              << __FILE__ << ": " << __LINE__ -4 << ": " << ex.getNote() 
              << std::endl;
#endif // _BOCCA_STDERR

  }

  // Un-provide gob.cca.ports.MPISetup port with port name MPISetup 
  try{
    services.removeProvidesPort("MPISetup");
  } catch ( ::gov::cca::CCAException ex )  {

#ifdef _BOCCA_STDERR
    std::cerr << "gob.cca.common.MPICommSource: Error calling removeProvidesPort("
              << "\"MPISetup\") at " 
              << __FILE__ << ": " << __LINE__ -4 << ": " << ex.getNote() 
              << std::endl;
#endif // _BOCCA_STDERR

  }

  // Release gov.cca.ports.ServiceRegistry port with port name sr 
  try{
    services.unregisterUsesPort("sr");
  } catch ( ::gov::cca::CCAException ex )  {

#ifdef _BOCCA_STDERR
    std::cerr << "gob.cca.common.MPICommSource: Error calling unregisterUsesPort("
              << "\"sr\") at " 
              << __FILE__ << ":" << __LINE__ -4 << ": " << ex.getNote() 
              << std::endl;
#endif // _BOCCA_STDERR

  }

  return;
  // Bocca generated code. bocca.protected.end(gob.cca.common.MPICommSource.boccaReleaseServices)
    
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource.boccaReleaseServices)
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
gob::cca::common::MPICommSource_impl::setServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource.setServices)

  gov::cca::TypeMap typeMap;
  gov::cca::Port    port;

  this->d_services = services;

  typeMap = this->d_services.createTypeMap();

  port = ::babel_cast< gov::cca::Port>(*this);
  if (port._is_nil()) {
    BOCCA_THROW_CXX( ::sidl::SIDLException , 
                     "gob.cca.common.MPICommSource: Error casting self to gov::cca::Port");
  } 



  // Provide a gob.cca.ports.MPISetup port with port name MPISetup 
  try{
    this->d_services.addProvidesPort(
                   port,              // implementing object
                   "MPISetup", // port instance name
                   "gob.cca.ports.MPISetup",     // full sidl type of port
                   typeMap);          // properties for the port
  } catch ( ::gov::cca::CCAException ex )  {
    BOCCA_EXTEND_THROW_CXX(ex, 
        "gob.cca.common.MPICommSource: Error calling addProvidesPort(port,"
        "\"MPISetup\", \"gob.cca.ports.MPISetup\", typeMap) ", -2);
    throw;
  }    

  // Use a gov.cca.ports.ServiceRegistry port with port name sr 
  try{
    this->d_services.registerUsesPort(
                   "sr", // port instance name
                   "gov.cca.ports.ServiceRegistry",     // full sidl type of port
                    typeMap);         // properties for the port
  } catch ( ::gov::cca::CCAException ex )  {
    BOCCA_EXTEND_THROW_CXX(ex,
       "gob.cca.common.MPICommSource: Error calling registerUsesPort(\"sr\", "
       "\"gov.cca.ports.ServiceRegistry\", typeMap) ", -2);
    throw;
  }


  gov::cca::ComponentRelease cr = 
        ::babel_cast< gov::cca::ComponentRelease>(*this);
  this->d_services.registerForRelease(cr);

  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource.setServices)
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
gob::cca::common::MPICommSource_impl::releaseServices_impl (
  /* in */::gov::cca::Services& services ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource.releaseServices)
	this->d_services=0;

	// Un-provide gob.cca.ports.MPIService port with port name mpiservice 
	if (initialized) {
		try{
			services.removeProvidesPort("mpiservice");
		} catch ( ::gov::cca::CCAException ex )  {

#ifdef _BOCCA_STDERR
    std::cerr << "gob.cca.common.MPICommSource: Error calling removeProvidesPort("
              << "\"mpiservice\") at " 
              << __FILE__ << ": " << __LINE__ -4 << ": " << ex.getNote() 
              << std::endl;
#endif // _BOCCA_STDERR

		}
	}

	// Un-provide gob.cca.ports.MPISetup port with port name MPISetup 
	try{
		services.removeProvidesPort("MPISetup");
	} catch ( ::gov::cca::CCAException ex )  {

#ifdef _BOCCA_STDERR
		std::cerr << "gob.cca.common.MPICommSource: Error calling removeProvidesPort("
		      << "\"MPISetup\") at " 
		      << __FILE__ << ": " << __LINE__ -4 << ": " << ex.getNote() 
		      << std::endl;
#endif // _BOCCA_STDERR

	}

	// Release gov.cca.ports.ServiceRegistry port with port name sr 
	try{
		services.unregisterUsesPort("sr");
	} catch ( ::gov::cca::CCAException ex )  {

#ifdef _BOCCA_STDERR
		std::cerr << "gob.cca.common.MPICommSource: Error calling unregisterUsesPort("
		      << "\"sr\") at " 
		      << __FILE__ << ":" << __LINE__ -4 << ": " << ex.getNote() 
		      << std::endl;
#endif // _BOCCA_STDERR

	}
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource.releaseServices)
}

/**
 *  Get an mpi communicator with the same scope as the component instance.
 * Won't be mpi_comm_null. The communicator returned will be
 * private to the recipient, which implies an mpi_comm_dup by the provider.
 * Call must be made collectively.
 * @return The comm produced, in FORTRAN form. C callers use comm_f2c
 * method defined by their mpi implementation, usually MPI_Comm_f2c,
 * to convert result to MPI_Comm.
 * @throw CCAException if the service cannot be implemented because MPI is
 * not present.
 */
int64_t
gob::cca::common::MPICommSource_impl::getComm_impl () 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException

{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource.getComm)
        if (! initialized || finalized ) {
                WHINE("gob::cca::common::MPICommSource_impl::getComm: not setup in run mode.");
        }
	MPI_Comm ccomm = m1->getComm();
	int64_t fcomm = MPI_Comm_c2f(ccomm);
	return fcomm;
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource.getComm)
}

/**
 *  Let go the communicator. previously fetched with getComm.
 * Call must be made collectively.
 * @throw CCAException if an error is detected.
 */
void
gob::cca::common::MPICommSource_impl::releaseComm_impl (
  /* in */int64_t comm ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource.releaseComm)
        if (! initialized || finalized) {
                WHINE("gob::cca::common::MPICommSource_impl::releaseComm: not setup in run mode.");
        }
	MPI_Comm ccomm = MPI_Comm_f2c(comm);
	m1->releaseComm(ccomm);
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource.releaseComm)
}

/**
 *  Get the typically needed basic parallelism information for a component that
 * requires no MPI communication and thus does not need an independent communicator.
 * Rationale: on very large machines, the cost of a Comm_dup should be avoided where possible;
 * The other calls on a MPI Comm object may affect its state, and thus should not
 * be proxied here.
 * @throw CCAException if an error is detected.
 */
void
gob::cca::common::MPICommSource_impl::getSizeRank_impl (
  /* out */int64_t& commSize,
  /* out */int64_t& commRank ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource.getSizeRank)
        if (! initialized) {
                WHINE("gob::cca::common::MPIComponent::finalize called before init or setServices.");
        }
	m1->sizeRank(commSize, commRank);
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource.getSizeRank)
}

/**
 *  Check instance status. Only one init call per instance is allowed.
 * @return true if initAsService or initComponent already done.
 */
bool
gob::cca::common::MPICommSource_impl::isInitialized_impl () 

{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource.isInitialized)
	return initialized;
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource.isInitialized)
}

/**
 * 
 * This method is for treating an instance from an external driver
 * to set up a general service instance global to a frame.
 * Useful in at least the static linking case.
 * 
 * Create and add to the framework MPIService
 * support. This will appear in the frame as an
 * MPICommSource component instance without necessarily existing
 * in the BuilderService accessible class list. 
 * MPI_Init must have been called before this is called.
 * This entry point should work for any cca framework bootstrapping
 * in commworld or otherwise scoped communicator via the standard
 * ServiceRegistry interface. This will not automatically
 * cause the component class providing this interface to appear in the
 * list of classes the user may instantiate.
 * In the MPI sense, this call must be collective on the scope of
 * dupComm.
 * 
 * @param dupComm  the (valid) communicator (in fortran form) to duplicate
 * for those using MPIService.
 * @param af The frame into which the server will add itself.
 * In principle, the caller should be able to forget about the class object
 * they are holding to make this call.
 */
void
gob::cca::common::MPICommSource_impl::initAsInstance_impl (
  /* in */int64_t dupComm,
  /* inout */::gov::cca::AbstractFramework& af ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource.initAsInstance)

        if (initialized) {
                WHINE("gob::cca::common::MPICommSource::initAsInstance called redundantly.");
        }
        if (af._is_nil()) {
                std::cerr << "Ugh! no framework given to initAsInstance. dying..." << std::endl;
                WHINE("gob::cca::common::MPICommSource::initAsInstance  got af==0");
        }
        naf = af;
        isExternal = true;
        gov::cca::TypeMap mstm;
        d_services = naf.getServices("MPISetup","gov.cca.common.MPICommSource", mstm);
        initPrivate(dupComm,true);

  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource.initAsInstance)
}

/**
 *  Set the communicators on an uninitialized mpi support component
 * instance created like any other and register the component through
 * the ServiceRegistry.
 * 
 * In the MPI sense, this call must be collective on the scope of 
 * dupComm.
 * 
 * @param dupComm  the (valid) communicator (in fortran form) to duplicate
 * for those using MPIService.
 * @param af The frame into which the server will add itself.
 * In principle, the caller should be able to forget about the class object
 * they are holding to make this call.
 */
void
gob::cca::common::MPICommSource_impl::initAsService_impl (
  /* in */int64_t dupComm ) 
{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource.initAsService)
        if (initialized) {
                WHINE("gob::cca::common::MPICommSource::initAsService called redundantly.");
        }
	isExternal = false;
        initPrivate(dupComm, true);
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource.initAsService)
}

/**
 *  Set the communicators on an uninitialized mpi support component
 * instance created like any other.
 * This does NOT cause the component being initialized to register itself
 * as a service for all comers.
 * This method is for treating an instance from inside a frame or
 * subframe as a peer component that may serve only certain other
 * components in the frame, e.g after a comm split.
 * 
 * In the MPI sense, this call must be collective on the scope of
 * dupComm.
 * 
 * @param dupComm  the (valid) communicator (in fortran form) to duplicate
 * for those using MPIService.
 * @param af The frame into which the server will add itself.
 * In principle, the caller should be able to forget about the class object
 * they are holding to make this call.
 */
void
gob::cca::common::MPICommSource_impl::initComponent_impl (
  /* in */int64_t dupComm ) 
{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource.initComponent)
        if (initialized) {
                WHINE("gob::cca::common::MPICommSource::initComponent called redundantly.");
        }
	isExternal = false;
        initPrivate(dupComm, false);
  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource.initComponent)
}

/**
 * Shutdown the previous mpi-related services.
 * @param reclaim if reclaim true, try to release communicator
 * resources allocated in MPIService support.
 * Otherwise, lose them.
 */
void
gob::cca::common::MPICommSource_impl::finalize_impl (
  /* in */bool reclaim ) 
// throws:
//    ::gov::cca::CCAException
//    ::sidl::RuntimeException
{
  // DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource.finalize)
        if (! initialized) {
                WHINE("gob::cca::common::MPIComponent::finalize called before init or setServices.");
        }
        if (! finalized) {
                finalized = true;
		if (sr._not_nil()) {
                	sr.removeService("gob.cca.ports.MPIService");
                	d_services.releasePort("serviceRegistry");
		}
        }
        if (isExternal) {
                naf.releaseServices(d_services);
                naf = 0;
		isExternal = false;
        }
        d_services = 0;
        mpis = 0;

        if (reclaim) { // fixme
                m1->reclaim();
        }


  // DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource.finalize)
}


// DO-NOT-DELETE splicer.begin(gob.cca.common.MPICommSource._misc)

void gob::cca::common::MPICommSource_impl::initPrivate(int64_t dupComm, bool registery)
throw ( ::gov::cca::CCAException)
{
        if (initialized) {
                WHINE("gob::cca::common::MPICommSource_impl::initPrivate called redundantly.");
        }
        MPI_Fint dc = dupComm; // CAST
        mpis = *this;
        MPI_Comm dComm = MPI_Comm_f2c(dc);
        m1 = new priv_gob_cca_common::MPIService_Impl(dComm);

	// Provide a gob.cca.ports.MPIService port with port name mpiservice 
	gov::cca::TypeMap typeMap;
	gov::cca::Port    port;
	typeMap = this->d_services.createTypeMap();
	port = ::babel_cast< gov::cca::Port>(*this);
	try{
		this->d_services.addProvidesPort(
		   port,              // implementing object
		   "mpiservice", // port instance name
		   "gob.cca.ports.MPIService",     // full sidl type of port
		   typeMap);          // properties for the port
	} catch ( ::gov::cca::CCAException ex )  {
		BOCCA_EXTEND_THROW_CXX(ex, 
		"gob.cca.common.MPICommSource: Error calling addProvidesPort(port,"
		"\"mpiservice\", \"gob.cca.ports.MPIService\", typeMap) ", -2);
		throw;
	}    

	if (registery) {
		gov::cca::TypeMap mstm;
		d_services.registerUsesPort("serviceRegistry", "cca.ServiceRegistry", mstm);
		gov::cca::Port p = d_services.getPort("serviceRegistry");
		sr = ::babel_cast< gov::cca::ports::ServiceRegistry >(p); // CAST
		if (sr._is_nil())
		{
			WHINE("no ServiceRegistry available.");
		}

		if (!sr.addSingletonService("gob.cca.ports.MPIService", mpis))
		{
			WHINE("ServiceRegistry rejected gob.cca.ports.MPIService." );
		}
	}
        initialized = true;
}

// DO-NOT-DELETE splicer.end(gob.cca.common.MPICommSource._misc)

