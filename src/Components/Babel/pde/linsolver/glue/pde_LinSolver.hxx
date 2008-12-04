// 
// File:          pde_LinSolver.hxx
// Symbol:        pde.LinSolver-v0.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Client-side glue code for pde.LinSolver
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_pde_LinSolver_hxx
#define included_pde_LinSolver_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace pde { 

  class LinSolver;
} // end namespace pde

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::pde::LinSolver >;
}
// 
// Forward declarations for method dependencies.
// 
namespace gov { 
  namespace cca { 

    class CCAException;
  } // end namespace cca
} // end namespace gov

namespace gov { 
  namespace cca { 

    class Services;
  } // end namespace cca
} // end namespace gov

namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_pde_LinSolver_IOR_h
#include "pde_LinSolver_IOR.h"
#endif
#ifndef included_gov_cca_Component_hxx
#include "gov_cca_Component.hxx"
#endif
#ifndef included_pdeports_LinSolverPort_hxx
#include "pdeports_LinSolverPort.hxx"
#endif
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
#endif
namespace sidl {
  namespace rmi {
    class Call;
    class Return;
    class Ticket;
  }
  namespace rmi {
    class InstanceHandle;
  }
}
namespace pde { 

  /**
   * Symbol "pde.LinSolver" (version 0.1)
   */
  class LinSolver: public virtual ::gov::cca::Component, public virtual 
    ::pdeports::LinSolverPort, public virtual ::sidl::BaseClass {

    //////////////////////////////////////////////////
    // 
    // Special methods for throwing exceptions
    // 

  private:
    static 
    void
    throwException1(
      const char* methodName,
      struct sidl_BaseInterface__object *_exception
    )
      // throws:
      //    ::gov::cca::CCAException
      //    ::sidl::RuntimeException
    ;
    static 
    void
    throwException0(
      const char* methodName,
      struct sidl_BaseInterface__object *_exception
    )
      // throws:
    ;

    //////////////////////////////////////////////////
    // 
    // User Defined Methods
    // 

  public:
    /**
     * user defined non-static method
     */
    int32_t
    jacobi (
      /* in array<double,2> */const ::sidl::array<double>& A,
      /* in array<double> */const ::sidl::array<double>& b,
      /* out array<double> */::sidl::array<double>& x
    )
    ;



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
    setServices (
      /* in */const ::gov::cca::Services& services
    )
    // throws:
    //    ::gov::cca::CCAException
    //    ::sidl::RuntimeException
    ;



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct pde_LinSolver__object ior_t;
    typedef struct pde_LinSolver__external ext_t;
    typedef struct pde_LinSolver__sepv sepv_t;

    // default constructor
    LinSolver() { }
    // static constructor
    static ::pde::LinSolver _create();


#ifdef WITH_RMI

    // RMI constructor
    static ::pde::LinSolver _create( /*in*/ const std::string& url );

    // RMI connect
    static inline ::pde::LinSolver _connect( /*in*/ const std::string& url ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::pde::LinSolver _connect( /*in*/ const std::string& url, /*in*/ 
      const bool ar  );


#endif /*WITH_RMI*/

    // default destructor
    virtual ~LinSolver () { }

    // copy constructor
    LinSolver ( const LinSolver& original );

    // assignment operator
    LinSolver& operator= ( const LinSolver& rhs );


    protected:
    // Internal data wrapping method
    static ior_t*  _wrapObj(void* private_data);


    public:
    // conversion from ior to C++ class
    LinSolver ( LinSolver::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    LinSolver ( LinSolver::ior_t* ior, bool isWeak );

    inline ior_t* _get_ior() const throw() {
      return reinterpret_cast< ior_t*>(d_self);
    }

    inline void _set_ior( ior_t* ptr ) throw () { 
      if(d_self == ptr) {return;}
      d_self = reinterpret_cast< void*>(ptr);

      if( ptr != NULL ) {
        gov_cca_Port_IORCache = &((*ptr).d_gov_cca_port);
        gov_cca_Component_IORCache = &((*ptr).d_gov_cca_component);
        pdeports_LinSolverPort_IORCache = &((*ptr).d_pdeports_linsolverport);
      } else {
        gov_cca_Port_IORCache = NULL;
        gov_cca_Component_IORCache = NULL;
        pdeports_LinSolverPort_IORCache = NULL;
      }
    }

    virtual int _set_ior_typesafe( struct sidl_BaseInterface__object *obj,
                                   const ::std::type_info &argtype );

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return "pde.LinSolver";}

    static struct pde_LinSolver__object* _cast(const void* src);

    // execute member function by name
    void _exec(const std::string& methodName,
               ::sidl::rmi::Call& inArgs,
               ::sidl::rmi::Return& outArgs);

    /**
     * Get the URL of the Implementation of this object (for RMI)
     */
    ::std::string
    _getURL() // throws:
    //    ::sidl::RuntimeException
    ;


    /**
     * Method to enable/disable method hooks invocation.
     */
    void
    _set_hooks (
      /* in */bool enable
    )
    // throws:
    //    ::sidl::RuntimeException
    ;


    /**
     * Method to enable/disable interface contract enforcement.
     */
    void
    _set_contracts (
      /* in */bool enable,
      /* in */const ::std::string& enfFilename,
      /* in */bool resetCounters
    )
    // throws:
    //    ::sidl::RuntimeException
    ;


    /**
     * Method to dump contract enforcement statistics.
     */
    void
    _dump_stats (
      /* in */const ::std::string& filename,
      /* in */const ::std::string& prefix
    )
    // throws:
    //    ::sidl::RuntimeException
    ;

    // return true iff object is remote
    bool _isRemote() const { 
      ior_t* self = const_cast<ior_t*>(_get_ior() );
      struct sidl_BaseInterface__object *throwaway_exception;
      return (*self->d_epv->f__isRemote)(self, &throwaway_exception) == TRUE;
    }

    // return true iff object is local
    bool _isLocal() const {
      return !_isRemote();
    }

  protected:
    // Pointer to external (DLL loadable) symbols (shared among instances)
    static const ext_t * s_ext;

  public:
    static const ext_t * _get_ext() throw ( ::sidl::NullIORException );

  }; // end class LinSolver
} // end namespace pde

extern "C" {


#pragma weak pde_LinSolver__connectI

  /**
   * RMI connector function for the class. (no addref)
   */
  struct pde_LinSolver__object*
  pde_LinSolver__connectI(const char * url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::pde::LinSolver > {
    typedef array< ::pde::LinSolver > cxx_array_t;
    typedef ::pde::LinSolver cxx_item_t;
    typedef struct pde_LinSolver__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct pde_LinSolver__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::pde::LinSolver > > iterator;
    typedef const_array_iter< array_traits< ::pde::LinSolver > > const_iterator;
  };

  // array specialization
  template<>
  class array< ::pde::LinSolver >: public interface_array< array_traits< 
    ::pde::LinSolver > > {
  public:
    typedef interface_array< array_traits< ::pde::LinSolver > > Base;
    typedef array_traits< ::pde::LinSolver >::cxx_array_t          cxx_array_t;
    typedef array_traits< ::pde::LinSolver >::cxx_item_t           cxx_item_t;
    typedef array_traits< ::pde::LinSolver >::ior_array_t          ior_array_t;
    typedef array_traits< ::pde::LinSolver >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::pde::LinSolver >::ior_item_t           ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct pde_LinSolver__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::pde::LinSolver >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::pde::LinSolver >&
    operator =( const array< ::pde::LinSolver >&rhs ) { 
      if (d_array != rhs._get_baseior()) {
        if (d_array) deleteRef();
        d_array = const_cast<sidl__array *>(rhs._get_baseior());
        if (d_array) addRef();
      }
      return *this;
    }

  };
}

#ifndef included_gov_cca_Services_hxx
#include "gov_cca_Services.hxx"
#endif
#endif
