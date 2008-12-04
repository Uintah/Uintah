// 
// File:          pdeports_ViewPort.hxx
// Symbol:        pdeports.ViewPort-v0.1
// Symbol Type:   interface
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Client-side glue code for pdeports.ViewPort
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_pdeports_ViewPort_hxx
#define included_pdeports_ViewPort_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
// declare class before main #includes
// (this alleviates circular #include guard problems)[BUG#393]
namespace pdeports { 

  class ViewPort;
} // end namespace pdeports

// Some compilers need to define array template before the specializations
namespace sidl {
  template<>
  class array< ::pdeports::ViewPort >;
}
// 
// Forward declarations for method dependencies.
// 
namespace sidl { 

  class RuntimeException;
} // end namespace sidl

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_pdeports_ViewPort_IOR_h
#include "pdeports_ViewPort_IOR.h"
#endif
#ifndef included_gov_cca_Port_hxx
#include "gov_cca_Port.hxx"
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
namespace pdeports { 

  /**
   * Symbol "pdeports.ViewPort" (version 0.1)
   */
  class ViewPort: public virtual ::gov::cca::Port {

    //////////////////////////////////////////////////
    // 
    // Special methods for throwing exceptions
    // 

  private:
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
    view2dPDE (
      /* in array<double> */const ::sidl::array<double>& nodes,
      /* in array<int> */const ::sidl::array<int32_t>& triangles,
      /* in array<double> */const ::sidl::array<double>& solution
    )
    ;



    //////////////////////////////////////////////////
    // 
    // End User Defined Methods
    // (everything else in this file is specific to
    //  Babel's C++ bindings)
    // 

  public:
    typedef struct pdeports_ViewPort__object ior_t;
    typedef struct pdeports_ViewPort__external ext_t;
    typedef struct pdeports_ViewPort__sepv sepv_t;

    // default constructor
    ViewPort()  : pdeports_ViewPort_IORCache((ior_t*) NULL){ }

#ifdef WITH_RMI

    // RMI connect
    static inline ::pdeports::ViewPort _connect( /*in*/ const std::string& url 
      ) { 
      return _connect(url, true);
    }

    // RMI connect 2
    static ::pdeports::ViewPort _connect( /*in*/ const std::string& url, /*in*/ 
      const bool ar  );


#endif /*WITH_RMI*/

    // default destructor
    virtual ~ViewPort () { }

    // copy constructor
    ViewPort ( const ViewPort& original );

    // assignment operator
    ViewPort& operator= ( const ViewPort& rhs );

    // conversion from ior to C++ class
    ViewPort ( ViewPort::ior_t* ior );

    // Alternate constructor: does not call addRef()
    // (sets d_weak_reference=isWeak)
    // For internal use by Impls (fixes bug#275)
    ViewPort ( ViewPort::ior_t* ior, bool isWeak );

    inline ior_t* _get_ior() const throw() {
      if(!pdeports_ViewPort_IORCache) { 
        pdeports_ViewPort_IORCache = ::pdeports::ViewPort::_cast((void*)d_self);
        if (pdeports_ViewPort_IORCache) {
          struct sidl_BaseInterface__object *throwaway_exception;
          (pdeports_ViewPort_IORCache->d_epv->f_deleteRef)(
            pdeports_ViewPort_IORCache->d_object, &throwaway_exception);  
        }  
      }
      return pdeports_ViewPort_IORCache;
    }

    inline void _set_ior( ior_t* ptr ) throw () { 
      if(d_self == ptr) {return;}
      d_self = reinterpret_cast< void*>(ptr);
      pdeports_ViewPort_IORCache = (ior_t*) ptr;

      gov_cca_Port_IORCache = NULL;
    }

    virtual int _set_ior_typesafe( struct sidl_BaseInterface__object *obj,
                                   const ::std::type_info &argtype );

    bool _is_nil() const throw () { return (d_self==0); }

    bool _not_nil() const throw () { return (d_self!=0); }

    bool operator !() const throw () { return (d_self==0); }

    static inline const char * type_name() throw () { return 
      "pdeports.ViewPort";}

    static struct pdeports_ViewPort__object* _cast(const void* src);

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


    //////////////////////////////////////////////////
    // 
    // Locally Cached IOR pointer
    // 

  protected:
    mutable ior_t* pdeports_ViewPort_IORCache;
  }; // end class ViewPort
} // end namespace pdeports

extern "C" {


#pragma weak pdeports_ViewPort__connectI

  /**
   * RMI connector function for the class. (no addref)
   */
  struct pdeports_ViewPort__object*
  pdeports_ViewPort__connectI(const char * url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex);


} // end extern "C"
namespace sidl {
  // traits specialization
  template<>
  struct array_traits< ::pdeports::ViewPort > {
    typedef array< ::pdeports::ViewPort > cxx_array_t;
    typedef ::pdeports::ViewPort cxx_item_t;
    typedef struct pdeports_ViewPort__array ior_array_t;
    typedef sidl_interface__array ior_array_internal_t;
    typedef struct pdeports_ViewPort__object ior_item_t;
    typedef cxx_item_t value_type;
    typedef value_type reference;
    typedef value_type* pointer;
    typedef const value_type const_reference;
    typedef const value_type* const_pointer;
    typedef array_iter< array_traits< ::pdeports::ViewPort > > iterator;
    typedef const_array_iter< array_traits< ::pdeports::ViewPort > > 
      const_iterator;
  };

  // array specialization
  template<>
  class array< ::pdeports::ViewPort >: public interface_array< array_traits< 
    ::pdeports::ViewPort > > {
  public:
    typedef interface_array< array_traits< ::pdeports::ViewPort > > Base;
    typedef array_traits< ::pdeports::ViewPort >::cxx_array_t          
      cxx_array_t;
    typedef array_traits< ::pdeports::ViewPort >::cxx_item_t           
      cxx_item_t;
    typedef array_traits< ::pdeports::ViewPort >::ior_array_t          
      ior_array_t;
    typedef array_traits< ::pdeports::ViewPort >::ior_array_internal_t 
      ior_array_internal_t;
    typedef array_traits< ::pdeports::ViewPort >::ior_item_t           
      ior_item_t;

    /**
     * conversion from ior to C++ class
     * (constructor/casting operator)
     */
    array( struct pdeports_ViewPort__array* src = 0) : Base(src) {}

    /**
     * copy constructor
     */
    array( const array< ::pdeports::ViewPort >&src) : Base(src) {}

    /**
     * assignment
     */
    array< ::pdeports::ViewPort >&
    operator =( const array< ::pdeports::ViewPort >&rhs ) { 
      if (d_array != rhs._get_baseior()) {
        if (d_array) deleteRef();
        d_array = const_cast<sidl__array *>(rhs._get_baseior());
        if (d_array) addRef();
      }
      return *this;
    }

  };
}

#endif
