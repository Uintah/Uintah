// 
// File:          scijump_ApplicationLoaderService_Impl.hxx
// Symbol:        scijump.ApplicationLoaderService-v0.2.1
// Symbol Type:   class
// Babel Version: 1.2.0
// Description:   Server-side implementation for scijump.ApplicationLoaderService
// 
// WARNING: Automatically generated; only changes within splicers preserved
// 
// 

#ifndef included_scijump_ApplicationLoaderService_Impl_hxx
#define included_scijump_ApplicationLoaderService_Impl_hxx

#ifndef included_sidl_cxx_hxx
#include "sidl_cxx.hxx"
#endif
#ifndef included_scijump_ApplicationLoaderService_IOR_h
#include "scijump_ApplicationLoaderService_IOR.h"
#endif
#ifndef included_gov_cca_ComponentID_hxx
#include "gov_cca_ComponentID.hxx"
#endif
#ifndef included_gov_cca_ConnectionID_hxx
#include "gov_cca_ConnectionID.hxx"
#endif
#ifndef included_sci_cca_AbstractFramework_hxx
#include "sci_cca_AbstractFramework.hxx"
#endif
#ifndef included_sci_cca_core_FrameworkService_hxx
#include "sci_cca_core_FrameworkService.hxx"
#endif
#ifndef included_sci_cca_ports_ApplicationLoaderService_hxx
#include "sci_cca_ports_ApplicationLoaderService.hxx"
#endif
#ifndef included_scijump_ApplicationLoaderService_hxx
#include "scijump_ApplicationLoaderService.hxx"
#endif
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif


// DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService._hincludes)
#include "sci_cca_ports.hxx"
#include "scijump.hxx"

#include <libxml/xmlreader.h>
#include <string>
#include <stack>

#if ! defined(LIBXML_WRITER_ENABLED) && ! defined(LIBXML_OUTPUT_ENABLED)
  #error "Writer or output support not compiled in"
#endif
// DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService._hincludes)

namespace scijump { 

  /**
   * Symbol "scijump.ApplicationLoaderService" (version 0.2.1)
   */
  class ApplicationLoaderService_impl : public virtual 
    ::scijump::ApplicationLoaderService 
  // DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService._inherits)
  // Insert-Code-Here {scijump.ApplicationLoaderService._inherits} (optional inheritance here)
  // DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService._inherits)

  {

  // All data marked protected will be accessable by 
  // descendant Impl classes
  protected:

    bool _wrapped;

    // DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService._implementation)
    void writeComponentNode(gov::cca::ComponentID cid,
			    gov::cca::TypeMap& properties,
			    xmlNode** rootNode);
    void writeConnectionNode(gov::cca::ConnectionID cid, 
			     xmlNode** rootNode);
    ::gov::cca::ComponentID readComponentNode(sci::cca::ports::BuilderService& bs, 
					      xmlNode** node);
    ::gov::cca::ConnectionID readConnectionNode(sci::cca::ports::BuilderService& bs, 
						xmlNode** node);
			      
    std::string fileName;
    scijump::SCIJumpFramework framework;

    std::stack<xmlNodePtr> nodeStack;
    xmlDocPtr xmlDoc;
    // DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService._implementation)

  public:
    // default constructor, used for data wrapping(required)
    ApplicationLoaderService_impl();
    // sidl constructor (required)
    // Note: alternate Skel constructor doesn't call addref()
    // (fixes bug #275)
      ApplicationLoaderService_impl( struct 
        scijump_ApplicationLoaderService__object * ior ) : StubBase(ior,true), 
      ::sci::cca::core::FrameworkService((ior==NULL) ? NULL : &((
        *ior).d_sci_cca_core_frameworkservice)),
      ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
    ::sci::cca::ports::ApplicationLoaderService((ior==NULL) ? NULL : &((
      *ior).d_sci_cca_ports_applicationloaderservice)) , _wrapped(false) {_ctor(
      );}


    // user defined construction
    void _ctor();

    // virtual destructor (required)
    virtual ~ApplicationLoaderService_impl() { _dtor(); }

    // user defined destruction
    void _dtor();

    // true if this object was created by a user newing the impl
    inline bool _isWrapped() {return _wrapped;}

    // static class initializer
    static void _load();

  public:
    /**
     * user defined static method
     */
    static ::sci::cca::core::FrameworkService
    create_impl (
      /* in */::sci::cca::AbstractFramework& framework
    )
    ;


    /**
     * user defined non-static method.
     */
    void
    initialize_impl (
      /* in */::sci::cca::AbstractFramework& framework
    )
    ;

    /**
     * user defined non-static method.
     */
    ::std::string
    getFileName_impl() ;
    /**
     * user defined non-static method.
     */
    void
    setFileName_impl (
      /* in */const ::std::string& filename
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    loadFile_impl (
      /* out array<gov.cca.ComponentID> */::sidl::array< 
        ::gov::cca::ComponentID>& cidList,
      /* out array<gov.cca.ConnectionID> */::sidl::array< 
        ::gov::cca::ConnectionID>& connList
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    loadFile_impl (
      /* in */const ::std::string& filename,
      /* out array<gov.cca.ComponentID> */::sidl::array< 
        ::gov::cca::ComponentID>& cidList,
      /* out array<gov.cca.ConnectionID> */::sidl::array< 
        ::gov::cca::ConnectionID>& connList
    )
    ;

    /**
     * user defined non-static method.
     */
    void
    saveFile_impl() ;
    /**
     * user defined non-static method.
     */
    void
    saveFile_impl (
      /* in */const ::std::string& filename
    )
    ;

  };  // end class ApplicationLoaderService_impl

} // end namespace scijump

// DO-NOT-DELETE splicer.begin(scijump.ApplicationLoaderService._hmisc)
// Insert-Code-Here {scijump.ApplicationLoaderService._hmisc} (miscellaneous things)
// DO-NOT-DELETE splicer.end(scijump.ApplicationLoaderService._hmisc)

#endif
