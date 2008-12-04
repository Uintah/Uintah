// 
// File:          pde_PDEdriver.cxx
// Symbol:        pde.PDEdriver-v0.1
// Symbol Type:   class
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Client-side glue code for pde.PDEdriver
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_pde_PDEdriver_hxx
#include "pde_PDEdriver.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
#endif
#ifndef included_sidl_BaseException_hxx
#include "sidl_BaseException.hxx"
#endif
#ifndef included_sidl_LangSpecificException_hxx
#include "sidl_LangSpecificException.hxx"
#endif
#ifndef included_sidl_RuntimeException_hxx
#include "sidl_RuntimeException.hxx"
#endif
#ifndef included_sidl_CastException_hxx
#include "sidl_CastException.hxx"
#endif
#ifndef included_sidl_rmi_Call_hxx
#include "sidl_rmi_Call.hxx"
#endif
#ifndef included_sidl_rmi_Return_hxx
#include "sidl_rmi_Return.hxx"
#endif
#ifndef included_sidl_rmi_Ticket_hxx
#include "sidl_rmi_Ticket.hxx"
#endif
#ifndef included_sidl_rmi_InstanceHandle_hxx
#include "sidl_rmi_InstanceHandle.hxx"
#endif
#include "sidl_String.h"
#include "sidl_rmi_ConnectRegistry.h"
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "sidl_Loader.hxx"
#include "sidl_DLL.hxx"
#endif
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

#define LANG_SPECIFIC_INIT()
extern "C" {
#ifdef WITH_RMI

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_rmi_ProtocolFactory_h
#include "sidl_rmi_ProtocolFactory.h"
#endif
#ifndef included_sidl_rmi_InstanceRegistry_h
#include "sidl_rmi_InstanceRegistry.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_Invocation_h
#include "sidl_rmi_Invocation.h"
#endif
#ifndef included_sidl_rmi_Response_h
#include "sidl_rmi_Response.h"
#endif
#ifndef included_sidl_rmi_ServerRegistry_h
#include "sidl_rmi_ServerRegistry.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#ifndef included_sidl_MemAllocException_h
#include "sidl_MemAllocException.h"
#endif
#ifndef included_sidl_NotImplementedException_h
#include "sidl_NotImplementedException.h"
#endif
#include "sidl_Exception.h"

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t pde_PDEdriver__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &pde_PDEdriver__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &pde_PDEdriver__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &pde_PDEdriver__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

  // Static variables to hold version of IOR
  static const int32_t s_IOR_MAJOR_VERSION = 2;
  static const int32_t s_IOR_MINOR_VERSION = 0;

  // Static variables for managing EPV initialization.
  static int s_remote_initialized = 0;

  static struct pde_PDEdriver__epv s_rem_epv__pde_pdedriver;

  static struct gov_cca_Component__epv s_rem_epv__gov_cca_component;

  static struct gov_cca_Port__epv s_rem_epv__gov_cca_port;

  static struct gov_cca_ports_GoPort__epv s_rem_epv__gov_cca_ports_goport;

  static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

  static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;


  // REMOTE CAST: dynamic type casting for remote objects.
  static void* remote_pde_PDEdriver__cast(
    struct pde_PDEdriver__object* self,
    const char* name, sidl_BaseInterface* _ex)
  {
    int cmp;
    void* cast = NULL;
    *_ex = NULL; /* default to no exception */
    cmp = strcmp(name, "pde.PDEdriver");
    if (!cmp) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = ((struct pde_PDEdriver__object*)self);
      return cast;
    }
    else if (cmp < 0) {
      cmp = strcmp(name, "gov.cca.Port");
      if (!cmp) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_gov_cca_port);
        return cast;
      }
      else if (cmp < 0) {
        cmp = strcmp(name, "gov.cca.Component");
        if (!cmp) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = &((*self).d_gov_cca_component);
          return cast;
        }
      }
      else if (cmp > 0) {
        cmp = strcmp(name, "gov.cca.ports.GoPort");
        if (!cmp) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = &((*self).d_gov_cca_ports_goport);
          return cast;
        }
      }
    }
    else if (cmp > 0) {
      cmp = strcmp(name, "sidl.BaseInterface");
      if (!cmp) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_sidl_baseclass.d_sidl_baseinterface);
        return cast;
      }
      else if (cmp < 0) {
        cmp = strcmp(name, "sidl.BaseClass");
        if (!cmp) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = ((struct sidl_BaseClass__object*)self);
          return cast;
        }
      }
    }
    if ((*self->d_epv->f_isType)(self,name, _ex)) {
      void* (*func)(struct sidl_rmi_InstanceHandle__object*, struct 
        sidl_BaseInterface__object**) = 
        (void* (*)(struct sidl_rmi_InstanceHandle__object*, struct 
          sidl_BaseInterface__object**)) 
        sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
      cast =  (*func)(((struct pde_PDEdriver__remote*)self->d_data)->d_ih, _ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // REMOTE DELETE: call the remote destructor for the object.
  static void remote_pde_PDEdriver__delete(
    struct pde_PDEdriver__object* self,
    struct sidl_BaseInterface__object* *_ex)
  {
    *_ex = NULL;
    free((void*) self);
  }

  // REMOTE GETURL: call the getURL function for the object.
  static char* remote_pde_PDEdriver__getURL(
    struct pde_PDEdriver__object* self, struct sidl_BaseInterface__object* *_ex)
  {
    struct sidl_rmi_InstanceHandle__object *conn = ((struct 
      pde_PDEdriver__remote*)self->d_data)->d_ih;
    *_ex = NULL;
    if(conn != NULL) {
      return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
    }
    return NULL;
  }

  // REMOTE ADDREF: For internal babel use only! Remote addRef.
  static void remote_pde_PDEdriver__raddRef(
    struct pde_PDEdriver__object* self,struct sidl_BaseInterface__object* *_ex)
  {
    struct sidl_BaseException__object* netex = NULL;
    // initialize a new invocation
    struct sidl_BaseInterface__object* _throwaway = NULL;
    struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
      pde_PDEdriver__remote*)self->d_data)->d_ih;
    sidl_rmi_Response _rsvp = NULL;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "addRef", _ex ); SIDL_CHECK(*_ex);
    // send actual RMI request
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex);SIDL_CHECK(*_ex);
    // Check for exceptions
    netex = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);
    if(netex != NULL) {
      *_ex = (struct sidl_BaseInterface__object*)netex;
      return;
    }

    // cleanup and return
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv,&_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp,&_throwaway); }
    return;
  }

  // REMOTE ISREMOTE: returns true if this object is Remote (it is).
  static sidl_bool
  remote_pde_PDEdriver__isRemote(
      struct pde_PDEdriver__object* self, 
      struct sidl_BaseInterface__object* *_ex) {
    *_ex = NULL;
    return TRUE;
  }

  // REMOTE METHOD STUB:_set_hooks
  static void
  remote_pde_PDEdriver__set_hooks(
    /* in */ struct pde_PDEdriver__object*self ,
    /* in */ sidl_bool enable,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pde_PDEdriver__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "_set_hooks", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packBool( _inv, "enable", enable, _ex);SIDL_CHECK(
        *_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pde.PDEdriver._set_hooks.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // Contract enforcement has not been implemented for remote use.
  // REMOTE METHOD STUB:_set_contracts
  static void
  remote_pde_PDEdriver__set_contracts(
    /* in */ struct pde_PDEdriver__object*self ,
    /* in */ sidl_bool enable,
    /* in */ const char* enfFilename,
    /* in */ sidl_bool resetCounters,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pde_PDEdriver__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "_set_contracts", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packBool( _inv, "enable", enable, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packString( _inv, "enfFilename", enfFilename, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "resetCounters", resetCounters, 
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pde.PDEdriver._set_contracts.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // Contract enforcement has not been implemented for remote use.
  // REMOTE METHOD STUB:_dump_stats
  static void
  remote_pde_PDEdriver__dump_stats(
    /* in */ struct pde_PDEdriver__object*self ,
    /* in */ const char* filename,
    /* in */ const char* prefix,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pde_PDEdriver__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "_dump_stats", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "filename", filename, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "prefix", prefix, _ex);SIDL_CHECK(
        *_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pde.PDEdriver._dump_stats.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE EXEC: call the exec function for the object.
  static void remote_pde_PDEdriver__exec(
    struct pde_PDEdriver__object* self,const char* methodName,
    sidl_rmi_Call inArgs,
    sidl_rmi_Return outArgs,
    struct sidl_BaseInterface__object* *_ex)
  {
    *_ex = NULL;
  }

  // REMOTE METHOD STUB:addRef
  static void
  remote_pde_PDEdriver_addRef(
    /* in */ struct pde_PDEdriver__object*self ,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct pde_PDEdriver__remote* r_obj = (struct 
        pde_PDEdriver__remote*)self->d_data;
      LOCK_STATIC_GLOBALS;
      r_obj->d_refcount++;
#ifdef SIDL_DEBUG_REFCOUNT
      fprintf(stderr, "babel: addRef %p new count %d (type %s)\n",
        r_obj, r_obj->d_refcount, 
        "pde.PDEdriver Remote Stub");
#endif /* SIDL_DEBUG_REFCOUNT */ 
      UNLOCK_STATIC_GLOBALS;
    }
  }

  // REMOTE METHOD STUB:deleteRef
  static void
  remote_pde_PDEdriver_deleteRef(
    /* in */ struct pde_PDEdriver__object*self ,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct pde_PDEdriver__remote* r_obj = (struct 
        pde_PDEdriver__remote*)self->d_data;
      LOCK_STATIC_GLOBALS;
      r_obj->d_refcount--;
#ifdef SIDL_DEBUG_REFCOUNT
      fprintf(stderr, "babel: deleteRef %p new count %d (type %s)\n",r_obj, r_obj->d_refcount, "pde.PDEdriver Remote Stub");
#endif /* SIDL_DEBUG_REFCOUNT */ 
      if(r_obj->d_refcount == 0) {
        sidl_rmi_InstanceHandle_deleteRef(r_obj->d_ih, _ex);
        free(r_obj);
        free(self);
      }
      UNLOCK_STATIC_GLOBALS;
    }
  }

  // REMOTE METHOD STUB:isSame
  static sidl_bool
  remote_pde_PDEdriver_isSame(
    /* in */ struct pde_PDEdriver__object*self ,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      sidl_bool _retval = FALSE;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pde_PDEdriver__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "isSame", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      if(iobj){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)iobj, 
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "iobj", _url, _ex);SIDL_CHECK(
          *_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "iobj", NULL, _ex);SIDL_CHECK(
          *_ex);
      }

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pde.PDEdriver.isSame.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:isType
  static sidl_bool
  remote_pde_PDEdriver_isType(
    /* in */ struct pde_PDEdriver__object*self ,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      sidl_bool _retval = FALSE;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pde_PDEdriver__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "isType", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pde.PDEdriver.isType.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:getClassInfo
  static struct sidl_ClassInfo__object*
  remote_pde_PDEdriver_getClassInfo(
    /* in */ struct pde_PDEdriver__object*self ,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      char*_retval_str = NULL;
      struct sidl_ClassInfo__object* _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pde_PDEdriver__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "getClassInfo", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pde.PDEdriver.getClassInfo.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str, 
        _ex);SIDL_CHECK(*_ex);
      _retval = sidl_ClassInfo__connectI(_retval_str, FALSE, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:setServices
  static void
  remote_pde_PDEdriver_setServices(
    /* in */ struct pde_PDEdriver__object*self ,
    /* in */ struct gov_cca_Services__object* services,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pde_PDEdriver__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "setServices", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      if(services){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)services, 
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "services", _url, _ex);SIDL_CHECK(
          *_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "services", NULL, _ex);SIDL_CHECK(
          *_ex);
      }

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pde.PDEdriver.setServices.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:go
  static int32_t
  remote_pde_PDEdriver_go(
    /* in */ struct pde_PDEdriver__object*self ,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      struct sidl_BaseInterface__object* _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      int32_t _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        pde_PDEdriver__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "go", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pde.PDEdriver.go.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE EPV: create remote entry point vectors (EPVs).
  static void pde_PDEdriver__init_remote_epv(void)
  {
    // assert( HAVE_LOCKED_STATIC_GLOBALS );
    struct pde_PDEdriver__epv*        epv = &s_rem_epv__pde_pdedriver;
    struct gov_cca_Component__epv*    e0  = &s_rem_epv__gov_cca_component;
    struct gov_cca_Port__epv*         e1  = &s_rem_epv__gov_cca_port;
    struct gov_cca_ports_GoPort__epv* e2  = &s_rem_epv__gov_cca_ports_goport;
    struct sidl_BaseClass__epv*       e3  = &s_rem_epv__sidl_baseclass;
    struct sidl_BaseInterface__epv*   e4  = &s_rem_epv__sidl_baseinterface;

    epv->f__cast               = remote_pde_PDEdriver__cast;
    epv->f__delete             = remote_pde_PDEdriver__delete;
    epv->f__exec               = remote_pde_PDEdriver__exec;
    epv->f__getURL             = remote_pde_PDEdriver__getURL;
    epv->f__raddRef            = remote_pde_PDEdriver__raddRef;
    epv->f__isRemote           = remote_pde_PDEdriver__isRemote;
    epv->f__set_hooks          = remote_pde_PDEdriver__set_hooks;
    epv->f__set_contracts      = remote_pde_PDEdriver__set_contracts;
    epv->f__dump_stats         = remote_pde_PDEdriver__dump_stats;
    epv->f__ctor               = NULL;
    epv->f__ctor2              = NULL;
    epv->f__dtor               = NULL;
    epv->f_addRef              = remote_pde_PDEdriver_addRef;
    epv->f_deleteRef           = remote_pde_PDEdriver_deleteRef;
    epv->f_isSame              = remote_pde_PDEdriver_isSame;
    epv->f_isType              = remote_pde_PDEdriver_isType;
    epv->f_getClassInfo        = remote_pde_PDEdriver_getClassInfo;
    epv->f_setServices         = remote_pde_PDEdriver_setServices;
    epv->f_go                  = remote_pde_PDEdriver_go;

    e0->f__cast          = (void* (*)(void*, const char*, struct 
      sidl_BaseInterface__object**)) epv->f__cast;
    e0->f__delete        = (void (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__delete;
    e0->f__getURL        = (char* (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__getURL;
    e0->f__raddRef       = (void (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__raddRef;
    e0->f__isRemote      = (sidl_bool (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__isRemote;
    e0->f__set_hooks     = (void (*)(void*, sidl_bool, struct 
      sidl_BaseInterface__object**)) epv->f__set_hooks;
    e0->f__set_contracts = (void (*)(void*, sidl_bool, const char*, sidl_bool, 
      struct sidl_BaseInterface__object**)) epv->f__set_contracts;
    e0->f__dump_stats    = (void (*)(void*, const char*, const char*, struct 
      sidl_BaseInterface__object**)) epv->f__dump_stats;
    e0->f__exec          = (void (*)(void*,const char*,struct 
      sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
      sidl_BaseInterface__object **)) epv->f__exec;
    e0->f_addRef         = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_addRef;
    e0->f_deleteRef      = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_deleteRef;
    e0->f_isSame         = (sidl_bool (*)(void*,struct 
      sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
      epv->f_isSame;
    e0->f_isType         = (sidl_bool (*)(void*,const char*,struct 
      sidl_BaseInterface__object **)) epv->f_isType;
    e0->f_getClassInfo   = (struct sidl_ClassInfo__object* (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_getClassInfo;
    e0->f_setServices    = (void (*)(void*,struct gov_cca_Services__object*,
      struct sidl_BaseInterface__object **)) epv->f_setServices;

    e1->f__cast          = (void* (*)(void*, const char*, struct 
      sidl_BaseInterface__object**)) epv->f__cast;
    e1->f__delete        = (void (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__delete;
    e1->f__getURL        = (char* (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__getURL;
    e1->f__raddRef       = (void (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__raddRef;
    e1->f__isRemote      = (sidl_bool (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__isRemote;
    e1->f__set_hooks     = (void (*)(void*, sidl_bool, struct 
      sidl_BaseInterface__object**)) epv->f__set_hooks;
    e1->f__set_contracts = (void (*)(void*, sidl_bool, const char*, sidl_bool, 
      struct sidl_BaseInterface__object**)) epv->f__set_contracts;
    e1->f__dump_stats    = (void (*)(void*, const char*, const char*, struct 
      sidl_BaseInterface__object**)) epv->f__dump_stats;
    e1->f__exec          = (void (*)(void*,const char*,struct 
      sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
      sidl_BaseInterface__object **)) epv->f__exec;
    e1->f_addRef         = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_addRef;
    e1->f_deleteRef      = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_deleteRef;
    e1->f_isSame         = (sidl_bool (*)(void*,struct 
      sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
      epv->f_isSame;
    e1->f_isType         = (sidl_bool (*)(void*,const char*,struct 
      sidl_BaseInterface__object **)) epv->f_isType;
    e1->f_getClassInfo   = (struct sidl_ClassInfo__object* (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_getClassInfo;

    e2->f__cast          = (void* (*)(void*, const char*, struct 
      sidl_BaseInterface__object**)) epv->f__cast;
    e2->f__delete        = (void (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__delete;
    e2->f__getURL        = (char* (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__getURL;
    e2->f__raddRef       = (void (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__raddRef;
    e2->f__isRemote      = (sidl_bool (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__isRemote;
    e2->f__set_hooks     = (void (*)(void*, sidl_bool, struct 
      sidl_BaseInterface__object**)) epv->f__set_hooks;
    e2->f__set_contracts = (void (*)(void*, sidl_bool, const char*, sidl_bool, 
      struct sidl_BaseInterface__object**)) epv->f__set_contracts;
    e2->f__dump_stats    = (void (*)(void*, const char*, const char*, struct 
      sidl_BaseInterface__object**)) epv->f__dump_stats;
    e2->f__exec          = (void (*)(void*,const char*,struct 
      sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
      sidl_BaseInterface__object **)) epv->f__exec;
    e2->f_addRef         = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_addRef;
    e2->f_deleteRef      = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_deleteRef;
    e2->f_isSame         = (sidl_bool (*)(void*,struct 
      sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
      epv->f_isSame;
    e2->f_isType         = (sidl_bool (*)(void*,const char*,struct 
      sidl_BaseInterface__object **)) epv->f_isType;
    e2->f_getClassInfo   = (struct sidl_ClassInfo__object* (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_getClassInfo;
    e2->f_go             = (int32_t (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_go;

    e3->f__cast          = (void* (*)(struct sidl_BaseClass__object*, const 
      char*, struct sidl_BaseInterface__object**)) epv->f__cast;
    e3->f__delete        = (void (*)(struct sidl_BaseClass__object*, struct 
      sidl_BaseInterface__object**)) epv->f__delete;
    e3->f__getURL        = (char* (*)(struct sidl_BaseClass__object*, struct 
      sidl_BaseInterface__object**)) epv->f__getURL;
    e3->f__raddRef       = (void (*)(struct sidl_BaseClass__object*, struct 
      sidl_BaseInterface__object**)) epv->f__raddRef;
    e3->f__isRemote      = (sidl_bool (*)(struct sidl_BaseClass__object*, 
      struct sidl_BaseInterface__object**)) epv->f__isRemote;
    e3->f__set_hooks     = (void (*)(struct sidl_BaseClass__object*, sidl_bool, 
      struct sidl_BaseInterface__object**)) epv->f__set_hooks;
    e3->f__set_contracts = (void (*)(struct sidl_BaseClass__object*, sidl_bool, 
      const char*, sidl_bool, struct sidl_BaseInterface__object**)) 
      epv->f__set_contracts;
    e3->f__dump_stats    = (void (*)(struct sidl_BaseClass__object*, const 
      char*, const char*, struct sidl_BaseInterface__object**)) 
      epv->f__dump_stats;
    e3->f__exec          = (void (*)(struct sidl_BaseClass__object*,const char*,
      struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
      sidl_BaseInterface__object **)) epv->f__exec;
    e3->f_addRef         = (void (*)(struct sidl_BaseClass__object*,struct 
      sidl_BaseInterface__object **)) epv->f_addRef;
    e3->f_deleteRef      = (void (*)(struct sidl_BaseClass__object*,struct 
      sidl_BaseInterface__object **)) epv->f_deleteRef;
    e3->f_isSame         = (sidl_bool (*)(struct sidl_BaseClass__object*,struct 
      sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
      epv->f_isSame;
    e3->f_isType         = (sidl_bool (*)(struct sidl_BaseClass__object*,const 
      char*,struct sidl_BaseInterface__object **)) epv->f_isType;
    e3->f_getClassInfo   = (struct sidl_ClassInfo__object* (*)(struct 
      sidl_BaseClass__object*,struct sidl_BaseInterface__object **)) 
      epv->f_getClassInfo;

    e4->f__cast          = (void* (*)(void*, const char*, struct 
      sidl_BaseInterface__object**)) epv->f__cast;
    e4->f__delete        = (void (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__delete;
    e4->f__getURL        = (char* (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__getURL;
    e4->f__raddRef       = (void (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__raddRef;
    e4->f__isRemote      = (sidl_bool (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__isRemote;
    e4->f__set_hooks     = (void (*)(void*, sidl_bool, struct 
      sidl_BaseInterface__object**)) epv->f__set_hooks;
    e4->f__set_contracts = (void (*)(void*, sidl_bool, const char*, sidl_bool, 
      struct sidl_BaseInterface__object**)) epv->f__set_contracts;
    e4->f__dump_stats    = (void (*)(void*, const char*, const char*, struct 
      sidl_BaseInterface__object**)) epv->f__dump_stats;
    e4->f__exec          = (void (*)(void*,const char*,struct 
      sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
      sidl_BaseInterface__object **)) epv->f__exec;
    e4->f_addRef         = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_addRef;
    e4->f_deleteRef      = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_deleteRef;
    e4->f_isSame         = (sidl_bool (*)(void*,struct 
      sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
      epv->f_isSame;
    e4->f_isType         = (sidl_bool (*)(void*,const char*,struct 
      sidl_BaseInterface__object **)) epv->f_isType;
    e4->f_getClassInfo   = (struct sidl_ClassInfo__object* (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_getClassInfo;

    s_remote_initialized = 1;
  }

  // Create an instance that connects to an existing remote object.
  static struct pde_PDEdriver__object*
  pde_PDEdriver__remoteConnect(const char *url, sidl_bool ar, struct 
    sidl_BaseInterface__object* *_ex)
  {
    struct pde_PDEdriver__object* self = NULL;

    struct pde_PDEdriver__object* s0;
    struct sidl_BaseClass__object* s1;

    struct pde_PDEdriver__remote* r_obj = NULL;
    sidl_rmi_InstanceHandle instance = NULL;
    char* objectID = NULL;
    objectID = NULL;
    *_ex = NULL;
    if(url == NULL) {return NULL;}
    objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
    if(objectID) {
      struct pde_PDEdriver__object* retobj = NULL;
      struct sidl_BaseInterface__object *throwaway_exception;
      sidl_BaseInterface bi = (
        sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(
        objectID, _ex); SIDL_CHECK(*_ex);
      (*bi->d_epv->f_deleteRef)(bi->d_object, &throwaway_exception);
      retobj = (struct pde_PDEdriver__object*) (*bi->d_epv->f__cast)(
        bi->d_object, "pde.PDEdriver", _ex);
      if(!ar) { 
        (*bi->d_epv->f_deleteRef)(bi->d_object, &throwaway_exception);
      }
      return retobj;
    }
    instance = sidl_rmi_ProtocolFactory_connectInstance(url, "pde.PDEdriver", 
      ar, _ex ); SIDL_CHECK(*_ex);
    if ( instance == NULL) { return NULL; }
    self =
      (struct pde_PDEdriver__object*) malloc(
        sizeof(struct pde_PDEdriver__object));

    r_obj =
      (struct pde_PDEdriver__remote*) malloc(
        sizeof(struct pde_PDEdriver__remote));

    if(!self || !r_obj) {
      sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(
        _ex);
      SIDL_CHECK(*_ex);
      sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(
        *_ex);
      sidl_MemAllocException_add(ex, __FILE__, __LINE__, 
        "pde.PDEdriver.EPVgeneration", _ex);
      SIDL_CHECK(*_ex);
      *_ex = (struct sidl_BaseInterface__object*)ex;
      goto EXIT;
    }

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                          self;
    s1 =                          &s0->d_sidl_baseclass;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      pde_PDEdriver__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s1->d_sidl_baseinterface.d_object = (void*) self;

    s1->d_data = (void*) r_obj;
    s1->d_epv  = &s_rem_epv__sidl_baseclass;

    s0->d_gov_cca_component.d_epv    = &s_rem_epv__gov_cca_component;
    s0->d_gov_cca_component.d_object = (void*) self;

    s0->d_gov_cca_port.d_epv    = &s_rem_epv__gov_cca_port;
    s0->d_gov_cca_port.d_object = (void*) self;

    s0->d_gov_cca_ports_goport.d_epv    = &s_rem_epv__gov_cca_ports_goport;
    s0->d_gov_cca_ports_goport.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__pde_pdedriver;

    self->d_data = (void*) r_obj;

    return self;
    EXIT:
    if(self) { free(self); }
    if(r_obj) { free(r_obj); }
    return NULL;
  }
  // Create an instance that uses an already existing 
  // InstanceHandle to connect to an existing remote object.
  static struct pde_PDEdriver__object*
  pde_PDEdriver__IHConnect(sidl_rmi_InstanceHandle instance, struct 
    sidl_BaseInterface__object* *_ex)
  {
    struct pde_PDEdriver__object* self = NULL;

    struct pde_PDEdriver__object* s0;
    struct sidl_BaseClass__object* s1;

    struct pde_PDEdriver__remote* r_obj = NULL;
    self =
      (struct pde_PDEdriver__object*) malloc(
        sizeof(struct pde_PDEdriver__object));

    r_obj =
      (struct pde_PDEdriver__remote*) malloc(
        sizeof(struct pde_PDEdriver__remote));

    if(!self || !r_obj) {
      sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(
        _ex);
      SIDL_CHECK(*_ex);
      sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(
        *_ex);
      sidl_MemAllocException_add(ex, __FILE__, __LINE__, 
        "pde.PDEdriver.EPVgeneration", _ex);
      SIDL_CHECK(*_ex);
      *_ex = (struct sidl_BaseInterface__object*)ex;
      goto EXIT;
    }

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                          self;
    s1 =                          &s0->d_sidl_baseclass;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      pde_PDEdriver__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s1->d_sidl_baseinterface.d_object = (void*) self;

    s1->d_data = (void*) r_obj;
    s1->d_epv  = &s_rem_epv__sidl_baseclass;

    s0->d_gov_cca_component.d_epv    = &s_rem_epv__gov_cca_component;
    s0->d_gov_cca_component.d_object = (void*) self;

    s0->d_gov_cca_port.d_epv    = &s_rem_epv__gov_cca_port;
    s0->d_gov_cca_port.d_object = (void*) self;

    s0->d_gov_cca_ports_goport.d_epv    = &s_rem_epv__gov_cca_ports_goport;
    s0->d_gov_cca_ports_goport.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__pde_pdedriver;

    self->d_data = (void*) r_obj;

    sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
    return self;
    EXIT:
    if(self) { free(self); }
    if(r_obj) { free(r_obj); }
    return NULL;
  }
  // REMOTE: generate remote instance given URL string.
  static struct pde_PDEdriver__object*
  pde_PDEdriver__remoteCreate(const char *url, struct 
    sidl_BaseInterface__object **_ex)
  {
    struct sidl_BaseInterface__object* _throwaway_exception = NULL;
    struct pde_PDEdriver__object* self = NULL;

    struct pde_PDEdriver__object* s0;
    struct sidl_BaseClass__object* s1;

    struct pde_PDEdriver__remote* r_obj = NULL;
    sidl_rmi_InstanceHandle instance = sidl_rmi_ProtocolFactory_createInstance(
      url, "pde.PDEdriver", _ex ); SIDL_CHECK(*_ex);
    if ( instance == NULL) { return NULL; }
    self =
      (struct pde_PDEdriver__object*) malloc(
        sizeof(struct pde_PDEdriver__object));

    r_obj =
      (struct pde_PDEdriver__remote*) malloc(
        sizeof(struct pde_PDEdriver__remote));

    if(!self || !r_obj) {
      sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(
        _ex);
      SIDL_CHECK(*_ex);
      sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(
        *_ex);
      sidl_MemAllocException_add(ex, __FILE__, __LINE__, 
        "pde.PDEdriver.EPVgeneration", _ex);
      SIDL_CHECK(*_ex);
      *_ex = (struct sidl_BaseInterface__object*)ex;
      goto EXIT;
    }

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                          self;
    s1 =                          &s0->d_sidl_baseclass;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      pde_PDEdriver__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s1->d_sidl_baseinterface.d_object = (void*) self;

    s1->d_data = (void*) r_obj;
    s1->d_epv  = &s_rem_epv__sidl_baseclass;

    s0->d_gov_cca_component.d_epv    = &s_rem_epv__gov_cca_component;
    s0->d_gov_cca_component.d_object = (void*) self;

    s0->d_gov_cca_port.d_epv    = &s_rem_epv__gov_cca_port;
    s0->d_gov_cca_port.d_object = (void*) self;

    s0->d_gov_cca_ports_goport.d_epv    = &s_rem_epv__gov_cca_ports_goport;
    s0->d_gov_cca_ports_goport.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__pde_pdedriver;

    self->d_data = (void*) r_obj;

    return self;
    EXIT:
    if(instance) { sidl_rmi_InstanceHandle_deleteRef(instance, 
      &_throwaway_exception); }
    if(self) { free(self); }
    if(r_obj) { free(r_obj); }
    return NULL;
  }
  // 
  // RMI connector function for the class.
  // 
  struct pde_PDEdriver__object*
  pde_PDEdriver__connectI(const char* url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex)
  {
    return pde_PDEdriver__remoteConnect(url, ar, _ex);
  }


#endif /*WITH_RMI*/
}

//////////////////////////////////////////////////
// 
// Special methods for throwing exceptions
// 

void
pde::PDEdriver::throwException1(
  const char* methodName,
  struct sidl_BaseInterface__object *_exception
)
  // throws:
  //    ::gov::cca::CCAException
  //    ::sidl::RuntimeException
{
  void * _p = 0;
  struct sidl_BaseInterface__object *throwaway_exception;

  if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object, 
    "gov.cca.CCAException", &throwaway_exception)) != 0 ) {
    struct gov_cca_CCAException__object * _realtype = reinterpret_cast< struct 
      gov_cca_CCAException__object*>(_p);
    (*_exception->d_epv->f_deleteRef)(_exception->d_object, 
      &throwaway_exception);
    // Note: alternate constructor does not increment refcount.
    ::gov::cca::CCAException _resolved_exception = ::gov::cca::CCAException( 
      _realtype, false );
    (_resolved_exception._get_ior()->d_epv->f_add) (
      _resolved_exception._get_ior()->d_object , __FILE__, __LINE__, methodName,
      &throwaway_exception);throw _resolved_exception;
  }
  if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object, 
    "sidl.RuntimeException", &throwaway_exception)) != 0 ) {
    struct sidl_RuntimeException__object * _realtype = reinterpret_cast< struct 
      sidl_RuntimeException__object*>(_p);
    (*_exception->d_epv->f_deleteRef)(_exception->d_object, 
      &throwaway_exception);
    // Note: alternate constructor does not increment refcount.
    ::sidl::RuntimeException _resolved_exception = ::sidl::RuntimeException( 
      _realtype, false );
    (_resolved_exception._get_ior()->d_epv->f_add) (
      _resolved_exception._get_ior()->d_object , __FILE__, __LINE__, methodName,
      &throwaway_exception);throw _resolved_exception;
  }
  // Any unresolved exception is treated as LangSpecificException
  ::sidl::LangSpecificException _unexpected = 
    ::sidl::LangSpecificException::_create();
  _unexpected.add(__FILE__,__LINE__, "Unknown method");
  _unexpected.setNote("Unexpected exception received by C++ stub.");
  throw _unexpected;
}
void
pde::PDEdriver::throwException0(
  const char* methodName,
  struct sidl_BaseInterface__object *_exception
)
  // throws:
{
  void * _p = 0;
  struct sidl_BaseInterface__object *throwaway_exception;

  if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object, 
    "sidl.RuntimeException", &throwaway_exception)) != 0 ) {
    struct sidl_RuntimeException__object * _realtype = reinterpret_cast< struct 
      sidl_RuntimeException__object*>(_p);
    (*_exception->d_epv->f_deleteRef)(_exception->d_object, 
      &throwaway_exception);
    // Note: alternate constructor does not increment refcount.
    ::sidl::RuntimeException _resolved_exception = ::sidl::RuntimeException( 
      _realtype, false );
    (_resolved_exception._get_ior()->d_epv->f_add) (
      _resolved_exception._get_ior()->d_object , __FILE__, __LINE__, methodName,
      &throwaway_exception);throw _resolved_exception;
  }
  // Any unresolved exception is treated as LangSpecificException
  ::sidl::LangSpecificException _unexpected = 
    ::sidl::LangSpecificException::_create();
  _unexpected.add(__FILE__,__LINE__, "Unknown method");
  _unexpected.setNote("Unexpected exception received by C++ stub.");
  throw _unexpected;
}

//////////////////////////////////////////////////
// 
// User Defined Methods
// 


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
pde::PDEdriver::setServices( /* in */const ::gov::cca::Services& services )
// throws:
//   ::gov::cca::CCAException
//   ::sidl::RuntimeException

{

  ior_t* const loc_self = (struct pde_PDEdriver__object*) 
    ::pde::PDEdriver::_get_ior();
  struct gov_cca_Services__object* _local_services = (struct 
    gov_cca_Services__object*) services.::gov::cca::Services::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_setServices))(loc_self, /* in */ _local_services, 
    &_exception );
  /*dispatch to ior*/
  if (_exception != NULL ) {

    throwException1("setServices", _exception);
  }
  /*unpack results and cleanup*/
}



/**
 *  
 * Execute some encapsulated functionality on the component. 
 * Return 0 if ok, -1 if internal error but component may be 
 * used further, and -2 if error so severe that component cannot
 * be further used safely.
 */
int32_t
pde::PDEdriver::go(  )

{
  int32_t _result;
  ior_t* const loc_self = (struct pde_PDEdriver__object*) 
    ::pde::PDEdriver::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_go))(loc_self, &_exception );
  /*dispatch to ior*/
  if (_exception != NULL ) {

    throwException0("go", _exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



//////////////////////////////////////////////////
// 
// End User Defined Methods
// (everything else in this file is specific to
//  Babel's C++ bindings)
// 

// static constructor
::pde::PDEdriver
pde::PDEdriver::_create() {
  struct sidl_BaseInterface__object * _exception = NULL;
  ::pde::PDEdriver self( (*_get_ext()->createObject)(NULL, &_exception), false 
    );
  if (_exception != NULL) {
    throwException0("::pde::PDEdriver"
      "static constructor", _exception);
  }
  return self;
}

// Internal data wrapping method
::pde::PDEdriver::ior_t*
pde::PDEdriver::_wrapObj(void* private_data) {
  struct sidl_BaseInterface__object *_exception = NULL;
  ::pde::PDEdriver::ior_t* returnValue = (*_get_ext()->createObject)(
    private_data ,&_exception);
  if (_exception) {
    throwException0("::pde::PDEdriver._wrap", _exception);
  }
  return returnValue;
}

// remote constructor
::pde::PDEdriver
pde::PDEdriver::_create(const std::string& url) {
  ior_t* ior_self;
  struct sidl_BaseInterface__object *_exception = NULL;
  ior_self = pde_PDEdriver__remoteCreate( url.c_str(), &_exception );
  if (_exception != NULL ) {
    throwException0("::pde::PDEdriver remoteCreate", _exception);
  }
  return ::pde::PDEdriver( ior_self, false );
}

// remote connector
::pde::PDEdriver
pde::PDEdriver::_connect(const std::string& url, const bool ar ) {
  ior_t* ior_self;
  struct sidl_BaseInterface__object *_exception = NULL;
  ior_self = pde_PDEdriver__remoteConnect( url.c_str(), ar?TRUE:FALSE, 
    &_exception );
  if (_exception != NULL ) {
    throwException0("::pde::PDEdriver connect",_exception);
  }
  return ::pde::PDEdriver( ior_self, false );
}

// copy constructor
pde::PDEdriver::PDEdriver ( const ::pde::PDEdriver& original ) {
  _set_ior((struct pde_PDEdriver__object*) original.::pde::PDEdriver::_get_ior(
    ));
  if(d_self) {
    addRef();
  }
  d_weak_reference = false;
}

// assignment operator
::pde::PDEdriver&
pde::PDEdriver::operator=( const ::pde::PDEdriver& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    _set_ior((struct pde_PDEdriver__object*) rhs.::pde::PDEdriver::_get_ior());
    if(d_self) {
      addRef();
    }
    d_weak_reference = false;
  }
  return *this;
}

// conversion from ior to C++ class
pde::PDEdriver::PDEdriver ( ::pde::PDEdriver::ior_t* ior ) : 
  StubBase(reinterpret_cast< void*>(ior)), 
  ::gov::cca::Component((ior==NULL) ? NULL : &((*ior).d_gov_cca_component)),
  ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
  ::gov::cca::ports::GoPort((ior==NULL) ? NULL : &((
    *ior).d_gov_cca_ports_goport)) {}

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
pde::PDEdriver::PDEdriver ( ::pde::PDEdriver::ior_t* ior, bool isWeak ) : 
  StubBase(reinterpret_cast< void*>(ior), isWeak), 
  ::gov::cca::Component((ior==NULL) ? NULL : &((*ior).d_gov_cca_component)),
  ::gov::cca::Port((ior==NULL) ? NULL : &((*ior).d_gov_cca_port)),
  ::gov::cca::ports::GoPort((ior==NULL) ? NULL : &((
    *ior).d_gov_cca_ports_goport)) {}

// This safe IOR cast addresses Roundup issue475
int ::pde::PDEdriver::_set_ior_typesafe( struct sidl_BaseInterface__object *obj,
                                         const ::std::type_info &argtype) { 
  if ( obj == NULL || argtype == typeid(*this) ) {
    // optimized case:  _set_ior() is sufficient
    _set_ior( reinterpret_cast<ior_t*>(obj) );
    return 0;
  } else {
    // Attempt to downcast ior pointer to matching stub type
    ior_t* _my_ptr = NULL;
    if ((_my_ptr = _cast( obj )) == NULL ) {
      return 1;
    } else {
      _set_ior(_my_ptr);
      struct sidl_BaseInterface__object* _throwaway=NULL;
      sidl_BaseInterface_deleteRef(obj,&_throwaway);
      return 0;
    }
  }
}

// exec has special argument passing to avoid #include circularities
void ::pde::PDEdriver::_exec( const std::string& methodName, 
                        sidl::rmi::Call& inArgs,
                        sidl::rmi::Return& outArgs) { 
  ::pde::PDEdriver::ior_t* const loc_self = _get_ior();
  struct sidl_BaseInterface__object *throwaway_exception;
  (*loc_self->d_epv->f__exec)(loc_self,
                                methodName.c_str(),
                                inArgs._get_ior(),
                                outArgs._get_ior(),
                                &throwaway_exception);
}


/**
 * Get the URL of the Implementation of this object (for RMI)
 */
::std::string
pde::PDEdriver::_getURL(  )
// throws:
//   ::sidl::RuntimeException

{
  ::std::string _result;
  ior_t* const loc_self = (struct pde_PDEdriver__object*) 
    ::pde::PDEdriver::_get_ior();
  char * _local_result;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f__getURL))(loc_self, &_exception );
  /*dispatch to ior*/
  if (_exception != NULL ) {

    throwException0("_getURL", _exception);
  }
  if (_local_result) {
    _result = _local_result;
    ::sidl_String_free( _local_result );
  }
  /*unpack results and cleanup*/
  return _result;
}


/**
 * Method to enable/disable method hooks invocation.
 */
void
pde::PDEdriver::_set_hooks( /* in */bool enable )
// throws:
//   ::sidl::RuntimeException

{

  ior_t* const loc_self = (struct pde_PDEdriver__object*) 
    ::pde::PDEdriver::_get_ior();
  sidl_bool _local_enable = enable;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f__set_hooks))(loc_self, /* in */ _local_enable, 
    &_exception );
  /*dispatch to ior*/
  if (_exception != NULL ) {

    throwException0("_set_hooks", _exception);
  }
  /*unpack results and cleanup*/
}


/**
 * Method to enable/disable interface contract enforcement.
 */
void
pde::PDEdriver::_set_contracts( /* in */bool enable, /* in */const 
  ::std::string& enfFilename, /* in */bool resetCounters )
// throws:
//   ::sidl::RuntimeException

{

  ior_t* const loc_self = (struct pde_PDEdriver__object*) 
    ::pde::PDEdriver::_get_ior();
  sidl_bool _local_enable = enable;
  sidl_bool _local_resetCounters = resetCounters;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f__set_contracts))(loc_self, /* in */ _local_enable, /* 
    in */ enfFilename.c_str(), /* in */ _local_resetCounters, &_exception );
  /*dispatch to ior*/
  if (_exception != NULL ) {

    throwException0("_set_contracts", _exception);
  }
  /*unpack results and cleanup*/
}


/**
 * Method to dump contract enforcement statistics.
 */
void
pde::PDEdriver::_dump_stats( /* in */const ::std::string& filename, /* in 
  */const ::std::string& prefix )
// throws:
//   ::sidl::RuntimeException

{

  ior_t* const loc_self = (struct pde_PDEdriver__object*) 
    ::pde::PDEdriver::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f__dump_stats))(loc_self, /* in */ filename.c_str(), /* 
    in */ prefix.c_str(), &_exception );
  /*dispatch to ior*/
  if (_exception != NULL ) {

    throwException0("_dump_stats", _exception);
  }
  /*unpack results and cleanup*/
}

// protected method that implements casting
struct pde_PDEdriver__object* pde::PDEdriver::_cast(const void* src)
{
  static int connect_loaded = 0;
  ior_t* cast = NULL;

  if(!connect_loaded) {
    struct sidl_BaseInterface__object *throwaway_exception;
    sidl_rmi_ConnectRegistry_registerConnect("pde.PDEdriver", (
      void*)pde_PDEdriver__IHConnect, &throwaway_exception);
    connect_loaded = 1;
  }
  if ( src != 0 ) {
    // Actually, this thing is still const
    void* tmp = const_cast<void*>(src);
    struct sidl_BaseInterface__object *throwaway_exception;
    struct sidl_BaseInterface__object * base = reinterpret_cast< struct 
      sidl_BaseInterface__object *>(tmp);
    cast = reinterpret_cast< ior_t*>((*base->d_epv->f__cast)(base->d_object, 
      "pde.PDEdriver", &throwaway_exception));
  }
  return cast;
}

// Static data type
const ::pde::PDEdriver::ext_t * pde::PDEdriver::s_ext = 0;

// private static method to get static data type
const ::pde::PDEdriver::ext_t *
pde::PDEdriver::_get_ext()
  throw ( ::sidl::NullIORException)
{
  if (! s_ext ) {
#ifdef SIDL_STATIC_LIBRARY
    s_ext = pde_PDEdriver__externals();
#else
    s_ext = (struct pde_PDEdriver__external*)sidl_dynamicLoadIOR(
      "pde.PDEdriver","pde_PDEdriver__externals") ;
#endif
    sidl_checkIORVersion("pde.PDEdriver", s_ext->d_ior_major_version, 
      s_ext->d_ior_minor_version, 2, 0);
  }
  return s_ext;
}

