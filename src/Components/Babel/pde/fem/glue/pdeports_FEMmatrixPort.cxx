// 
// File:          pdeports_FEMmatrixPort.cxx
// Symbol:        pdeports.FEMmatrixPort-v0.1
// Symbol Type:   interface
// Babel Version: 1.4.0 (Revision: 6574 release-1-4-0)
// Description:   Client-side glue code for pdeports.FEMmatrixPort
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_pdeports_FEMmatrixPort_hxx
#include "pdeports_FEMmatrixPort.hxx"
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
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
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
static struct sidl_recursive_mutex_t pdeports__FEMmatrixPort__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &pdeports__FEMmatrixPort__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &pdeports__FEMmatrixPort__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &pdeports__FEMmatrixPort__mutex )==EDEADLOCK) */
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

  static struct pdeports__FEMmatrixPort__epv s_rem_epv__pdeports__femmatrixport;

  static struct gov_cca_Port__epv s_rem_epv__gov_cca_port;

  static struct pdeports_FEMmatrixPort__epv s_rem_epv__pdeports_femmatrixport;

  static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;


  // REMOTE CAST: dynamic type casting for remote objects.
  static void* remote_pdeports__FEMmatrixPort__cast(
    struct pdeports__FEMmatrixPort__object* self,
    const char* name, sidl_BaseInterface* _ex)
  {
    int cmp;
    void* cast = NULL;
    *_ex = NULL; /* default to no exception */
    cmp = strcmp(name, "pdeports._FEMmatrixPort");
    if (!cmp) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = ((struct pdeports__FEMmatrixPort__object*)self);
      return cast;
    }
    else if (cmp < 0) {
      cmp = strcmp(name, "pdeports.FEMmatrixPort");
      if (!cmp) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_pdeports_femmatrixport);
        return cast;
      }
      else if (cmp < 0) {
        cmp = strcmp(name, "gov.cca.Port");
        if (!cmp) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = &((*self).d_gov_cca_port);
          return cast;
        }
      }
    }
    else if (cmp > 0) {
      cmp = strcmp(name, "sidl.BaseInterface");
      if (!cmp) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_sidl_baseinterface);
        return cast;
      }
    }
    if ((*self->d_epv->f_isType)(self,name, _ex)) {
      void* (*func)(struct sidl_rmi_InstanceHandle__object*, struct 
        sidl_BaseInterface__object**) = 
        (void* (*)(struct sidl_rmi_InstanceHandle__object*, struct 
          sidl_BaseInterface__object**)) 
        sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
      cast =  (*func)(((struct 
        pdeports__FEMmatrixPort__remote*)self->d_data)->d_ih, _ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // REMOTE DELETE: call the remote destructor for the object.
  static void remote_pdeports__FEMmatrixPort__delete(
    struct pdeports__FEMmatrixPort__object* self,
    struct sidl_BaseInterface__object* *_ex)
  {
    *_ex = NULL;
    free((void*) self);
  }

  // REMOTE GETURL: call the getURL function for the object.
  static char* remote_pdeports__FEMmatrixPort__getURL(
    struct pdeports__FEMmatrixPort__object* self, struct 
      sidl_BaseInterface__object* *_ex)
  {
    struct sidl_rmi_InstanceHandle__object *conn = ((struct 
      pdeports__FEMmatrixPort__remote*)self->d_data)->d_ih;
    *_ex = NULL;
    if(conn != NULL) {
      return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
    }
    return NULL;
  }

  // REMOTE ADDREF: For internal babel use only! Remote addRef.
  static void remote_pdeports__FEMmatrixPort__raddRef(
    struct pdeports__FEMmatrixPort__object* self,struct 
      sidl_BaseInterface__object* *_ex)
  {
    struct sidl_BaseException__object* netex = NULL;
    // initialize a new invocation
    struct sidl_BaseInterface__object* _throwaway = NULL;
    struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
      pdeports__FEMmatrixPort__remote*)self->d_data)->d_ih;
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
  remote_pdeports__FEMmatrixPort__isRemote(
      struct pdeports__FEMmatrixPort__object* self, 
      struct sidl_BaseInterface__object* *_ex) {
    *_ex = NULL;
    return TRUE;
  }

  // REMOTE METHOD STUB:_set_hooks
  static void
  remote_pdeports__FEMmatrixPort__set_hooks(
    /* in */ struct pdeports__FEMmatrixPort__object*self ,
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
        pdeports__FEMmatrixPort__remote*)self->d_data)->d_ih;
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
      "Exception unserialized from pdeports._FEMmatrixPort._set_hooks.",
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
  remote_pdeports__FEMmatrixPort__set_contracts(
    /* in */ struct pdeports__FEMmatrixPort__object*self ,
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
        pdeports__FEMmatrixPort__remote*)self->d_data)->d_ih;
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
      "Exception unserialized from pdeports._FEMmatrixPort._set_contracts.",
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
  remote_pdeports__FEMmatrixPort__dump_stats(
    /* in */ struct pdeports__FEMmatrixPort__object*self ,
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
        pdeports__FEMmatrixPort__remote*)self->d_data)->d_ih;
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
      "Exception unserialized from pdeports._FEMmatrixPort._dump_stats.",
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
  static void remote_pdeports__FEMmatrixPort__exec(
    struct pdeports__FEMmatrixPort__object* self,const char* methodName,
    sidl_rmi_Call inArgs,
    sidl_rmi_Return outArgs,
    struct sidl_BaseInterface__object* *_ex)
  {
    *_ex = NULL;
  }

  // REMOTE METHOD STUB:makeFEMmatrices
  static int32_t
  remote_pdeports__FEMmatrixPort_makeFEMmatrices(
    /* in */ struct pdeports__FEMmatrixPort__object*self ,
    /* in array<int> */ struct sidl_int__array* mesh,
    /* in array<double> */ struct sidl_double__array* nodes,
    /* in array<int> */ struct sidl_int__array* dirichletNodes,
    /* in array<double> */ struct sidl_double__array* dirichletValues,
    /* out array<double,2> */ struct sidl_double__array** Ag,
    /* out array<double> */ struct sidl_double__array** fg,
    /* out */ int32_t* size,
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
        pdeports__FEMmatrixPort__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "makeFEMmatrices", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packIntArray( _inv, "mesh",mesh,0,0,0, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDoubleArray( _inv, "nodes",nodes,0,0,0, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "dirichletNodes",dirichletNodes,0,
        0,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDoubleArray( _inv, "dirichletValues",
        dirichletValues,0,0,0, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pdeports._FEMmatrixPort.makeFEMmatrices.",
          &throwaway_exception);
        *_ex = (struct sidl_BaseInterface__object*) sidl_BaseInterface__cast(
          _be,&throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments
      sidl_rmi_Response_unpackDoubleArray( _rsvp, "Ag", Ag,0,0,FALSE, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Response_unpackDoubleArray( _rsvp, "fg", fg,0,0,FALSE, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Response_unpackInt( _rsvp, "size", size, _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:addRef
  static void
  remote_pdeports__FEMmatrixPort_addRef(
    /* in */ struct pdeports__FEMmatrixPort__object*self ,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct pdeports__FEMmatrixPort__remote* r_obj = (struct 
        pdeports__FEMmatrixPort__remote*)self->d_data;
      LOCK_STATIC_GLOBALS;
      r_obj->d_refcount++;
#ifdef SIDL_DEBUG_REFCOUNT
      fprintf(stderr, "babel: addRef %p new count %d (type %s)\n",
        r_obj, r_obj->d_refcount, 
        "pdeports._FEMmatrixPort Remote Stub");
#endif /* SIDL_DEBUG_REFCOUNT */ 
      UNLOCK_STATIC_GLOBALS;
    }
  }

  // REMOTE METHOD STUB:deleteRef
  static void
  remote_pdeports__FEMmatrixPort_deleteRef(
    /* in */ struct pdeports__FEMmatrixPort__object*self ,
    /* out */ struct sidl_BaseInterface__object **_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct pdeports__FEMmatrixPort__remote* r_obj = (struct 
        pdeports__FEMmatrixPort__remote*)self->d_data;
      LOCK_STATIC_GLOBALS;
      r_obj->d_refcount--;
#ifdef SIDL_DEBUG_REFCOUNT
      fprintf(stderr, "babel: deleteRef %p new count %d (type %s)\n",r_obj, r_obj->d_refcount, "pdeports._FEMmatrixPort Remote Stub");
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
  remote_pdeports__FEMmatrixPort_isSame(
    /* in */ struct pdeports__FEMmatrixPort__object*self ,
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
        pdeports__FEMmatrixPort__remote*)self->d_data)->d_ih;
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
      "Exception unserialized from pdeports._FEMmatrixPort.isSame.",
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
  remote_pdeports__FEMmatrixPort_isType(
    /* in */ struct pdeports__FEMmatrixPort__object*self ,
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
        pdeports__FEMmatrixPort__remote*)self->d_data)->d_ih;
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
      "Exception unserialized from pdeports._FEMmatrixPort.isType.",
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
  remote_pdeports__FEMmatrixPort_getClassInfo(
    /* in */ struct pdeports__FEMmatrixPort__object*self ,
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
        pdeports__FEMmatrixPort__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "getClassInfo", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if (_be != NULL) {
        struct sidl_BaseInterface__object* throwaway_exception = NULL;
        sidl_BaseException_addLine(_be, 
      "Exception unserialized from pdeports._FEMmatrixPort.getClassInfo.",
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

  // REMOTE EPV: create remote entry point vectors (EPVs).
  static void pdeports__FEMmatrixPort__init_remote_epv(void)
  {
    // assert( HAVE_LOCKED_STATIC_GLOBALS );
    struct pdeports__FEMmatrixPort__epv* epv = 
      &s_rem_epv__pdeports__femmatrixport;
    struct gov_cca_Port__epv*            e0  = &s_rem_epv__gov_cca_port;
    struct pdeports_FEMmatrixPort__epv*  e1  = 
      &s_rem_epv__pdeports_femmatrixport;
    struct sidl_BaseInterface__epv*      e2  = &s_rem_epv__sidl_baseinterface;

    epv->f__cast                = remote_pdeports__FEMmatrixPort__cast;
    epv->f__delete              = remote_pdeports__FEMmatrixPort__delete;
    epv->f__exec                = remote_pdeports__FEMmatrixPort__exec;
    epv->f__getURL              = remote_pdeports__FEMmatrixPort__getURL;
    epv->f__raddRef             = remote_pdeports__FEMmatrixPort__raddRef;
    epv->f__isRemote            = remote_pdeports__FEMmatrixPort__isRemote;
    epv->f__set_hooks           = remote_pdeports__FEMmatrixPort__set_hooks;
    epv->f__set_contracts       = remote_pdeports__FEMmatrixPort__set_contracts;
    epv->f__dump_stats          = remote_pdeports__FEMmatrixPort__dump_stats;
    epv->f__ctor                = NULL;
    epv->f__ctor2               = NULL;
    epv->f__dtor                = NULL;
    epv->f_makeFEMmatrices      = 
      remote_pdeports__FEMmatrixPort_makeFEMmatrices;
    epv->f_addRef               = remote_pdeports__FEMmatrixPort_addRef;
    epv->f_deleteRef            = remote_pdeports__FEMmatrixPort_deleteRef;
    epv->f_isSame               = remote_pdeports__FEMmatrixPort_isSame;
    epv->f_isType               = remote_pdeports__FEMmatrixPort_isType;
    epv->f_getClassInfo         = remote_pdeports__FEMmatrixPort_getClassInfo;

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

    e1->f__cast           = (void* (*)(void*, const char*, struct 
      sidl_BaseInterface__object**)) epv->f__cast;
    e1->f__delete         = (void (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__delete;
    e1->f__getURL         = (char* (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__getURL;
    e1->f__raddRef        = (void (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__raddRef;
    e1->f__isRemote       = (sidl_bool (*)(void*, struct 
      sidl_BaseInterface__object**)) epv->f__isRemote;
    e1->f__set_hooks      = (void (*)(void*, sidl_bool, struct 
      sidl_BaseInterface__object**)) epv->f__set_hooks;
    e1->f__set_contracts  = (void (*)(void*, sidl_bool, const char*, sidl_bool, 
      struct sidl_BaseInterface__object**)) epv->f__set_contracts;
    e1->f__dump_stats     = (void (*)(void*, const char*, const char*, struct 
      sidl_BaseInterface__object**)) epv->f__dump_stats;
    e1->f__exec           = (void (*)(void*,const char*,struct 
      sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
      sidl_BaseInterface__object **)) epv->f__exec;
    e1->f_makeFEMmatrices = (int32_t (*)(void*,struct sidl_int__array*,struct 
      sidl_double__array*,struct sidl_int__array*,struct sidl_double__array*,
      struct sidl_double__array**,struct sidl_double__array**,int32_t*,struct 
      sidl_BaseInterface__object **)) epv->f_makeFEMmatrices;
    e1->f_addRef          = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_addRef;
    e1->f_deleteRef       = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_deleteRef;
    e1->f_isSame          = (sidl_bool (*)(void*,struct 
      sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
      epv->f_isSame;
    e1->f_isType          = (sidl_bool (*)(void*,const char*,struct 
      sidl_BaseInterface__object **)) epv->f_isType;
    e1->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,struct 
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

    s_remote_initialized = 1;
  }

  // Create an instance that connects to an existing remote object.
  static struct pdeports_FEMmatrixPort__object*
  pdeports_FEMmatrixPort__remoteConnect(const char *url, sidl_bool ar, struct 
    sidl_BaseInterface__object* *_ex)
  {
    struct pdeports__FEMmatrixPort__object* self = NULL;

    struct pdeports__FEMmatrixPort__object* s0;

    struct pdeports__FEMmatrixPort__remote* r_obj = NULL;
    sidl_rmi_InstanceHandle instance = NULL;
    struct pdeports_FEMmatrixPort__object* ret_self = NULL;
    char* objectID = NULL;
    objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
    if(objectID) {
      sidl_BaseInterface bi = (sidl_BaseInterface) 
        sidl_rmi_InstanceRegistry_getInstanceByString(objectID, _ex);
      return (struct pdeports_FEMmatrixPort__object*)(*bi->d_epv->f__cast)(
        bi->d_object, "pdeports.FEMmatrixPort", _ex);
    }
    instance = sidl_rmi_ProtocolFactory_connectInstance(url, 
      "pdeports.FEMmatrixPort", ar, _ex );
    if ( instance == NULL) { return NULL; }
    self =
      (struct pdeports__FEMmatrixPort__object*) malloc(
        sizeof(struct pdeports__FEMmatrixPort__object));

    r_obj =
      (struct pdeports__FEMmatrixPort__remote*) malloc(
        sizeof(struct pdeports__FEMmatrixPort__remote));

    if(!self || !r_obj) {
      sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(
        _ex);
      SIDL_CHECK(*_ex);
      sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(
        *_ex);
      sidl_MemAllocException_add(ex, __FILE__, __LINE__, 
        "pdeports._FEMmatrixPort.EPVgeneration", _ex);
      SIDL_CHECK(*_ex);
      *_ex = (struct sidl_BaseInterface__object*)ex;
      goto EXIT;
    }

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                                   self;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      pdeports__FEMmatrixPort__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s0->d_gov_cca_port.d_epv    = &s_rem_epv__gov_cca_port;
    s0->d_gov_cca_port.d_object = (void*) self;

    s0->d_pdeports_femmatrixport.d_epv    = &s_rem_epv__pdeports_femmatrixport;
    s0->d_pdeports_femmatrixport.d_object = (void*) self;

    s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s0->d_sidl_baseinterface.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__pdeports__femmatrixport;

    self->d_data = (void*) r_obj;

    ret_self = (struct pdeports_FEMmatrixPort__object*) (*self->d_epv->f__cast)(
      self, "pdeports.FEMmatrixPort", _ex);
    if(*_ex || !ret_self) { goto EXIT; }
    return ret_self;
    EXIT:
    if(self) { free(self); }
    if(r_obj) { free(r_obj); }
    return NULL;
  }
  // Create an instance that uses an already existing 
  // InstanceHandel to connect to an existing remote object.
  static struct pdeports_FEMmatrixPort__object*
  pdeports_FEMmatrixPort__IHConnect(sidl_rmi_InstanceHandle instance, struct 
    sidl_BaseInterface__object* *_ex)
  {
    struct pdeports__FEMmatrixPort__object* self = NULL;

    struct pdeports__FEMmatrixPort__object* s0;

    struct pdeports__FEMmatrixPort__remote* r_obj = NULL;
    struct pdeports_FEMmatrixPort__object* ret_self = NULL;
    self =
      (struct pdeports__FEMmatrixPort__object*) malloc(
        sizeof(struct pdeports__FEMmatrixPort__object));

    r_obj =
      (struct pdeports__FEMmatrixPort__remote*) malloc(
        sizeof(struct pdeports__FEMmatrixPort__remote));

    if(!self || !r_obj) {
      sidl_MemAllocException ex = sidl_MemAllocException_getSingletonException(
        _ex);
      SIDL_CHECK(*_ex);
      sidl_MemAllocException_setNote(ex, "Out of memory.", _ex); SIDL_CHECK(
        *_ex);
      sidl_MemAllocException_add(ex, __FILE__, __LINE__, 
        "pdeports._FEMmatrixPort.EPVgeneration", _ex);
      SIDL_CHECK(*_ex);
      *_ex = (struct sidl_BaseInterface__object*)ex;
      goto EXIT;
    }

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                                   self;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      pdeports__FEMmatrixPort__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s0->d_gov_cca_port.d_epv    = &s_rem_epv__gov_cca_port;
    s0->d_gov_cca_port.d_object = (void*) self;

    s0->d_pdeports_femmatrixport.d_epv    = &s_rem_epv__pdeports_femmatrixport;
    s0->d_pdeports_femmatrixport.d_object = (void*) self;

    s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s0->d_sidl_baseinterface.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__pdeports__femmatrixport;

    self->d_data = (void*) r_obj;

    sidl_rmi_InstanceHandle_addRef(instance, _ex);

    ret_self = (struct pdeports_FEMmatrixPort__object*) (*self->d_epv->f__cast)(
      self, "pdeports.FEMmatrixPort", _ex);
    if(*_ex || !ret_self) { goto EXIT; }
    return ret_self;
    EXIT:
    if(self) { free(self); }
    if(r_obj) { free(r_obj); }
    return NULL;
  }
  // 
  // RMI connector function for the class.
  // 
  struct pdeports_FEMmatrixPort__object*
  pdeports_FEMmatrixPort__connectI(const char* url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex)
  {
    return pdeports_FEMmatrixPort__remoteConnect(url, ar, _ex);
  }


#endif /*WITH_RMI*/
}

//////////////////////////////////////////////////
// 
// Special methods for throwing exceptions
// 

void
pdeports::FEMmatrixPort::throwException0(
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
 * user defined non-static method.
 */
int32_t
pdeports::FEMmatrixPort::makeFEMmatrices( /* in array<int> */const 
  ::sidl::array<int32_t>& mesh, /* in array<double> */const 
  ::sidl::array<double>& nodes, /* in array<int> */const 
  ::sidl::array<int32_t>& dirichletNodes, /* in array<double> */const 
  ::sidl::array<double>& dirichletValues, /* out array<double,2> 
  */::sidl::array<double>& Ag, /* out array<double> */::sidl::array<double>& fg,
  /* out */int32_t& size )

{
  int32_t _result;
  ior_t* const loc_self = (struct pdeports_FEMmatrixPort__object*) 
    ::pdeports::FEMmatrixPort::_get_ior();
  struct sidl_double__array* _local_Ag;
  struct sidl_double__array* _local_fg;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_makeFEMmatrices))(loc_self->d_object, /* in 
    array<int> */ mesh._get_ior(), /* in array<double> */ nodes._get_ior(), /* 
    in array<int> */ dirichletNodes._get_ior(), /* in array<double> */ 
    dirichletValues._get_ior(), /* out array<double,2> */ &_local_Ag, /* out 
    array<double> */ &_local_fg, /* out */ &size, &_exception );
  /*dispatch to ior*/
  if (_exception != NULL ) {

    throwException0("makeFEMmatrices", _exception);
  }
  Ag._set_ior(_local_Ag);
  fg._set_ior(_local_fg);
  /*unpack results and cleanup*/
  return _result;
}



//////////////////////////////////////////////////
// 
// End User Defined Methods
// (everything else in this file is specific to
//  Babel's C++ bindings)
// 

// remote connector
::pdeports::FEMmatrixPort
pdeports::FEMmatrixPort::_connect(const std::string& url, const bool ar ) {
  ior_t* ior_self;
  struct sidl_BaseInterface__object *_exception = NULL;
  ior_self = pdeports_FEMmatrixPort__remoteConnect( url.c_str(), ar?TRUE:FALSE, 
    &_exception );
  if (_exception != NULL ) {
    throwException0("::pdeports::FEMmatrixPort connect",_exception);
  }
  return ::pdeports::FEMmatrixPort( ior_self, false );
}

// copy constructor
pdeports::FEMmatrixPort::FEMmatrixPort ( const ::pdeports::FEMmatrixPort& 
  original ) {
  _set_ior((struct pdeports_FEMmatrixPort__object*) 
    original.::pdeports::FEMmatrixPort::_get_ior());
  if(d_self) {
    addRef();
  }
  d_weak_reference = false;
}

// assignment operator
::pdeports::FEMmatrixPort&
pdeports::FEMmatrixPort::operator=( const ::pdeports::FEMmatrixPort& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    _set_ior((struct pdeports_FEMmatrixPort__object*) 
      rhs.::pdeports::FEMmatrixPort::_get_ior());
    if(d_self) {
      addRef();
    }
    d_weak_reference = false;
  }
  return *this;
}

// conversion from ior to C++ class
pdeports::FEMmatrixPort::FEMmatrixPort ( ::pdeports::FEMmatrixPort::ior_t* ior 
  ) : 
  StubBase(reinterpret_cast< void*>(ior)), 
  pdeports_FEMmatrixPort_IORCache((ior_t*) ior) {}

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
pdeports::FEMmatrixPort::FEMmatrixPort ( ::pdeports::FEMmatrixPort::ior_t* ior, 
  bool isWeak ) : 
  StubBase(reinterpret_cast< void*>(ior), isWeak), 
pdeports_FEMmatrixPort_IORCache((ior_t*) ior) {}

// This safe IOR cast addresses Roundup issue475
int ::pdeports::FEMmatrixPort::_set_ior_typesafe( struct 
  sidl_BaseInterface__object *obj,
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
void ::pdeports::FEMmatrixPort::_exec( const std::string& methodName, 
                        sidl::rmi::Call& inArgs,
                        sidl::rmi::Return& outArgs) { 
  ::pdeports::FEMmatrixPort::ior_t* const loc_self = _get_ior();
  struct sidl_BaseInterface__object *throwaway_exception;
  (*loc_self->d_epv->f__exec)(loc_self->d_object,
                                methodName.c_str(),
                                inArgs._get_ior(),
                                outArgs._get_ior(),
                                &throwaway_exception);
}


/**
 * Get the URL of the Implementation of this object (for RMI)
 */
::std::string
pdeports::FEMmatrixPort::_getURL(  )
// throws:
//   ::sidl::RuntimeException

{
  ::std::string _result;
  ior_t* const loc_self = (struct pdeports_FEMmatrixPort__object*) 
    ::pdeports::FEMmatrixPort::_get_ior();
  char * _local_result;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f__getURL))(loc_self->d_object, 
    &_exception );
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
pdeports::FEMmatrixPort::_set_hooks( /* in */bool enable )
// throws:
//   ::sidl::RuntimeException

{

  ior_t* const loc_self = (struct pdeports_FEMmatrixPort__object*) 
    ::pdeports::FEMmatrixPort::_get_ior();
  sidl_bool _local_enable = enable;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f__set_hooks))(loc_self->d_object, /* in */ _local_enable,
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
pdeports::FEMmatrixPort::_set_contracts( /* in */bool enable, /* in */const 
  ::std::string& enfFilename, /* in */bool resetCounters )
// throws:
//   ::sidl::RuntimeException

{

  ior_t* const loc_self = (struct pdeports_FEMmatrixPort__object*) 
    ::pdeports::FEMmatrixPort::_get_ior();
  sidl_bool _local_enable = enable;
  sidl_bool _local_resetCounters = resetCounters;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f__set_contracts))(loc_self->d_object, /* in */ 
    _local_enable, /* in */ enfFilename.c_str(), /* in */ _local_resetCounters, 
    &_exception );
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
pdeports::FEMmatrixPort::_dump_stats( /* in */const ::std::string& filename, /* 
  in */const ::std::string& prefix )
// throws:
//   ::sidl::RuntimeException

{

  ior_t* const loc_self = (struct pdeports_FEMmatrixPort__object*) 
    ::pdeports::FEMmatrixPort::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f__dump_stats))(loc_self->d_object, /* in */ 
    filename.c_str(), /* in */ prefix.c_str(), &_exception );
  /*dispatch to ior*/
  if (_exception != NULL ) {

    throwException0("_dump_stats", _exception);
  }
  /*unpack results and cleanup*/
}

// protected method that implements casting
struct pdeports_FEMmatrixPort__object* pdeports::FEMmatrixPort::_cast(const 
  void* src)
{
  static int connect_loaded = 0;
  ior_t* cast = NULL;

  if(!connect_loaded) {
    struct sidl_BaseInterface__object *throwaway_exception;
    sidl_rmi_ConnectRegistry_registerConnect("pdeports.FEMmatrixPort", (
      void*)pdeports_FEMmatrixPort__IHConnect, &throwaway_exception);
    connect_loaded = 1;
  }
  if ( src != 0 ) {
    // Actually, this thing is still const
    void* tmp = const_cast<void*>(src);
    struct sidl_BaseInterface__object *throwaway_exception;
    struct sidl_BaseInterface__object * base = reinterpret_cast< struct 
      sidl_BaseInterface__object *>(tmp);
    cast = reinterpret_cast< ior_t*>((*base->d_epv->f__cast)(base->d_object, 
      "pdeports.FEMmatrixPort", &throwaway_exception));
  }
  return cast;
}

