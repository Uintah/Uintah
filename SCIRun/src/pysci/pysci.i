
%module pysci
%{
#include "api.h"
#include "../Core/Thread/Mutex.h"
#include "../Core/Persistent/Persistent.h"
#include "../Core/Util/ProgressReporter.h"
#include "../Core/Datatypes/Datatype.h"
#include "../Core/Events/BaseEvent.h"
#include "../Core/Geom/OpenGLContext.h"
#include "../Core/Geom/CallbackOpenGLContext.h"

#include <iostream>

static int PythonCallBack(void *clientdata)
{
   PyObject *func, *arglist;
   PyObject *result;
   int    ires = 0;
   
   func = (PyObject *) clientdata;               // Get Python function
   if (! func) {
     std::cerr << "NULL python function pointer" << std::endl;
     return 0;
   }	
   PyGILState_STATE state = PyGILState_Ensure();	


   arglist = Py_BuildValue("()");                // Build argument list
   result = PyEval_CallObject(func, arglist);     // Call Python

   Py_DECREF(arglist);                           // Trash arglist
   if (result) {                                 // If no errors, return int
     ires = PyInt_AsLong(result);
     Py_DECREF(result);
   }

   PyGILState_Release(state);
   return ires;
}


%}
%include std_vector.i
%include std_string.i
%include std_map.i
%include std_deque.i

%include "../Core/Geom/OpenGLContext.h"
%include "../Core/Geom/CallbackOpenGLContext.h"
%include "../Core/Thread/Mutex.h"
%include "../Core/Util/ProgressReporter.h"
%include "../Core/Persistent/Persistent.h"
%include "../Core/Datatypes/Datatype.h"
%include "../Core/Events/BaseEvent.h"


namespace std {
   %template(vector_string) vector<string>;
   %template(vector_double) vector<double>;
};

// This tells SWIG to treat char ** as a special case
%typemap(python,in) char ** {
  /* Check if is a list */
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    $1 = (char **) malloc((size+1)*sizeof(char *));
    for (i = 0; i < size; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (PyString_Check(o))
	$1[i] = PyString_AsString(PyList_GetItem($input,i));
      else {
	PyErr_SetString(PyExc_TypeError,"list must contain strings");
	free($1);
	return NULL;
      }
    }
    $1[i] = 0;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}

// This cleans up the char ** array we malloc'd before the function call
%typemap(python,freearg) char ** {
  free((char *) $1);
}

%exception {
  Py_BEGIN_ALLOW_THREADS
  $function
  Py_END_ALLOW_THREADS
}

// Grab a Python function object as a Python object.
%typemap(python,in) PyObject *pyfunc {
  if (!PyCallable_Check($input)) {
      PyErr_SetString(PyExc_TypeError, "Need a callable object!");
      return NULL;
  }
  $1 = $input;
}


// Attach a new method to our plot widget for adding Python functions
%extend SCIRun::CallbackOpenGLContext {
   // Set a Python function object as a callback function
   // Note : PyObject *pyfunc is remapped with a typempap
   void set_pymake_current_func(PyObject *pyfunc) {
     self->set_make_current_func(PythonCallBack, (void *) pyfunc);
     Py_INCREF(pyfunc);
   }
   void set_pyrelease_func(PyObject *pyfunc) {
     self->set_release_func(PythonCallBack, (void *) pyfunc);
     Py_INCREF(pyfunc);
   }
   void set_pyswap_func(PyObject *pyfunc) {
     self->set_swap_func(PythonCallBack, (void *) pyfunc);
     Py_INCREF(pyfunc);
   }
   void set_pywidth_func(PyObject *pyfunc) {
     self->set_width_func(PythonCallBack, (void *) pyfunc);
     Py_INCREF(pyfunc);
   }
   void set_pyheight_func(PyObject *pyfunc) {
     self->set_height_func(PythonCallBack, (void *) pyfunc);
     Py_INCREF(pyfunc);
   }
}



%include "api.h"

