# This file was created automatically by SWIG.
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.

import _pysci

def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "this"):
        if isinstance(value, class_type):
            self.__dict__[name] = value.this
            if hasattr(value,"thisown"): self.__dict__["thisown"] = value.thisown
            del value.thisown
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name) or (name == "thisown"):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0
del types


class OpenGLContext(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, OpenGLContext, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, OpenGLContext, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::OpenGLContext instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __del__(self, destroy=_pysci.delete_OpenGLContext):
        try:
            if self.thisown: destroy(self)
        except: pass

    def make_current(*args): return _pysci.OpenGLContext_make_current(*args)
    def release(*args): return _pysci.OpenGLContext_release(*args)
    def width(*args): return _pysci.OpenGLContext_width(*args)
    def height(*args): return _pysci.OpenGLContext_height(*args)
    def swap(*args): return _pysci.OpenGLContext_swap(*args)

class OpenGLContextPtr(OpenGLContext):
    def __init__(self, this):
        _swig_setattr(self, OpenGLContext, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, OpenGLContext, 'thisown', 0)
        _swig_setattr(self, OpenGLContext,self.__class__,OpenGLContext)
_pysci.OpenGLContext_swigregister(OpenGLContextPtr)

class CallbackOpenGLContext(OpenGLContext):
    __swig_setmethods__ = {}
    for _s in [OpenGLContext]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, CallbackOpenGLContext, name, value)
    __swig_getmethods__ = {}
    for _s in [OpenGLContext]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, CallbackOpenGLContext, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::CallbackOpenGLContext instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, CallbackOpenGLContext, 'this', _pysci.new_CallbackOpenGLContext(*args))
        _swig_setattr(self, CallbackOpenGLContext, 'thisown', 1)
    def __del__(self, destroy=_pysci.delete_CallbackOpenGLContext):
        try:
            if self.thisown: destroy(self)
        except: pass

    def make_current(*args): return _pysci.CallbackOpenGLContext_make_current(*args)
    def release(*args): return _pysci.CallbackOpenGLContext_release(*args)
    def width(*args): return _pysci.CallbackOpenGLContext_width(*args)
    def height(*args): return _pysci.CallbackOpenGLContext_height(*args)
    def swap(*args): return _pysci.CallbackOpenGLContext_swap(*args)
    def set_make_current_func(*args): return _pysci.CallbackOpenGLContext_set_make_current_func(*args)
    def set_release_func(*args): return _pysci.CallbackOpenGLContext_set_release_func(*args)
    def set_width_func(*args): return _pysci.CallbackOpenGLContext_set_width_func(*args)
    def set_height_func(*args): return _pysci.CallbackOpenGLContext_set_height_func(*args)
    def set_swap_func(*args): return _pysci.CallbackOpenGLContext_set_swap_func(*args)
    def set_pymake_current_func(*args): return _pysci.CallbackOpenGLContext_set_pymake_current_func(*args)
    def set_pyrelease_func(*args): return _pysci.CallbackOpenGLContext_set_pyrelease_func(*args)
    def set_pyswap_func(*args): return _pysci.CallbackOpenGLContext_set_pyswap_func(*args)
    def set_pywidth_func(*args): return _pysci.CallbackOpenGLContext_set_pywidth_func(*args)
    def set_pyheight_func(*args): return _pysci.CallbackOpenGLContext_set_pyheight_func(*args)

class CallbackOpenGLContextPtr(CallbackOpenGLContext):
    def __init__(self, this):
        _swig_setattr(self, CallbackOpenGLContext, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, CallbackOpenGLContext, 'thisown', 0)
        _swig_setattr(self, CallbackOpenGLContext,self.__class__,CallbackOpenGLContext)
_pysci.CallbackOpenGLContext_swigregister(CallbackOpenGLContextPtr)

class vector_string(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, vector_string, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, vector_string, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ std::vector<std::string > instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def empty(*args): return _pysci.vector_string_empty(*args)
    def size(*args): return _pysci.vector_string_size(*args)
    def clear(*args): return _pysci.vector_string_clear(*args)
    def swap(*args): return _pysci.vector_string_swap(*args)
    def get_allocator(*args): return _pysci.vector_string_get_allocator(*args)
    def pop_back(*args): return _pysci.vector_string_pop_back(*args)
    def __init__(self, *args):
        _swig_setattr(self, vector_string, 'this', _pysci.new_vector_string(*args))
        _swig_setattr(self, vector_string, 'thisown', 1)
    def push_back(*args): return _pysci.vector_string_push_back(*args)
    def front(*args): return _pysci.vector_string_front(*args)
    def back(*args): return _pysci.vector_string_back(*args)
    def assign(*args): return _pysci.vector_string_assign(*args)
    def resize(*args): return _pysci.vector_string_resize(*args)
    def reserve(*args): return _pysci.vector_string_reserve(*args)
    def capacity(*args): return _pysci.vector_string_capacity(*args)
    def __nonzero__(*args): return _pysci.vector_string___nonzero__(*args)
    def __len__(*args): return _pysci.vector_string___len__(*args)
    def pop(*args): return _pysci.vector_string_pop(*args)
    def __getslice__(*args): return _pysci.vector_string___getslice__(*args)
    def __setslice__(*args): return _pysci.vector_string___setslice__(*args)
    def __delslice__(*args): return _pysci.vector_string___delslice__(*args)
    def __delitem__(*args): return _pysci.vector_string___delitem__(*args)
    def __getitem__(*args): return _pysci.vector_string___getitem__(*args)
    def __setitem__(*args): return _pysci.vector_string___setitem__(*args)
    def append(*args): return _pysci.vector_string_append(*args)
    def __del__(self, destroy=_pysci.delete_vector_string):
        try:
            if self.thisown: destroy(self)
        except: pass


class vector_stringPtr(vector_string):
    def __init__(self, this):
        _swig_setattr(self, vector_string, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, vector_string, 'thisown', 0)
        _swig_setattr(self, vector_string,self.__class__,vector_string)
_pysci.vector_string_swigregister(vector_stringPtr)

class vector_double(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, vector_double, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, vector_double, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ std::vector<double > instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def empty(*args): return _pysci.vector_double_empty(*args)
    def size(*args): return _pysci.vector_double_size(*args)
    def clear(*args): return _pysci.vector_double_clear(*args)
    def swap(*args): return _pysci.vector_double_swap(*args)
    def get_allocator(*args): return _pysci.vector_double_get_allocator(*args)
    def pop_back(*args): return _pysci.vector_double_pop_back(*args)
    def __init__(self, *args):
        _swig_setattr(self, vector_double, 'this', _pysci.new_vector_double(*args))
        _swig_setattr(self, vector_double, 'thisown', 1)
    def push_back(*args): return _pysci.vector_double_push_back(*args)
    def front(*args): return _pysci.vector_double_front(*args)
    def back(*args): return _pysci.vector_double_back(*args)
    def assign(*args): return _pysci.vector_double_assign(*args)
    def resize(*args): return _pysci.vector_double_resize(*args)
    def reserve(*args): return _pysci.vector_double_reserve(*args)
    def capacity(*args): return _pysci.vector_double_capacity(*args)
    def __nonzero__(*args): return _pysci.vector_double___nonzero__(*args)
    def __len__(*args): return _pysci.vector_double___len__(*args)
    def pop(*args): return _pysci.vector_double_pop(*args)
    def __getslice__(*args): return _pysci.vector_double___getslice__(*args)
    def __setslice__(*args): return _pysci.vector_double___setslice__(*args)
    def __delslice__(*args): return _pysci.vector_double___delslice__(*args)
    def __delitem__(*args): return _pysci.vector_double___delitem__(*args)
    def __getitem__(*args): return _pysci.vector_double___getitem__(*args)
    def __setitem__(*args): return _pysci.vector_double___setitem__(*args)
    def append(*args): return _pysci.vector_double_append(*args)
    def __del__(self, destroy=_pysci.delete_vector_double):
        try:
            if self.thisown: destroy(self)
        except: pass


class vector_doublePtr(vector_double):
    def __init__(self, this):
        _swig_setattr(self, vector_double, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, vector_double, 'thisown', 0)
        _swig_setattr(self, vector_double,self.__class__,vector_double)
_pysci.vector_double_swigregister(vector_doublePtr)


init_pysci = _pysci.init_pysci

terminate = _pysci.terminate

test_function = _pysci.test_function

tetgen_2surf = _pysci.tetgen_2surf

run_viewer_thread = _pysci.run_viewer_thread

add_key_event = _pysci.add_key_event

add_motion_notify_event = _pysci.add_motion_notify_event

add_pointer_down_event = _pysci.add_pointer_down_event

add_pointer_up_event = _pysci.add_pointer_up_event

