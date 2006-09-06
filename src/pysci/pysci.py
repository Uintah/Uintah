# This file was created automatically by SWIG 1.3.27.
# Don't modify this file, modify the SWIG interface instead.

import _pysci

# This file is compatible with both classic and new-style classes.
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
        self.__class__ = OpenGLContext
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
        self.__class__ = CallbackOpenGLContext
_pysci.CallbackOpenGLContext_swigregister(CallbackOpenGLContextPtr)

class Mutex(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Mutex, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Mutex, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::Mutex instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, Mutex, 'this', _pysci.new_Mutex(*args))
        _swig_setattr(self, Mutex, 'thisown', 1)
    def __del__(self, destroy=_pysci.delete_Mutex):
        try:
            if self.thisown: destroy(self)
        except: pass

    def lock(*args): return _pysci.Mutex_lock(*args)
    def tryLock(*args): return _pysci.Mutex_tryLock(*args)
    def unlock(*args): return _pysci.Mutex_unlock(*args)

class MutexPtr(Mutex):
    def __init__(self, this):
        _swig_setattr(self, Mutex, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Mutex, 'thisown', 0)
        self.__class__ = Mutex
_pysci.Mutex_swigregister(MutexPtr)

class ProgressReporter(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, ProgressReporter, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, ProgressReporter, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::ProgressReporter instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    Starting = _pysci.ProgressReporter_Starting
    Compiling = _pysci.ProgressReporter_Compiling
    CompilationDone = _pysci.ProgressReporter_CompilationDone
    Done = _pysci.ProgressReporter_Done
    def __init__(self, *args):
        _swig_setattr(self, ProgressReporter, 'this', _pysci.new_ProgressReporter(*args))
        _swig_setattr(self, ProgressReporter, 'thisown', 1)
    def __del__(self, destroy=_pysci.delete_ProgressReporter):
        try:
            if self.thisown: destroy(self)
        except: pass

    def error(*args): return _pysci.ProgressReporter_error(*args)
    def warning(*args): return _pysci.ProgressReporter_warning(*args)
    def remark(*args): return _pysci.ProgressReporter_remark(*args)
    def compile_error(*args): return _pysci.ProgressReporter_compile_error(*args)
    def add_raw_message(*args): return _pysci.ProgressReporter_add_raw_message(*args)
    def msg_stream(*args): return _pysci.ProgressReporter_msg_stream(*args)
    def msg_stream_flush(*args): return _pysci.ProgressReporter_msg_stream_flush(*args)
    def report_progress(*args): return _pysci.ProgressReporter_report_progress(*args)
    def update_progress(*args): return _pysci.ProgressReporter_update_progress(*args)
    def increment_progress(*args): return _pysci.ProgressReporter_increment_progress(*args)

class ProgressReporterPtr(ProgressReporter):
    def __init__(self, this):
        _swig_setattr(self, ProgressReporter, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, ProgressReporter, 'thisown', 0)
        self.__class__ = ProgressReporter
_pysci.ProgressReporter_swigregister(ProgressReporterPtr)

SCI_project_Persistent_h = _pysci.SCI_project_Persistent_h
class PersistentTypeID(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, PersistentTypeID, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, PersistentTypeID, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::PersistentTypeID instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    __swig_setmethods__["type"] = _pysci.PersistentTypeID_type_set
    __swig_getmethods__["type"] = _pysci.PersistentTypeID_type_get
    if _newclass:type = property(_pysci.PersistentTypeID_type_get, _pysci.PersistentTypeID_type_set)
    __swig_setmethods__["parent"] = _pysci.PersistentTypeID_parent_set
    __swig_getmethods__["parent"] = _pysci.PersistentTypeID_parent_get
    if _newclass:parent = property(_pysci.PersistentTypeID_parent_get, _pysci.PersistentTypeID_parent_set)
    __swig_setmethods__["maker"] = _pysci.PersistentTypeID_maker_set
    __swig_getmethods__["maker"] = _pysci.PersistentTypeID_maker_get
    if _newclass:maker = property(_pysci.PersistentTypeID_maker_get, _pysci.PersistentTypeID_maker_set)
    def __init__(self, *args):
        _swig_setattr(self, PersistentTypeID, 'this', _pysci.new_PersistentTypeID(*args))
        _swig_setattr(self, PersistentTypeID, 'thisown', 1)
    def __del__(self, destroy=_pysci.delete_PersistentTypeID):
        try:
            if self.thisown: destroy(self)
        except: pass

    __swig_setmethods__["bc_maker1"] = _pysci.PersistentTypeID_bc_maker1_set
    __swig_getmethods__["bc_maker1"] = _pysci.PersistentTypeID_bc_maker1_get
    if _newclass:bc_maker1 = property(_pysci.PersistentTypeID_bc_maker1_get, _pysci.PersistentTypeID_bc_maker1_set)
    __swig_setmethods__["bc_maker2"] = _pysci.PersistentTypeID_bc_maker2_set
    __swig_getmethods__["bc_maker2"] = _pysci.PersistentTypeID_bc_maker2_get
    if _newclass:bc_maker2 = property(_pysci.PersistentTypeID_bc_maker2_get, _pysci.PersistentTypeID_bc_maker2_set)

class PersistentTypeIDPtr(PersistentTypeID):
    def __init__(self, this):
        _swig_setattr(self, PersistentTypeID, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, PersistentTypeID, 'thisown', 0)
        self.__class__ = PersistentTypeID
_pysci.PersistentTypeID_swigregister(PersistentTypeIDPtr)

class Piostream(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Piostream, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Piostream, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::Piostream instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    Read = _pysci.Piostream_Read
    Write = _pysci.Piostream_Write
    Big = _pysci.Piostream_Big
    Little = _pysci.Piostream_Little
    PERSISTENT_VERSION = _pysci.cvar.Piostream_PERSISTENT_VERSION
    def flag_error(*args): return _pysci.Piostream_flag_error(*args)
    __swig_setmethods__["file_name"] = _pysci.Piostream_file_name_set
    __swig_getmethods__["file_name"] = _pysci.Piostream_file_name_get
    if _newclass:file_name = property(_pysci.Piostream_file_name_get, _pysci.Piostream_file_name_set)
    def __del__(self, destroy=_pysci.delete_Piostream):
        try:
            if self.thisown: destroy(self)
        except: pass

    def peek_class(*args): return _pysci.Piostream_peek_class(*args)
    def begin_class(*args): return _pysci.Piostream_begin_class(*args)
    def end_class(*args): return _pysci.Piostream_end_class(*args)
    def begin_cheap_delim(*args): return _pysci.Piostream_begin_cheap_delim(*args)
    def end_cheap_delim(*args): return _pysci.Piostream_end_cheap_delim(*args)
    def eof(*args): return _pysci.Piostream_eof(*args)
    def io(*args): return _pysci.Piostream_io(*args)
    def reading(*args): return _pysci.Piostream_reading(*args)
    def writing(*args): return _pysci.Piostream_writing(*args)
    def error(*args): return _pysci.Piostream_error(*args)
    def version(*args): return _pysci.Piostream_version(*args)
    def backwards_compat_id(*args): return _pysci.Piostream_backwards_compat_id(*args)
    def set_backwards_compat_id(*args): return _pysci.Piostream_set_backwards_compat_id(*args)
    def supports_block_io(*args): return _pysci.Piostream_supports_block_io(*args)
    def block_io(*args): return _pysci.Piostream_block_io(*args)
    def disable_pointer_hashing(*args): return _pysci.Piostream_disable_pointer_hashing(*args)

class PiostreamPtr(Piostream):
    def __init__(self, this):
        _swig_setattr(self, Piostream, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Piostream, 'thisown', 0)
        self.__class__ = Piostream
_pysci.Piostream_swigregister(PiostreamPtr)
cvar = _pysci.cvar

class Persistent(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Persistent, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Persistent, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::Persistent instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __del__(self, destroy=_pysci.delete_Persistent):
        try:
            if self.thisown: destroy(self)
        except: pass

    def io(*args): return _pysci.Persistent_io(*args)

class PersistentPtr(Persistent):
    def __init__(self, this):
        _swig_setattr(self, Persistent, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Persistent, 'thisown', 0)
        self.__class__ = Persistent
_pysci.Persistent_swigregister(PersistentPtr)

auto_istream = _pysci.auto_istream

auto_ostream = _pysci.auto_ostream

SCI_project_Datatype_h = _pysci.SCI_project_Datatype_h
class Datatype(Persistent):
    __swig_setmethods__ = {}
    for _s in [Persistent]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, Datatype, name, value)
    __swig_getmethods__ = {}
    for _s in [Persistent]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, Datatype, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::Datatype instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    __swig_setmethods__["ref_cnt"] = _pysci.Datatype_ref_cnt_set
    __swig_getmethods__["ref_cnt"] = _pysci.Datatype_ref_cnt_get
    if _newclass:ref_cnt = property(_pysci.Datatype_ref_cnt_get, _pysci.Datatype_ref_cnt_set)
    __swig_getmethods__["lock"] = _pysci.Datatype_lock_get
    if _newclass:lock = property(_pysci.Datatype_lock_get)
    __swig_setmethods__["generation"] = _pysci.Datatype_generation_set
    __swig_getmethods__["generation"] = _pysci.Datatype_generation_get
    if _newclass:generation = property(_pysci.Datatype_generation_get, _pysci.Datatype_generation_set)
    def __del__(self, destroy=_pysci.delete_Datatype):
        try:
            if self.thisown: destroy(self)
        except: pass

    __swig_getmethods__["compute_new_generation"] = lambda x: _pysci.Datatype_compute_new_generation
    if _newclass:compute_new_generation = staticmethod(_pysci.Datatype_compute_new_generation)

class DatatypePtr(Datatype):
    def __init__(self, this):
        _swig_setattr(self, Datatype, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, Datatype, 'thisown', 0)
        self.__class__ = Datatype
_pysci.Datatype_swigregister(DatatypePtr)

Pio = _pysci.Pio

Datatype_compute_new_generation = _pysci.Datatype_compute_new_generation

class BaseEvent(Datatype):
    __swig_setmethods__ = {}
    for _s in [Datatype]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, BaseEvent, name, value)
    __swig_getmethods__ = {}
    for _s in [Datatype]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, BaseEvent, name)
    def __init__(self): raise RuntimeError, "No constructor defined"
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::BaseEvent instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __del__(self, destroy=_pysci.delete_BaseEvent):
        try:
            if self.thisown: destroy(self)
        except: pass

    def clone(*args): return _pysci.BaseEvent_clone(*args)
    def io(*args): return _pysci.BaseEvent_io(*args)
    def get_time(*args): return _pysci.BaseEvent_get_time(*args)
    def get_target(*args): return _pysci.BaseEvent_get_target(*args)
    def set_time(*args): return _pysci.BaseEvent_set_time(*args)
    def set_target(*args): return _pysci.BaseEvent_set_target(*args)
    def is_pointer_event(*args): return _pysci.BaseEvent_is_pointer_event(*args)
    def is_key_event(*args): return _pysci.BaseEvent_is_key_event(*args)
    def is_window_event(*args): return _pysci.BaseEvent_is_window_event(*args)
    def is_scene_graph_event(*args): return _pysci.BaseEvent_is_scene_graph_event(*args)
    def is_tm_notify_event(*args): return _pysci.BaseEvent_is_tm_notify_event(*args)

class BaseEventPtr(BaseEvent):
    def __init__(self, this):
        _swig_setattr(self, BaseEvent, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, BaseEvent, 'thisown', 0)
        self.__class__ = BaseEvent
_pysci.BaseEvent_swigregister(BaseEventPtr)

class EventModifiers(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, EventModifiers, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, EventModifiers, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::EventModifiers instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, EventModifiers, 'this', _pysci.new_EventModifiers(*args))
        _swig_setattr(self, EventModifiers, 'thisown', 1)
    def __del__(self, destroy=_pysci.delete_EventModifiers):
        try:
            if self.thisown: destroy(self)
        except: pass

    def io(*args): return _pysci.EventModifiers_io(*args)
    SHIFT_E = _pysci.EventModifiers_SHIFT_E
    CAPS_LOCK_E = _pysci.EventModifiers_CAPS_LOCK_E
    CONTROL_E = _pysci.EventModifiers_CONTROL_E
    ALT_E = _pysci.EventModifiers_ALT_E
    M1_E = _pysci.EventModifiers_M1_E
    M2_E = _pysci.EventModifiers_M2_E
    M3_E = _pysci.EventModifiers_M3_E
    M4_E = _pysci.EventModifiers_M4_E
    def get_modifiers(*args): return _pysci.EventModifiers_get_modifiers(*args)
    def set_modifiers(*args): return _pysci.EventModifiers_set_modifiers(*args)

class EventModifiersPtr(EventModifiers):
    def __init__(self, this):
        _swig_setattr(self, EventModifiers, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, EventModifiers, 'thisown', 0)
        self.__class__ = EventModifiers
_pysci.EventModifiers_swigregister(EventModifiersPtr)

class PointerEvent(BaseEvent,EventModifiers):
    __swig_setmethods__ = {}
    for _s in [BaseEvent,EventModifiers]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, PointerEvent, name, value)
    __swig_getmethods__ = {}
    for _s in [BaseEvent,EventModifiers]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, PointerEvent, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::PointerEvent instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    MOTION_E = _pysci.PointerEvent_MOTION_E
    BUTTON_PRESS_E = _pysci.PointerEvent_BUTTON_PRESS_E
    BUTTON_RELEASE_E = _pysci.PointerEvent_BUTTON_RELEASE_E
    BUTTON_1_E = _pysci.PointerEvent_BUTTON_1_E
    BUTTON_2_E = _pysci.PointerEvent_BUTTON_2_E
    BUTTON_3_E = _pysci.PointerEvent_BUTTON_3_E
    BUTTON_4_E = _pysci.PointerEvent_BUTTON_4_E
    BUTTON_5_E = _pysci.PointerEvent_BUTTON_5_E
    def __init__(self, *args):
        _swig_setattr(self, PointerEvent, 'this', _pysci.new_PointerEvent(*args))
        _swig_setattr(self, PointerEvent, 'thisown', 1)
    def __del__(self, destroy=_pysci.delete_PointerEvent):
        try:
            if self.thisown: destroy(self)
        except: pass

    def io(*args): return _pysci.PointerEvent_io(*args)
    def clone(*args): return _pysci.PointerEvent_clone(*args)
    def is_pointer_event(*args): return _pysci.PointerEvent_is_pointer_event(*args)
    def get_pointer_state(*args): return _pysci.PointerEvent_get_pointer_state(*args)
    def get_x(*args): return _pysci.PointerEvent_get_x(*args)
    def get_y(*args): return _pysci.PointerEvent_get_y(*args)
    def set_pointer_state(*args): return _pysci.PointerEvent_set_pointer_state(*args)
    def set_x(*args): return _pysci.PointerEvent_set_x(*args)
    def set_y(*args): return _pysci.PointerEvent_set_y(*args)

class PointerEventPtr(PointerEvent):
    def __init__(self, this):
        _swig_setattr(self, PointerEvent, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, PointerEvent, 'thisown', 0)
        self.__class__ = PointerEvent
_pysci.PointerEvent_swigregister(PointerEventPtr)

class KeyEvent(BaseEvent,EventModifiers):
    __swig_setmethods__ = {}
    for _s in [BaseEvent,EventModifiers]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, KeyEvent, name, value)
    __swig_getmethods__ = {}
    for _s in [BaseEvent,EventModifiers]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, KeyEvent, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::KeyEvent instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    KEY_PRESS_E = _pysci.KeyEvent_KEY_PRESS_E
    KEY_RELEASE_E = _pysci.KeyEvent_KEY_RELEASE_E
    def __init__(self, *args):
        _swig_setattr(self, KeyEvent, 'this', _pysci.new_KeyEvent(*args))
        _swig_setattr(self, KeyEvent, 'thisown', 1)
    def __del__(self, destroy=_pysci.delete_KeyEvent):
        try:
            if self.thisown: destroy(self)
        except: pass

    def clone(*args): return _pysci.KeyEvent_clone(*args)
    def io(*args): return _pysci.KeyEvent_io(*args)
    def is_key_event(*args): return _pysci.KeyEvent_is_key_event(*args)
    def get_key_state(*args): return _pysci.KeyEvent_get_key_state(*args)
    def get_keyval(*args): return _pysci.KeyEvent_get_keyval(*args)
    def get_key_string(*args): return _pysci.KeyEvent_get_key_string(*args)
    def set_key_state(*args): return _pysci.KeyEvent_set_key_state(*args)
    def set_keyval(*args): return _pysci.KeyEvent_set_keyval(*args)
    def set_key_string(*args): return _pysci.KeyEvent_set_key_string(*args)

class KeyEventPtr(KeyEvent):
    def __init__(self, this):
        _swig_setattr(self, KeyEvent, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, KeyEvent, 'thisown', 0)
        self.__class__ = KeyEvent
_pysci.KeyEvent_swigregister(KeyEventPtr)

class WindowEvent(BaseEvent):
    __swig_setmethods__ = {}
    for _s in [BaseEvent]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, WindowEvent, name, value)
    __swig_getmethods__ = {}
    for _s in [BaseEvent]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, WindowEvent, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::WindowEvent instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    CREATE_E = _pysci.WindowEvent_CREATE_E
    DESTROY_E = _pysci.WindowEvent_DESTROY_E
    ENTER_E = _pysci.WindowEvent_ENTER_E
    LEAVE_E = _pysci.WindowEvent_LEAVE_E
    EXPOSE_E = _pysci.WindowEvent_EXPOSE_E
    CONFIGURE_E = _pysci.WindowEvent_CONFIGURE_E
    FOCUSIN_E = _pysci.WindowEvent_FOCUSIN_E
    FOCUSOUT_E = _pysci.WindowEvent_FOCUSOUT_E
    REDRAW_E = _pysci.WindowEvent_REDRAW_E
    def __init__(self, *args):
        _swig_setattr(self, WindowEvent, 'this', _pysci.new_WindowEvent(*args))
        _swig_setattr(self, WindowEvent, 'thisown', 1)
    def __del__(self, destroy=_pysci.delete_WindowEvent):
        try:
            if self.thisown: destroy(self)
        except: pass

    def clone(*args): return _pysci.WindowEvent_clone(*args)
    def io(*args): return _pysci.WindowEvent_io(*args)
    def is_window_event(*args): return _pysci.WindowEvent_is_window_event(*args)
    def get_window_state(*args): return _pysci.WindowEvent_get_window_state(*args)
    def set_window_state(*args): return _pysci.WindowEvent_set_window_state(*args)

class WindowEventPtr(WindowEvent):
    def __init__(self, this):
        _swig_setattr(self, WindowEvent, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, WindowEvent, 'thisown', 0)
        self.__class__ = WindowEvent
_pysci.WindowEvent_swigregister(WindowEventPtr)

class QuitEvent(BaseEvent):
    __swig_setmethods__ = {}
    for _s in [BaseEvent]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, QuitEvent, name, value)
    __swig_getmethods__ = {}
    for _s in [BaseEvent]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, QuitEvent, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::QuitEvent instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, QuitEvent, 'this', _pysci.new_QuitEvent(*args))
        _swig_setattr(self, QuitEvent, 'thisown', 1)
    def __del__(self, destroy=_pysci.delete_QuitEvent):
        try:
            if self.thisown: destroy(self)
        except: pass

    def io(*args): return _pysci.QuitEvent_io(*args)
    def clone(*args): return _pysci.QuitEvent_clone(*args)

class QuitEventPtr(QuitEvent):
    def __init__(self, this):
        _swig_setattr(self, QuitEvent, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, QuitEvent, 'thisown', 0)
        self.__class__ = QuitEvent
_pysci.QuitEvent_swigregister(QuitEventPtr)

class RedrawEvent(BaseEvent):
    __swig_setmethods__ = {}
    for _s in [BaseEvent]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, RedrawEvent, name, value)
    __swig_getmethods__ = {}
    for _s in [BaseEvent]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, RedrawEvent, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::RedrawEvent instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, RedrawEvent, 'this', _pysci.new_RedrawEvent(*args))
        _swig_setattr(self, RedrawEvent, 'thisown', 1)
    def __del__(self, destroy=_pysci.delete_RedrawEvent):
        try:
            if self.thisown: destroy(self)
        except: pass

    def io(*args): return _pysci.RedrawEvent_io(*args)
    def clone(*args): return _pysci.RedrawEvent_clone(*args)

class RedrawEventPtr(RedrawEvent):
    def __init__(self, this):
        _swig_setattr(self, RedrawEvent, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, RedrawEvent, 'thisown', 0)
        self.__class__ = RedrawEvent
_pysci.RedrawEvent_swigregister(RedrawEventPtr)

class TMNotifyEvent(BaseEvent):
    __swig_setmethods__ = {}
    for _s in [BaseEvent]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, TMNotifyEvent, name, value)
    __swig_getmethods__ = {}
    for _s in [BaseEvent]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, TMNotifyEvent, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ SCIRun::TMNotifyEvent instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    START_E = _pysci.TMNotifyEvent_START_E
    STOP_E = _pysci.TMNotifyEvent_STOP_E
    SUSPEND_E = _pysci.TMNotifyEvent_SUSPEND_E
    RESUME_E = _pysci.TMNotifyEvent_RESUME_E
    def __init__(self, *args):
        _swig_setattr(self, TMNotifyEvent, 'this', _pysci.new_TMNotifyEvent(*args))
        _swig_setattr(self, TMNotifyEvent, 'thisown', 1)
    def __del__(self, destroy=_pysci.delete_TMNotifyEvent):
        try:
            if self.thisown: destroy(self)
        except: pass

    def io(*args): return _pysci.TMNotifyEvent_io(*args)
    def clone(*args): return _pysci.TMNotifyEvent_clone(*args)
    def is_tm_notify_event(*args): return _pysci.TMNotifyEvent_is_tm_notify_event(*args)
    def get_tool_id(*args): return _pysci.TMNotifyEvent_get_tool_id(*args)
    def get_notify_state(*args): return _pysci.TMNotifyEvent_get_notify_state(*args)
    def set_tool_id(*args): return _pysci.TMNotifyEvent_set_tool_id(*args)
    def set_notify_state(*args): return _pysci.TMNotifyEvent_set_notify_state(*args)

class TMNotifyEventPtr(TMNotifyEvent):
    def __init__(self, this):
        _swig_setattr(self, TMNotifyEvent, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, TMNotifyEvent, 'thisown', 0)
        self.__class__ = TMNotifyEvent
_pysci.TMNotifyEvent_swigregister(TMNotifyEventPtr)

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
        self.__class__ = vector_string
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
        self.__class__ = vector_double
_pysci.vector_double_swigregister(vector_doublePtr)


load_field = _pysci.load_field

show_field = _pysci.show_field

init_pysci = _pysci.init_pysci

terminate = _pysci.terminate

test_function = _pysci.test_function

tetgen_2surf = _pysci.tetgen_2surf

run_viewer_thread = _pysci.run_viewer_thread

add_pointer_event = _pysci.add_pointer_event

add_key_event = _pysci.add_key_event

add_tm_notify_event = _pysci.add_tm_notify_event

selection_target_changed = _pysci.selection_target_changed


