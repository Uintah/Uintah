
#include <Core/CCA/tools/strauss/c++ruby/rubyeval.h>

#include <sstream>
#include <ruby.h>

using namespace std;


RubyEval* RubyEval::_instance = 0;

/** View for Interpreter output */
static ostream& curout = cout;

EXTERN VALUE ruby_errinfo;
EXTERN VALUE rb_defout;

/** Called by Ruby for writing to STDOUT */
static VALUE defout_write(VALUE self, VALUE str)
{
  curout << RubyEval::val2str(str);
  return Qnil;
}

RubyEval* RubyEval::instance() {
  if (!_instance) {
    _instance = new RubyEval();
  }
  return _instance;
}

RubyEval::RubyEval(){
  ruby_init();
  ruby_init_loadpath();
  rb_set_safe_level(0);
  ruby_script("ruby");

  rb_defout = rb_str_new("", 0);
  rb_define_singleton_method(rb_defout, "write", (VALUE(*)(...))defout_write, 1);
}
RubyEval::~RubyEval(){
}
/** Convert Ruby value to string */
string RubyEval::val2str(const VALUE rval){
  return STR2CSTR(rb_funcall(rval, rb_intern("to_s"), 0));
}
/** Convert Ruby string value to string */
string RubyEval::strval2str(const VALUE rval){
  return string(RSTRING(rval)->ptr, RSTRING(rval)->len);
}
/** Run Ruby interpreter with @p filename */
void RubyEval::run_file(const char* filename, ostream& out){
  //ruby_options(int argc, char **argv")
  rb_load_protect(rb_str_new2(filename), 0, &status);
  if (status) {
    exception_print(out);
  }
}

/** Get Ruby error/exception info an print it */
void RubyEval::exception_print(ostream& errout){
  //Adapted from eruby_main by Shugo Maeda <shugo@modruby.net>
  VALUE errat;
  VALUE eclass;
  VALUE einfo;

  if (NIL_P(ruby_errinfo)) return;

  errat = rb_funcall(ruby_errinfo, rb_intern("backtrace"), 0);
  if (!NIL_P(errat)) {
    VALUE mesg = RARRAY(errat)->ptr[0];

    if (NIL_P(mesg)) {
      ID last_func = rb_frame_last_func();
      errout << rb_id2name(last_func);
    } else {
      errout << strval2str(mesg);
    }
  }

  eclass = CLASS_OF(ruby_errinfo);
  einfo = rb_obj_as_string(ruby_errinfo);
  if (eclass == rb_eRuntimeError && RSTRING(einfo)->len == 0) {
    errout << ": unhandled exception\n";
  } else {
    VALUE epath;

    epath = rb_class_path(eclass);
    if (RSTRING(einfo)->len == 0) {
      errout << ": ";
      errout << strval2str(epath);
      errout << "\n";
    } else {
      char *tail  = 0;
      int len = RSTRING(einfo)->len;

      if (RSTRING(epath)->ptr[0] == '#') epath = 0;
      if ((tail = strchr(RSTRING(einfo)->ptr, '\n')) != NULL) {
        len = tail - RSTRING(einfo)->ptr;
        tail++;   /* skip newline */
      }
      errout << ": ";
      errout << string(RSTRING(einfo)->ptr, len);
      if (epath) {
        errout << " (";
        errout << strval2str(epath);
        errout << ")\n";
      }
      if (tail) {
        errout << string(tail, RSTRING(einfo)->len - len - 1);
        errout << "\n";
      }
    }
  }

  if (!NIL_P(errat)) {
    int i;
    struct RArray *ep = RARRAY(errat);

#define TRACE_MAX (TRACE_HEAD+TRACE_TAIL+5)
#define TRACE_HEAD 8
#define TRACE_TAIL 5

    rb_ary_pop(errat);
    ep = RARRAY(errat);
    for (i=1; i<ep->len; i++) {
      if (TYPE(ep->ptr[i]) == T_STRING) {
        errout << "        from ";
        errout << strval2str(ep->ptr[i]);
        errout << "\n";
      }
      if (i == TRACE_HEAD && ep->len > TRACE_MAX) {
        errout << "         ... " << (ep->len - TRACE_HEAD - TRACE_TAIL)
               << " levels...\n";
        i = ep->len - TRACE_TAIL;
      }
    }
  }
  errout.flush();
}
/** Evaluate code string and print erros & results */
VALUE RubyEval::eval(const char* code){
  return rb_eval_string_protect(code, &status);
}
/** Evaluate code string and print erros & results */
VALUE RubyEval::eval(const char* code, ostream& errout){
  VALUE ret = rb_eval_string_protect(code, &status);
  if (status) {
    exception_print(errout);
  }
  return ret;
}
/** Last evaluation was successful */
bool RubyEval::evalOk(){
  return status == 0;
}
/** Get exception message as string */
string RubyEval::exception_info(){
  stringstream errstr;
  exception_print(errstr);
  errstr << ends;
  return errstr.str();
}
