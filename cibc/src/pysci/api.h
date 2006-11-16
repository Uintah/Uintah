// swig -v -o pysci_wrap.cc -DSCISHARE -python -c++ pysci.i


#include <string>

namespace SCIRun {

class CallbackOpenGLContext;
class KeyEvent;
class PointerEvent;
class TMNotifyEvent;
class CommandEvent;

using std::string;

unsigned int load_field(string fname);
bool show_field(unsigned int fld_id);
bool toggle_field_visibility(unsigned int fld_id);
void init_pysci(char**environment);
void terminate();
void test_function(string f1, string f2, string f3);
bool tetgen_2surf(string f1, string f2, string out);

void run_viewer_thread(CallbackOpenGLContext *ogl);
void add_pointer_event(PointerEvent *p);
void add_key_event(KeyEvent *k);
void add_tm_notify_event(TMNotifyEvent *t);
void add_command_event(CommandEvent *c);
void selection_target_changed(unsigned int fid);

}
