// swig -o pysci_wrap.cc -DSCISHARE -python -c++ pysci.i


#include <string>

namespace SCIRun {

class CallbackOpenGLContext;
using std::string;

int load_field(string fname);
bool show_field(int fld_id);
void init_pysci(char**environment);
void terminate();
void test_function(string f1, string f2, string f3);
bool tetgen_2surf(string f1, string f2, string out);

void run_viewer_thread(CallbackOpenGLContext *ogl);

void add_key_event(unsigned time, unsigned keval, 
		   string str, int keycode);
void add_motion_notify_event(unsigned time, int x, int y);
void add_pointer_down_event(unsigned time, int x, int y, int which);
void add_pointer_up_event(unsigned time, int x, int y, int which);


}
