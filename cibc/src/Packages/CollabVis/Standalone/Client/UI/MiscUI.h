#ifndef __misc_ui_h_
#define __misc_ui_h_

#include <tcl.h>
#include <string>

using namespace std;

namespace SemotusVisum {

class GuiArgs;
class SVCallback;

extern void redraw();
extern void draw();
extern char * ccast_unsafe(const string &str);
extern void execute(const string& str);
extern int eval(const string& str, string& result);
extern void source_once(const string& filename);
extern int do_command(ClientData cd, Tcl_Interp*, int argc, char* argv[]);
extern void add_command(const string&command, SVCallback* callback,
			void* userdata);
extern bool get(const std::string& name, std::string& value); 
extern void set(const std::string& name, const std::string& value);

}

typedef void (Tcl_LockProc)();
// Located in tclUnixNotify
extern "C" void Tcl_SetLock(Tcl_LockProc* lock, Tcl_LockProc* unlock);

#endif
