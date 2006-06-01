#include <pysci/api.h>
#include <Core/Init/init.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Util/Environment.h>
#include <Core/Geom/CallbackOpenGLContext.h>
#include <Core/Events/EventManager.h>
#include <Core/Events/BaseEvent.h>
#include <Core/Events/OpenGLViewer.h>
#include <Core/Datatypes/Field.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <main/sci_version.h>
#include <Dataflow/Modules/Fields/TetGen.h>
#include <iostream>

namespace SCIRun {

using std::cerr;
using std::endl;

EventManager *em = 0;

void init_pysci(char**environment)
{
  static bool init_done = false;
  if (!init_done) {
    create_sci_environment(environment, 0);
    sci_putenv("SCIRUN_VERSION", SCIRUN_VERSION);
    sci_putenv("SCIRUN_RCFILE_SUBVERSION", SCIRUN_RCFILE_SUBVERSION);

    // to solve the mysterious unresolved when loading dynamic compiled so's
    //_ZTIN6SCIRun5FieldE
    //typeinfo for SCIRun::Field
    SCIRunInit();

    //! create the EventManager thread.
    if (! em) {
      em = new EventManager();
      Thread *emt = scinew Thread(em, "Event Manager Thread");
      emt->detach();
    }

    init_done = true;
    cerr << "otf:" << sci_getenv("SCIRUN_ON_THE_FLY_LIBS_DIR") << endl;
  }
}

void test_function(string f1, string f2, string f3)
{
  cerr << "f1: " << f1 << endl;
  cerr << "f2: " << f2 << endl;
  cerr << "f3: " << f3 << endl;

}

bool 
tetgen_2surf(string f1, string f2, string outfile)
{
  FieldHandle inner, outer;
  Piostream* stream = auto_istream(f1);
  if (!stream) {
    cerr << "Couldn't open file: "<< f1 << endl;
    return false;
  }
  Pio(*stream, outer);

  stream = auto_istream(f2);
  if (!stream) {
    cerr << "Couldn't open file: "<< f2 << endl;
    return false;
  }
  Pio(*stream, inner);

  cerr << "mesh scalar?: " << inner->is_scalar() << endl;
  tetgenio in, out;

  unsigned idx = 0;
  unsigned fidx = 0;

  // indices start from 0.
  in.firstnumber = 0;
  in.mesh_dim = 3;
  ProgressReporter rep;
  
  //! Add the info for the outer surface, or tetvol to be refined.
  const TypeDescription *ostd = outer->get_type_description();
  const string tcn("TGSurfaceTGIO");
  CompileInfoHandle ci = TGSurfaceTGIOAlgo::get_compile_info(ostd, tcn);
  //Handle<TGSurfaceTGIOAlgo> algo;
  DynamicAlgoHandle dalgo;
  if (!DynamicCompilation::compile(ci, dalgo)) {
    //error("Could not get TetGen/SCIRun converter algorithm");
    cerr << "Could not get TetGen/SCIRun converter algorithm" << endl;
    return false;
  }
  int marker = -10;
  TGSurfaceTGIOAlgo* algo = (TGSurfaceTGIOAlgo*)dalgo.get_rep();
  //TGSurfaceTGIOAlgo* algo = dynamic_cast<TGSurfaceTGIOAlgo*>(dalgo.get_rep());
  if (!algo) {
    cerr << "WTF!" << endl;
    return false;
  }
  algo->to_tetgenio(&rep, outer, idx, fidx, marker, in);


  // Interior surface.
  marker *= 2;
  algo->to_tetgenio(&rep, inner, idx, fidx, marker, in);


  //update_progress(.2);
  // Save files for later debugging.
  in.save_nodes("/tmp/tgIN");
  in.save_poly("/tmp/tgIN");

  string tg_cmmd = "pqa300.0AAz";
  // Create the new mesh.
  tetrahedralize((char*)tg_cmmd.c_str(), &in, &out); 
  FieldHandle tetvol_out;
  //update_progress(.9);
  // Convert to a SCIRun TetVol.
  tetvol_out = algo->to_tetvol(out);
  //update_progress(1.0);
  if (tetvol_out.get_rep() == 0) { return false; }
  BinaryPiostream out_stream(outfile, Piostream::Write);
  Pio(out_stream, tetvol_out);
  return true;
}


void add_key_event(unsigned time, unsigned keval, 
		   string str, int keycode) 
{
  KeyEvent *k = new KeyEvent();
  k->set_key_state(KeyEvent::KEY_PRESS_E);
  k->set_time(time);
  event_handle_t event = k;
  EventManager::add_event(event);
}

void terminate() 
{
  // send a NULL event to terminate...
  event_handle_t event = new QuitEvent();
  EventManager::add_event(event);
}

void run_viewer_thread(CallbackOpenGLContext *ogl) 
{
  CallbackOpenGLContext *c = new CallbackOpenGLContext();
  *c = *ogl;
  cerr << "starting viewer thread with: " << c << endl;
  
  OpenGLViewer *v = new OpenGLViewer(c);
  Thread *vt = scinew Thread(v, "Viewer Thread");
  vt->detach(); // runs until thread exits.
}

}
