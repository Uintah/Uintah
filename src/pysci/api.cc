//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : api.cc
//    Author : Martin Cole
//    Date   : Mon Aug  7 15:21:20 2006


#include <pysci/api.h>
#include <Core/Init/init.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Util/Environment.h>
#include <Core/Geom/CallbackOpenGLContext.h>
#include <Core/Events/DataManager.h>
#include <Core/Events/EventManager.h>
#include <Core/Events/BaseEvent.h>
#include <Core/Events/SceneGraphEvent.h>
#include <Core/Events/SelectionTargetEvent.h>
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
#include <Core/Algorithms/Visualization/RenderField.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

using std::cerr;
using std::endl;

EventManager *em = 0;
DataManager *dm = 0;


void init_pysci(char**environment)
{
  static bool init_done = false;
  if (!init_done) {
    create_sci_environment(environment, 0);
    sci_putenv("SCIRUN_VERSION", SCIRUN_VERSION);
    sci_putenv("SCIRUN_RCFILE_SUBVERSION", SCIRUN_RCFILE_SUBVERSION);

    SCIRunInit();

    //! create the EventManager thread.
    if (! em) {
      em = new EventManager();
      Thread *emt = scinew Thread(em, "Event Manager Thread");
      emt->detach();
    }

    init_done = true;
    cerr << "otf:" << sci_getenv("SCIRUN_ON_THE_FLY_LIBS_DIR") << endl;

    if (! dm) {
      dm = new DataManager();
      Thread *dmt = scinew Thread(dm, "Data Manager Thread");
      dmt->detach();
    }
    
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
  unsigned int f1_id = dm->load_field(f1);
  if (f1_id == 0) {
    cerr << "Error opening file: "<< f1 << endl;
    return false;
  }
  unsigned int f2_id = dm->load_field(f2);
  if (f2_id == 0) {
    cerr << "Error opening file: "<< f2 << endl;
    return false;
  }

  FieldHandle outer = dm->get_field(f1_id);
  FieldHandle inner = dm->get_field(f2_id);

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


void add_pointer_event(PointerEvent *pe)
{
  PointerEvent *p = new PointerEvent();
  p->ref_cnt = 1; // workaround for assert in Datatype operator ==
  *p = *pe;
  event_handle_t event = p;
  EventManager::add_event(event);
}

void add_key_event(KeyEvent *ke)
{
  KeyEvent *k = new KeyEvent();
  k->ref_cnt = 1; // workaround for assert in Datatype operator ==
  *k = *ke;
  event_handle_t event = k;
  EventManager::add_event(event);
}

void add_tm_notify_event(TMNotifyEvent *te)
{
  TMNotifyEvent *t = new TMNotifyEvent();
  t->ref_cnt = 1; // workaround for assert in Datatype operator ==
  *t = *te;
  cerr << "--------" << t->get_tool_mode() << endl;

  event_handle_t event = t;
  EventManager::add_event(event);
}

void add_command_event(CommandEvent *ce)
{
  CommandEvent *c = new CommandEvent();
  c->ref_cnt = 1; // workaround for assert in Datatype operator ==
  *c = *ce;
  event_handle_t event = c;
  EventManager::add_event(event);
}

void selection_target_changed(unsigned int fid)
{
  dm->selection_target_changed(fid);
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

unsigned int load_field(string fname)
{
  return dm->load_field(fname);
}

bool show_field(unsigned int fld_id) 
{
  return dm->show_field(fld_id);  
}





}
