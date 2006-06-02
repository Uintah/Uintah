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
#include <Core/Algorithms/Visualization/RenderField.h>
#include <iostream>

namespace SCIRun {

using std::cerr;
using std::endl;

EventManager *em = 0;


map<int, FieldHandle> fields_;

int load_field(string fname)
{
  static int cur_field_num = 0;
  FieldHandle fld;
  Piostream* stream = auto_istream(fname);
  if (!stream) {
    cerr << "Couldn't open file: "<< fname << endl;
    return -1;
  }
  Pio(*stream, fld);

  fields_[cur_field_num++] = fld;
  return cur_field_num - 1;
}

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
  int f1_id = load_field(f1);
  if (f1_id == -1) {
    cerr << "Error opening file: "<< f1 << endl;
    return false;
  }
  int f2_id = load_field(f2);
  if (f2_id == -1) {
    cerr << "Error opening file: "<< f2 << endl;
    return false;
  }

  FieldHandle outer = fields_[f1_id];
  FieldHandle inner = fields_[f2_id];

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


void add_pointer_up_event(unsigned time, int x, int y, int which) 
{
  PointerEvent *p = new PointerEvent();
  unsigned s = PointerEvent::BUTTON_RELEASE_E;
  switch (which) {
  case 1:
    s |= PointerEvent::BUTTON_1_E;
    break;
  case 2:
    s |= PointerEvent::BUTTON_2_E;
    break;
  case 3:
    s |= PointerEvent::BUTTON_3_E;
    break;
  case 4:
    s |= PointerEvent::BUTTON_4_E;
    break;
  case 5:
    s |= PointerEvent::BUTTON_5_E;
    break;
  }
  p->set_pointer_state(s);
  p->set_x(x);
  p->set_y(y);
  p->set_time(time);
  event_handle_t event = p;
  EventManager::add_event(event);
}


void add_pointer_down_event(unsigned time, int x, int y, int which) 
{
  PointerEvent *p = new PointerEvent();
  unsigned s = PointerEvent::BUTTON_PRESS_E;
  switch (which) {
  case 1:
    s |= PointerEvent::BUTTON_1_E;
    break;
  case 2:
    s |= PointerEvent::BUTTON_2_E;
    break;
  case 3:
    s |= PointerEvent::BUTTON_3_E;
    break;
  case 4:
    s |= PointerEvent::BUTTON_4_E;
    break;
  case 5:
    s |= PointerEvent::BUTTON_5_E;
    break;
  }
  p->set_pointer_state(s);
  p->set_x(x);
  p->set_y(y);
  p->set_time(time);
  event_handle_t event = p;
  EventManager::add_event(event);
}

void add_motion_notify_event(unsigned time, int x, int y) 
{
  PointerEvent *p = new PointerEvent();
  unsigned s = PointerEvent::MOTION_E;
  p->set_pointer_state(s);
  p->set_x(x);
  p->set_y(y);
  p->set_time(time);
  event_handle_t event = p;
  EventManager::add_event(event);
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

bool show_field(int fld_id) 
{
  FieldHandle fld_handle = fields_[fld_id];
  // input parameters to RenderField::render
  // fld_handle - the field to render
  bool do_nodes = true; // show nodes or not.
  bool do_edges = true; // show edges or not.
  bool do_faces = true; // show faces or not.

//   ColorMap(const vector<Color>& rgb,
// 	   const vector<float>& rgbT,
// 	   const vector<float>& alphas,
// 	   const vector<float>& alphaT,
// 	   unsigned int resolution = 256);
  ColorMapHandle color_map = 0; // mapping data values to colors 
  //color when no color_map.
  MaterialHandle def_material = new Material(Color(0.3, 0.7, 0.3)); 
  string ndt = "Points"; // or "Spheres" or "Axes"  
  string edt = "Lines"; // or "Cylinders" 
  double ns = 0.3; // node scale factor. 
  double es = 0.3; // edge scale factor.
  double vscale = 0.3; // Vectors scale factor. 
    bool normalize_vectors  = true; // normalize vectors before rendering?
  // gluSphere resolution when rendering spheres for nodes.
  int node_resolution  = 6;
  // gluCylinder resolution when rendering spheres for nodes.
  int edge_resolution = 6;
  bool faces_normals = true; //use face normals (gouraud shading, not flat)
  bool nodes_transparency = false; // render transparent nodes?
  bool edges_transparency = false; // render transparent edges?
  bool faces_transparency = false; // render transparent faces?
  bool nodes_usedefcolor = true; // always use default color for nodes?
  bool edges_usedefcolor = true; // always use default color for edges?
  bool faces_usedefcolor = true; // always use default color for faces?
  int approx_div = 1; // divisions per simplex (high order field rendering)
  bool faces_usetexture = false; //use face texture rendering?


  const TypeDescription *ftd = fld_handle->get_type_description();
  const TypeDescription *ltd = fld_handle->order_type_description();
  // description for just the data in the field

  // Get the Algorithm.
  CompileInfoHandle ci = RenderFieldBase::get_compile_info(ftd, ltd);

  DynamicAlgoHandle dalgo;
  if (!DynamicCompilation::compile(ci, dalgo)) {
    return false;
  }
  
  RenderFieldBase *renderer = (RenderFieldBase*)dalgo.get_rep();
  if (!renderer) {
    cerr << "WTF!" << endl;
    return false;
  }

  if (faces_normals) fld_handle->mesh()->synchronize(Mesh::NORMALS_E);
  renderer->render(fld_handle,
		   do_nodes, do_edges, do_faces,
		   color_map, def_material,
		   ndt, edt, ns, es, vscale, normalize_vectors,
		   node_resolution, edge_resolution,
		   faces_normals,
		   nodes_transparency,
		   edges_transparency,
		   faces_transparency,
		   nodes_usedefcolor,
		   edges_usedefcolor,
		   faces_usedefcolor,
		   approx_div,
		   faces_usetexture);
  

  string fname = "myfield";
  if (do_nodes) 
  {
    GeomHandle gmat = scinew GeomMaterial(renderer->node_switch_, 
					  def_material);
    GeomHandle geom = scinew GeomSwitch(scinew GeomColorMap(gmat, color_map));
    const char *name = nodes_transparency?"Transparent Nodes":"Nodes";
    SceneGraphEvent* sge = new SceneGraphEvent(geom, fname + name);
    event_handle_t event = sge;
    EventManager::add_event(event);
    //node_id = ogeom_->addObj(geom, fname + name);
  }

  if (do_edges) 
  { 
    GeomHandle gmat = scinew GeomMaterial(renderer->edge_switch_, 
					  def_material);
    GeomHandle geom = scinew GeomSwitch(scinew GeomColorMap(gmat, color_map));
    const char *name = edges_transparency?"Transparent Edges":"Edges";
    SceneGraphEvent* sge = new SceneGraphEvent(geom, fname + name);
    event_handle_t event = sge;
    EventManager::add_event(event);
    //edge_id = ogeom_->addObj(geom, fname + name);
  }

  if (do_faces)
  {
    GeomHandle gmat = scinew GeomMaterial(renderer->face_switch_, 
					  def_material);
    GeomHandle geom = scinew GeomSwitch(scinew GeomColorMap(gmat, color_map));
    const char *name = faces_transparency?"Transparent Faces":"Faces";
    SceneGraphEvent* sge = new SceneGraphEvent(geom, fname + name);
    event_handle_t event = sge;
    EventManager::add_event(event);

    //face_id = ogeom_->addObj(geom, fname + name);
  }
  return true;
  
}

}
