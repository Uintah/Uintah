/*
 *  ManipulateField.cc:
 *
 *  Written by:
 *   Martin Cole
 *   Mon Nov 20 09:35:23 MST 2000
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/share/share.h>
#include <Core/TclInterface/TCLvar.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Dataflow/Network/StrX.h>
#include <Core/Util/DebugStream.h>
#include <dlfcn.h>

namespace SCIRun {

#ifndef FM_COMP_PATH
#error main/sub.mk needs to define FM_COMP_PATH
#endif

const char* FIELD_MANIP_COMPILATION_PATH = FM_COMP_PATH;

// The function pointer definition.
typedef FieldHandle& (*fieldManipFunction)(vector<FieldHandle> &);

class PSECORESHARE ManipFields : public Module 
{
public:
  ManipFields(const clString& id);
  virtual ~ManipFields();

  // Compile and load .so for the selected manipulation
  virtual void execute();
  virtual void tcl_command(TCLArgs& args, void* userdata);

private:
  // ---- Compilation vars and xml specific. ----
  struct ManipData;
  typedef vector<clString> svec_t;
  typedef map<clString, ManipData*> manips_map_t;

  // place holder for each manip
  struct ManipData {
    clString  d_desc;
    svec_t   d_libs;
    svec_t   d_libpath;
    svec_t   d_inc;
  };

  // ---- Static Data ----
  static manips_map_t      d_manips;
  static bool              d_write_xml_needed;

  // Load the xml data once at first module creation.
  static void load_existing_manips();

  void change_md_data(svec_t &v, const clString &arg);

  // helpers to deal with the XML
  int processSrcNode(DOM_Node n);
  int processSrcsNode(DOM_Node n);
  int ReadNodeFromFile(const char* filename);
  void getManips();
  void load_ui();
  void set_cur_manip(const clString& name);
  void write_sub_mk();

  // ---- Compilation and dynamic loading specific. ----
  typedef void* sohandle_t;

  bool compileFile(const char *filename);
  fieldManipFunction getFunctionFromDL(const char *filename);

  // ---- Member Data ----
  FieldIPort*                d_infield;
  FieldHandle                d_sfield;
  FieldOPort*                d_ofield;
  
  TCLstring                  d_name;
  clString                   d_id;
  fieldManipFunction         d_curFun;
};

// Static initializers
ManipFields::manips_map_t ManipFields::d_manips;
bool ManipFields::d_write_xml_needed = false;


//////////
// ManipFields::ManipFields
// 
ManipFields::ManipFields(const clString& id) : 
  Module("ManipFields", id, Source),
  d_sfield(0),
  d_name("manipulationName", id, this),
  d_id(id),
  d_curFun(0)
{
  // Create the input ports
  d_infield = scinew FieldIPort(this, "Input Field", FieldIPort::Atomic);
  add_iport(d_infield);
    
  // Create the output port
  d_ofield = scinew FieldOPort(this, "Output Field", FieldIPort::Atomic);
  add_oport(d_ofield);
}

//////////
// ManipFields::~ManipFields
// 
ManipFields::~ManipFields() 
{}

//////////
// ManipFields::execute
// 
// Compile and load .so for the selected manipulation
void 
ManipFields::execute()
{
  // Load all of the available Manipulations from the xml file.
  static int once = 0;
  if (once == 0) {
    once++;
    getManips();
  }
  
  cerr << "try to load Manip " << endl;

  if (d_curFun != 0) {

    // Package up the input Fields.
    d_infield->get(d_sfield);
    vector<FieldHandle> allIn;
    allIn.push_back(d_sfield);

    FieldHandle fh = (*d_curFun)(allIn);
    d_ofield->send(fh);
  } else {
    cerr << "no fieldManipFunction loaded" << endl; 
  }
  cerr << "done Manip execute" << endl;
}

//////////
// ManipFields::tcl_command
// 
void 
ManipFields::tcl_command(TCLArgs& args, void* userdata)
{
  d_name.reset();
  cout << "TCLCOMMAND: curname: " << d_name.get() << endl;
  
  if(args.count() < 2){
    args.error("ManipFields needs a minor command");
    return;
  }

  ManipData &md = *d_manips[d_name.get()];

  if (args[1] == "compile") {

    if (d_write_xml_needed) {
      write_sub_mk();
      //write_xml();
      d_write_xml_needed = false;
    }

    cout << "C++ compile now!!" << endl;
    d_curFun = getFunctionFromDL(d_name.get()());
  } else if (args[1] == "launch-editor") {

    char ecmd[1024];
    sprintf(ecmd, "%s %s/%s.cc &", args[2](), 
	    FIELD_MANIP_COMPILATION_PATH ,d_name.get()());
    
    cout << "Launch Editor: " << ecmd << endl;

    if (system(ecmd) != 0) {
      cerr << "SCIRun::Modules::Fields::ManipFields " << endl
	   << "editor command failed" << endl;
    }
  } else if (args[1] == "libs") {
    svec_t &v = md.d_libs;
    change_md_data(v, args[2]);
  } else if (args[1] == "libpath") {
    svec_t &v = md.d_libpath;
    change_md_data(v, args[2]);
  } else if (args[1] == "inc") {
    svec_t &v = md.d_inc;
    change_md_data(v, args[2]);
  } else if (args[1] == "loadmanip") {
    set_cur_manip(args[2]);
    d_curFun = 0;
  } else {
    Module::tcl_command(args, userdata);
  }
}

void 
ManipFields::change_md_data(svec_t &v, const clString &arg) {
  d_write_xml_needed = true;

  if (arg == "clear") { 
    cout << "v clearing" << endl;
    v.clear();
    return;
  }
  cout << "v adding :" << arg << endl;
  v.push_back(clString(arg()));  
}

//////////
// ManipFields::getFunctionFromDL
// 
fieldManipFunction
ManipFields::getFunctionFromDL(const char *f)
{
  // Compile the file.
  if (compileFile(f)) {
    // Load the dl.
    char theso[1024];
    sprintf(theso, "%s/%s.so", FIELD_MANIP_COMPILATION_PATH, f);

    void *h = dlopen(theso, RTLD_NOW);
    if (h == 0) {
      cerr << "ManipFields::getFunctionFromDL() error:"
	   << " dlopen failed" << endl;
      return 0;
    }
    // return the function from the dl.
    fieldManipFunction f = (fieldManipFunction)dlsym(h, "testFieldManip");
    if (f == 0) {
      cerr << "ManipFields::getFunctionFromDL() error: "
	   << "dlsym failed" << endl;
      return 0;
    }
    return f;
  }
  return 0;
}

//////////
// ManipFields::compileFile
//
// Attempt to compile filename into a .so, return true if it succeeded
// false otherwise.

bool 
ManipFields::compileFile(const char *file)
{
  char command[1024];
  sprintf(command, "cd %s; gmake %s.so", FIELD_MANIP_COMPILATION_PATH, file);
  if (system(command) != 0) {
    cerr << "SCIRun::Modules::Fields::ManipFields::compileFile() error: "
	 << "system call failed:\n" << command << endl;
    return false;
  }
  return true;
}

inline 
clString 
convert_to_clString(DOMString s) {  
  return clString(s.transcode());
}

int 
ManipFields::processSrcNode(DOM_Node n) {

  ManipData *md = 0;
  DOM_NamedNodeMap att = n.getAttributes();
  if (att != 0) {
    DOM_Node d = att.getNamedItem("name");
    cout << "name: " << d.getNodeValue() << endl;
    md = new ManipData;
    d_manips[convert_to_clString(d.getNodeValue())] = md;
  }

  for (DOM_Node child = n.getFirstChild(); child != 0;
       child = child.getNextSibling()) {
    
    DOMString childname = child.getNodeName();
    if (childname.equals("description")) {
      cout << "description: " 
	   << removeWhiteSpace(getSerializedChildren(child)) << endl;
      md->d_desc = clString(removeWhiteSpace(getSerializedChildren(child)));
    }
    if (childname.equals("lib")) {
      DOM_Node d = child.getAttributes().getNamedItem("name");
      cout << "lib: " << d.getNodeValue() << endl;

      md->d_libs.push_back(convert_to_clString(d.getNodeValue()));
    }
    if (childname.equals("libpath")) {
      DOM_Node d = child.getAttributes().getNamedItem("name");
      cout << "libpath : " << d.getNodeValue() << endl;
      md->d_libpath.push_back(convert_to_clString(d.getNodeValue()));
    }
    
    if (childname.equals("inc")) {
      DOM_Node d = child.getAttributes().getNamedItem("name");
      cout << "include: " << d.getNodeValue() << endl;
      md->d_inc.push_back(convert_to_clString(d.getNodeValue()));
    }
  }
  return 1;
}

int 
ManipFields::processSrcsNode(DOM_Node n) {

  for (DOM_Node child = n.getFirstChild(); child != 0;
       child = child.getNextSibling()) {
    DOMString childname = child.getNodeName();
    if (childname.equals("src"))
      processSrcNode(child);
  }
  return 1;
}

int 
ManipFields::ReadNodeFromFile(const char* filename)
{
  // Initialize the XML4C system
  try {
    XMLPlatformUtils::Initialize();
  } catch (const XMLException& toCatch) {
    std::cerr << "Error during initialization! :\n"
	 << StrX(toCatch.getMessage()) << endl;
    return -1;
  }
  // Clear out map completely.
  d_manips.clear();
  // Instantiate the DOM parser.
  DOMParser parser;
  parser.setDoValidation(true);
  
  try {
    parser.parse(filename);
  }  catch (const XMLException& toCatch) {
    std::cerr << clString("Error during parsing: '")+
		filename+"'\nException message is:  "+
		xmlto_string(toCatch.getMessage());
    return 0;
  }
  
  DOM_Document doc = parser.getDocument();
  DOM_NodeList list = doc.getElementsByTagName("srcs");
  int nlist = list.getLength();
  cout << "nlist = " << nlist << endl;
  if (nlist == 0) return 0;

  for (int i = 0; i < nlist; i++) {
    DOM_Node d = list.item(i);
    processSrcsNode(d);
  }

  // send the info to the ui
  load_ui();
  return 1;
}


// Write a new sub.mk
void
ManipFields::write_sub_mk()
{
  ofstream fstr((clString(FIELD_MANIP_COMPILATION_PATH) + "/sub.mk")());
  vector<clString> names;
  manips_map_t::iterator itermap = d_manips.begin();
  while(itermap != d_manips.end()) {
    const clString &n = (*itermap).first;
    itermap++;
    ManipData &md =*d_manips[n];
    fstr << n << "LIBS = \\" << endl; 
    svec_t::iterator iter = md.d_libs.begin();
    while (iter != md.d_libs.end()) {
      fstr << "\t\t-l" << *iter++ << " \\" << endl; 
    }

    fstr << n << "LIBPATH = \\" << endl;
    iter = md.d_libpath.begin();
    while (iter != md.d_libpath.end()) {
      fstr << "\t\t-L" << *iter++ << " \\" << endl; 
    }
    
    fstr << n << "INCLUDE = \\" << endl;
    iter = md.d_inc.begin();
    while (iter != md.d_inc.end()) {
      fstr << "\t\t-I" << *iter++ << " \\" << endl;
    }
    //build up a list of names.
    names.push_back(n);
  }
  fstr << "SRCS = \\" << endl;
  vector<clString>::iterator iter = names.begin();
  while (iter != names.end()) {
    fstr << "\t\t" << *iter++ << ".cc \\" << endl;
  }
}

// Tell the ui what its choices are.
void
ManipFields::load_ui()
{
  clString names = "";
  bool first_manip = true;
  // the default is to have the first manip the active manip.
  manips_map_t::iterator iter = d_manips.begin();
  while(iter != d_manips.end()) {
    const clString &n = (*iter).first;
    iter++;

    if (first_manip) {
      set_cur_manip(n);
      d_name.set(n);
      first_manip = false;
    }
    //build up a list of names.
    names += n + " ";
  }
  TCL::execute(d_id + " set_names " + "{" + names + "}");
}

// set the various make related strings in the ui.
void
ManipFields::set_cur_manip(const clString &name)
{
  if (d_manips.count(name) > 0) {

    ManipData &md = *d_manips[name];

    // Set the libs strings in the ui.
    clString libs = "";
    svec_t::iterator iter = md.d_libs.begin();
    while (iter != md.d_libs.end()) {
      libs += *iter++ + " ";
    }
    TCL::execute(d_id + " set_cur_libs " + "{" + libs + "}");

    // Set the libpath strings in the ui.
    clString libpath = "";
    iter = md.d_libpath.begin();
    while (iter != md.d_libpath.end()) {
      libpath += *iter++ + " ";
    }
    TCL::execute(d_id + " set_cur_libpath " + "{" + libpath + "}");

    // Set the inc strings in the ui.
    clString inc = "";
    iter = md.d_inc.begin();
    while (iter != md.d_inc.end()) {
      inc += *iter++ + " ";
    }
    TCL::execute(d_id + " set_cur_inc " + "{" + inc + "}");
  } else {
    // add a new manip 
    d_manips[name] = new ManipData();
    set_cur_manip(name);
  }
}



void
ManipFields::getManips()
{
  char xmlFile[1024];
  sprintf(xmlFile, "%s/srcs.xml", FIELD_MANIP_COMPILATION_PATH);
  // Read xml file to find out all of our available Manipulations
  int check = ReadNodeFromFile(xmlFile);
  if (check != 1) {
    cerr << "ManipFields: XML file did not pass validation" << endl;
  }
}

extern "C" PSECORESHARE Module* make_ManipFields(const clString& id) {
  return new ManipFields(id);
}

} // End namespace SCIRun


