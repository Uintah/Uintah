/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  ManipMatrix.cc:
 *
 *  Written by:
 *   Dimitri Yatsenko
 *   Mon Mar 26 09:35:23 MST 2000
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/share/share.h>
#include <Core/Parts/GuiVar.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
#include <Dataflow/Network/StrX.h>
#include <Core/Util/DebugStream.h>
#include <dlfcn.h>
#include <Core/Util/sci_system.h>
#include <string>

namespace SCIRun {

using namespace std;

#ifndef MM_SRC
#error sub.mk needs to define MM_SRC
#endif

#ifndef MM_OBJ
#error sub.mk needs to define MM_OBJ
#endif

const char* MATRIX_MANIP_SRC = MM_SRC;
const char* MATRIX_MANIP_OBJ = MM_OBJ;


typedef void (*MatrixManipFunction)(vector<MatrixHandle>& in, vector<MatrixHandle>& out);



class PSECORESHARE ManipMatrix : public Module 
{
public:
  ManipMatrix(const string& id);
  virtual ~ManipMatrix();

  // Compile and load .so for the selected manipulation
  virtual void execute();
  virtual void tcl_command(TCLArgs& args, void* userdata);

private:
  typedef vector<string>          StringVector;

  // ---- Compilation vars and xml specific. ----
  struct ManipData
  {
    string         desc_;
    StringVector   libs_;
    StringVector   libpath_;
    StringVector   inc_;
  };

  typedef map<string, ManipData*> manips_map_t;

  static manips_map_t      manips_;  //TODO: why is it static?

  static string vector_to_string( const StringVector&, const string& seperator ); 
  void change_md_data( StringVector &v, const string &arg );

  // helpers to deal with the XML 
  void readSrcNode     ( DOM_Node );
  void readSrcsNode    ( DOM_Node );
  int  ReadNodeFromFile( const string& filename );
  void getManips       ();
  void load_ui         ();
  void set_cur_manip   ( const string& name );
  static bool create_dummy_cc ( const string& filename );
  bool write_sub_mk    () const;
  bool save_xml        () const;

  // ---- Compilation and dynamic loading specific. ----
  bool               compileFile      ( const string& filename );
  MatrixManipFunction getFunctionFromDL( const string& filename);

  GuiString                  name_;
  MatrixManipFunction        curFun_;
};



// Static initializers
ManipMatrix::manips_map_t ManipMatrix::manips_;



ManipMatrix::ManipMatrix( const string& id ) 
  : Module("ManipMatrix", id, Source, "Math", "SCIRun" )
  , name_("manipulationName", id, this)
  , curFun_(0)
{
}



ManipMatrix::~ManipMatrix() 
{}



//////////
// ManipMatrix::execute
// 
// Compile and load .so for the selected manipulation
void 
ManipMatrix::execute()
{
  // Load all of the available Manipulations from the xml file.

  getManips();
  
  cerr << "try to load Manip " << endl;

  if( !curFun_ ) 
  {
     cerr << "no MatrixManipFunction loaded" << endl;
  }
  else
  {
    vector<MatrixHandle> allIn;
    vector<MatrixHandle> allOut; 
    MatrixHandle         smatrix;

    //collect input matrix
    MatrixIPort* ifield;
    for( int i = 0; i < niports(); ++i )
    {
      ifield = dynamic_cast<MatrixIPort*>( get_iport(i) );
      ifield->get( smatrix );
      allIn.push_back( smatrix );
    }

    curFun_( allIn, allOut );
    
    //send out output fields
    MatrixOPort* ofield;
    int portCount(0);
    for( vector<MatrixHandle>::const_iterator i = allOut.begin()
         ; i != allOut.end()
         , portCount < noports()    //TODO: ask Chris Moulding what to do here
         ; ++i
         , ++portCount
       )
    {
      ofield = dynamic_cast<MatrixOPort*>( get_oport(portCount) ); //can cast because it was newed as MatrixOPort
      ofield->send( *i );
    }
  } 
  cerr << "done Manip execute" << endl;
}



//////////
// ManipMatrix::tcl_command
// 
void 
ManipMatrix::tcl_command(TCLArgs& args, void* userdata)
{
  name_.reset();
  cout << "TCLCOMMAND: curname: " << name_.get() << endl;
  
  //TODO: if not already created, add to XML

  if( args.count() < 2 )
  {
    args.error("ManipMatrix needs a minor command");
    return;
  }

  cout << "TCLCOMMMAND: arg1: " << args[1] << endl; 

  if( manips_.count( name_.get() ) == 0 ) 
  {
    string name = name_.get();
    //add new manip. 
    if( !name.empty() )
    { 
      cout << "adding " << name << "to manips" << endl;
      manips_[name] = new ManipData();
      cout << "Creating dummy cc" << endl;
      create_dummy_cc(name);
      cout << "Writing new xml" << endl;
      save_xml() ;
    }
  }

  ManipData &manipData = *manips_[name_.get()];

  if( args[1] == "compile" ) 
  {
    write_sub_mk();
    cout << "C++ compile now!!" << endl;
    curFun_ = getFunctionFromDL(name_.get());
    if( !curFun_ )
      cerr << "Failed to load so" << endl;
  }  
  else if( args[1] == "launch-editor" )
  {
    string ecmd = args[2] + " " + MATRIX_MANIP_SRC
      + "/" + name_.get() + ".cc &";
    
    cout << "Launch Editor: " << ecmd << endl;

    if( sci_system(ecmd.c_str()) ) 
    {
      cerr << "SCIRun::Modules::Matrix::ManipMatrix " << endl
	   << "editor command failed" << endl;
    }
  } 
  else if( args[1] == "libs" ) 
  {
    change_md_data(manipData.libs_, args[2]());
  } 
  else if( args[1] == "libpath" ) 
  {
    change_md_data(manipData.libpath_, args[2]());
  } 
  else if (args[1] == "inc") 
  {
    change_md_data(manipData.inc_, args[2]());
  } 
  else if( args[1] == "loadmanip" )
  {
    for( manips_map_t::const_iterator m = manips_.begin(); m != manips_.end(); ++m)  
    {
      cout << m->first << ": " << m->second->desc_ << endl;
    }
    set_cur_manip(args[2]());
    curFun_ = 0;
  } 
  else 
  {
    Module::tcl_command(args, userdata);
  }
}



void 
ManipMatrix::change_md_data(StringVector &v, const string& arg) 
{
  cout << "change_md_data: " << arg << endl;

  if (arg == "clear") 
  { 
    cout << "v clearing" << endl;
    v.clear();
  }
  else
  {
    cout << "v adding :" << arg << endl;
    v.push_back(arg);  
  }
  save_xml();
}



bool ManipMatrix::save_xml() const
{
  ofstream f( ( string(MATRIX_MANIP_SRC) + "/srcs.xml").c_str() );
  bool ret = f.is_open();
  
  if( !ret )
  { 
     cerr << "Could not open: " << MATRIX_MANIP_SRC << "/srcs.xml"
          << endl;
  }
  else
  { 
    cout << "Saving srcs.xml" << endl;
    f << "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n";
    f << "<!DOCTYPE srcs SYSTEM \"srcs.dtd\">\n\n"; 
    f << "<srcs>\n";
    for( manips_map_t::const_iterator i = manips_.begin()
            ; i != manips_.end(); ++i )
    {
      f << "  <src name=\"" << i->first <<"\">\n";
      f << "    <description>" << i->second->desc_ << "</description>\n";

      for( StringVector::const_iterator s = i->second->libs_.begin()
            ; s != i->second->libs_.end(); ++s )
      {
        f << "    <lib name=\"" << *s << "\"/>\n";
      }

      for( StringVector::const_iterator s = i->second->libpath_.begin()
            ; s != i->second->libpath_.end(); ++s )
      {
        f << "    <libpath name=\"" << *s << "\"/>\n";
      }
      
      for( StringVector::const_iterator s = i->second->inc_.begin()
            ; s != i->second->inc_.end(); ++s )
      {
        f << "    <inc name=\"" << *s << "\"/>\n";
      }

      f << "  </src>\n";
    }
    f << "</srcs>\n";
  }

  return ret;
}


//////////
// ManipMatrix::getFunctionFromDL
// 
MatrixManipFunction
ManipMatrix::getFunctionFromDL(const string& f)
{
  MatrixManipFunction ret = 0;

  if( compileFile(f) ) 
  {
    // Load the dl.
    string dynamicLibrary = string( MATRIX_MANIP_OBJ ) + "/" 
      + f + ".so";

    void *h = dlopen( dynamicLibrary.c_str(), RTLD_NOW );
    if( h == 0 ) 
    {
      cerr << "ManipMatrix::getFunctionFromDL() error:"
	   << "could not open " << dynamicLibrary << endl;
      cerr << "Reported Error: " << dlerror() << endl;
    }
    else
    {
      ret = (MatrixManipFunction)dlsym(h, "execute"); 
      if( !ret ) 
      {
        cerr << "ManipMatrix::getFunctionFromDL() error: "
             << "dlsym failed" << endl;
	cerr << "Reported Error: " << dlerror() << endl;
      }
    }
  }
  return ret;
}



//////////
// ManipMatrix::compileFile
//
// Attempt to compile filename into a .so, return true if it succeeded
// false otherwise.

bool 
ManipMatrix::compileFile(const string& file)
{
  string command = string("cd ") + MATRIX_MANIP_OBJ 
          + "; gmake " + file + ".so";

  cout << "Executing: " << command << endl;
  bool compiledSuccessfully =  sci_system(command.c_str()) == 0; 
  if( !compiledSuccessfully )
  {
    cerr << "SCIRun::Modules::Matrix::ManipMatrix::compileFile() error: "
	 << "system call failed:\n" << command << endl;
  }

  return compiledSuccessfully;
}



void 
ManipMatrix::readSrcNode(DOM_Node n) 
{
  ManipData* manipData = 0;
  DOM_NamedNodeMap attribute = n.getAttributes();

  if( attribute != 0 ) 
  {
    DOM_Node d = attribute.getNamedItem("name");
    cout << "name: " << d.getNodeValue() << endl;
    manipData = new ManipData;
    manips_[d.getNodeValue()] = manipData;
  }

  if( manipData ) 
  {
    for( DOM_Node child = n.getFirstChild(); child != 0
              ; child = child.getNextSibling() )
    {
      DOMString childname = child.getNodeName();
      if( childname.equals("description") ) 
      {
        cout << "description: " 
             << removeWhiteSpace(getSerializedChildren(child)) << endl;
        manipData->desc_ = string(removeWhiteSpace
                         (getSerializedChildren(child)))();
      }
      else if( childname.equals("lib") ) 
      {
        DOM_Node d = child.getAttributes().getNamedItem("name");
        cout << "lib: " << d.getNodeValue() << endl;

        manipData->libs_.push_back(d.getNodeValue());
      }
      else if (childname.equals("libpath")) 
      {
        DOM_Node d = child.getAttributes().getNamedItem("name");
        cout << "libpath : " << d.getNodeValue() << endl;
        manipData->libpath_.push_back(d.getNodeValue)));
      }
      else if( childname.equals("inc") )
      {
        DOM_Node d = child.getAttributes().getNamedItem("name");
        cout << "include: " << d.getNodeValue() << endl;
        manipData->inc_.push_back(d.getNodeValue());
      }
    }
  }
}



void 
ManipMatrix::readSrcsNode(DOM_Node n) 
{
  for( DOM_Node child = n.getFirstChild(); child != 0
              ; child = child.getNextSibling()) 
  {
    DOMString childname = child.getNodeName();
    if( childname.equals("src") )
      readSrcNode(child);
  }
}



int 
ManipMatrix::ReadNodeFromFile(const string& filename)
{
  // Initialize the XML4C system
  try 
  {
    XMLPlatformUtils::Initialize();
  } 
  catch( const XMLException& toCatch )
  {
    std::cerr << "Error during initialization! :\n"
	 << StrX(toCatch.getMessage()) << endl;
    return -1;
  }

  manips_.clear();
 
  DOMParser parser;
  parser.setDoValidation(true);
  
  try 
  {
    parser.parse( filename.c_str() );
  }  
  catch (const XMLException& toCatch) 
  {
    std::cerr << "Error during parsing: '"
		+ filename + "'\nException message is:  " 
		+ xmlto_string(toCatch.getMessage())();
    return 0;
  }
  
  DOM_Document doc  = parser.getDocument();
  DOM_NodeList list = doc.getElementsByTagName("srcs");
  int nlist = list.getLength();
  cout << "nlist = " << nlist << endl;

  if (nlist == 0) 
    return 0;

  for( int i = 0; i < nlist; ++i ) 
  {
    DOM_Node d = list.item(i);
    readSrcsNode(d);
  }

  // send the info to the ui
  load_ui();
  return 1;
}



bool ManipMatrix::create_dummy_cc( const string& filename )
{
  cout << "Accessing " << filename << endl;
  ifstream f( ( MATRIX_MANIP_SRC 
          + ( "/" + filename + ".cc" ) ).c_str() );
  bool ret = f.is_open();

  if( ret )
  {
    cout << filename << " already exists." << endl; 
  }
  else 
  {
     cout << "Creating " + filename + ".cc";
     f.close();
     string command = string("cp ") 
          + MATRIX_MANIP_SRC + "/template.cc "
          + MATRIX_MANIP_SRC + "/" + filename + ".cc"; 
     ret = sci_system( command.c_str() );
     if( !ret )
       cerr << "Could not execute: " << command << endl;
  } 

  return ret;
}



// Write a new sub.mk
bool
ManipMatrix::write_sub_mk() const
{
  ofstream f( ( string(MATRIX_MANIP_SRC) + "/sub.mk").c_str() );

  bool ret( f.is_open() );

  if( ret )
  {
    f << "#Makefile for matrix manipulation code to be compiled and "
      << "loaded at run time.\n\n";

    for( manips_map_t::const_iterator m = manips_.begin()
        ; m != manips_.end(); ++m ) 
    {
      const string&       manipName = m->first;
      const ManipData&    manipData = *m->second;

      f << manipName << "LIBS = \\\n";
 
      for( StringVector::const_iterator i = manipData.libs_.begin()
         ; i != manipData.libs_.end(); ++i )
      {
        f << "\t\t-l" << *i << " \\\n"; 
      }
      f << "\n";

      f << manipName << "LIBPATH = \\\n" << endl;
      for( StringVector::const_iterator i = manipData.libpath_.begin()
         ; i != manipData.libpath_.end(); ++i ) 
      {
        f << "\t\t-L" << *i << " \\\n"; 
      }
      f << "\n";
    
      f << manipName << "INCLUDE = \\" << endl;
      for( StringVector::const_iterator i = manipData.inc_.begin()
         ; i != manipData.inc_.end(); ++i )
      {
        f << "\t\t-I" << *i << " \\\n";
      }
      f << "\n";
    }

    f << "SRCS = \\\n";
    for( manips_map_t::const_iterator m = manips_.begin()
  	; m != manips_.end(); ++m )
    {
      f << "\t\t" << m->first << ".cc \\\n";
    }
    f << "\n"; 
  }
  return ret;
}



// Tell the ui what its choices are.
void
ManipMatrix::load_ui()
{
  string names;

  // the default is to have the first manip the active manip.
  for( manips_map_t::const_iterator i = manips_.begin() 
          ; i != manips_.end(); ++i )
  {
    const string& manipName = i->first;

    if( i == manips_.begin() ) 
    {
      set_cur_manip( manipName );
      name_.set    ( manipName.c_str() );
    }

    names += manipName + " ";
  }

  tcl_execute(( id + " set_names " + "{" + names + "}" ).c_str());
}



string
ManipMatrix::vector_to_string( const StringVector& stringVector,
			       const string& separator = " " )
{
  string ret;
  for( StringVector::const_iterator i = stringVector.begin()
     ; i != stringVector.end(); ++i )
  {
    ret += *i + separator;
  }
  return ret;
}



// set the various make related strings in the ui.
void
ManipMatrix::set_cur_manip( const string& name )
{
  ManipData& manipData = *manips_[name];

  // Set ui strings
  cout << "Setting ui strings" << endl;
  const string libs    = vector_to_string( manipData.libs_    );
  const string libpath = vector_to_string( manipData.libpath_ );
  const string inc     = vector_to_string( manipData.inc_     );

  tcl_execute((id + " set_cur_libs "    + "{" + libs    + "}").c_str());
  tcl_execute((id + " set_cur_libpath " + "{" + libpath + "}").c_str());
  tcl_execute((id + " set_cur_inc "     + "{" + inc     + "}").c_str());
}



void
ManipMatrix::getManips()
{
  //reads all available manipulations from an xml file
  static bool beenHere( false ); 
  if( !beenHere ) 
  {
    beenHere = true;
    string xmlFile
        = string( MATRIX_MANIP_SRC ) + "/srcs.xml";
    int check = ReadNodeFromFile(xmlFile);
    if( check != 1 ) 
    {
      cerr << "ManipMatrix: XML file did not pass validation" << endl;
    }
  }
}



extern "C" PSECORESHARE Module* make_ManipMatrix(const string& id) 
{
  return new ManipMatrix(id());
}

} // End namespace SCIRun


