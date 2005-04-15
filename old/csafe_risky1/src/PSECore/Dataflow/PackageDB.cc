// PackageDB.cc - Interface to module-finding and loading mechanisms
// $Id$


#include <SCICore/Util/soloader.h>
#ifdef ASSERT
#undef ASSERT
#endif
#include <SCICore/Containers/AVLTree.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/PackageDB.h>
#include <iostream>
#include <ctype.h>
using std::cerr;
using std::ostream;
using std::endl;
using std::cout;
#include <vector>
using std::vector;

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <util/PlatformUtils.hpp>
#include <sax/SAXException.hpp>
#include <sax/SAXParseException.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_NamedNodeMap.hpp>
#include <sax/ErrorHandler.hpp>
#ifdef __sgi
#pragma reset woff 1375
#endif

namespace PSECore {
namespace Dataflow {

using namespace SCICore::Containers;
using SCICore::Containers::AVLTree;
using SCICore::Containers::AVLTreeIter;

typedef struct ModuleInfo_tag {
  ModuleMaker maker;
  clString uiFile;
} ModuleInfo;

typedef AVLTree<clString,ModuleInfo*> Category;
typedef AVLTree<clString,Category*> Package;
typedef AVLTree<clString,Package*> Packages;

typedef AVLTreeIter<clString,ModuleInfo*> CategoryIter;
typedef AVLTreeIter<clString,Category*> PackageIter;
typedef AVLTreeIter<clString,Package*> PackagesIter;

PackageDB packageDB;

PackageDB::PackageDB(void) :
  d_db((void*)new Packages), d_packageList(0)
{
}

PackageDB::~PackageDB(void)
  { 
    delete (Packages*)d_db; 
  }

typedef void (*pkgInitter)(const clString& tclPath);

static void postMessage(const clString& errmsg, bool err=true)
{
    clString tag;
    if(err)
	tag += " errtag";
    TCL::execute(clString(".top.errorFrame.text insert end \"")+errmsg+"\\n\""+tag);
    TCL::execute(".top.errorFrame.text see end");
}

class PackageDBHandler : public ErrorHandler
{
public:
    bool foundError;

    PackageDBHandler();
    ~PackageDBHandler();

    void warning(const SAXParseException& e);
    void error(const SAXParseException& e);
    void fatalError(const SAXParseException& e);
    void resetErrors();

private :
    PackageDBHandler(const PackageDBHandler&);
    void operator=(const PackageDBHandler&);
};

// ---------------------------------------------------------------------------
//  This is a simple class that lets us do easy (though not terribly efficient)
//  trancoding of XMLCh data to local code page for display.
// ---------------------------------------------------------------------------
class StrX
{
public :
    // -----------------------------------------------------------------------
    //  Constructors and Destructor
    // -----------------------------------------------------------------------
    StrX(const XMLCh* const toTranscode)
    {
        // Call the private transcoding method
        fLocalForm = XMLString::transcode(toTranscode);
    }

    StrX(const DOMString& str)
    {
        // Call the transcoding method
        fLocalForm = str.transcode();
    }

    ~StrX()
    {
        delete [] fLocalForm;
    }


    // -----------------------------------------------------------------------
    //  Getter methods
    // -----------------------------------------------------------------------
    const char* localForm() const
    {
        return fLocalForm;
    }

private :
    // -----------------------------------------------------------------------
    //  Private data members
    //
    //  fLocalForm
    //      This is the local code page form of the string.
    // -----------------------------------------------------------------------
    char*   fLocalForm;
};

inline ostream& operator<<(ostream& target, const StrX& toDump)
{
    target << toDump.localForm();
    return target;
}

clString xmlto_string(const DOMString& str)
{
    char* s = str.transcode();
    clString ret = clString(s);
    delete[] s;
    return ret;
}

clString xmlto_string(const XMLCh* const str)
{
    char* s = XMLString::transcode(str);
    clString ret = clString(s);
    delete[] s;
    return ret;
}

PackageDBHandler::PackageDBHandler()
{
    foundError=false;
}

PackageDBHandler::~PackageDBHandler()
{
}

void PackageDBHandler::error(const SAXParseException& e)
{
    foundError=true;
    postMessage(clString("Error at (file ")+xmlto_string(e.getSystemId())
		+", line "+to_string((int)e.getLineNumber())
		+", char "+to_string((int)e.getColumnNumber())
		+"): "+xmlto_string(e.getMessage()));
}

void PackageDBHandler::fatalError(const SAXParseException& e)
{
    foundError=true;
    postMessage(clString("Fatal Error at (file ")+xmlto_string(e.getSystemId())
		+", line "+to_string((int)e.getLineNumber())
		+", char "+to_string((int)e.getColumnNumber())
		+"): "+xmlto_string(e.getMessage()));
}

void PackageDBHandler::warning(const SAXParseException& e)
{
    postMessage(clString("Warning at (file ")+xmlto_string(e.getSystemId())
		+", line "+to_string((int)e.getLineNumber())
		+", char "+to_string((int)e.getColumnNumber())
		+"): "+xmlto_string(e.getMessage()));
}

void PackageDBHandler::resetErrors()
{
}

static void invalidNode(const DOM_Node& n, const clString& filename)
{
    if(n.getNodeType() == DOM_Node::COMMENT_NODE)
	return;
    if(n.getNodeType() == DOM_Node::TEXT_NODE){
	DOMString s = n.getNodeValue();
	char* str = s.transcode();
	bool allwhite=true;
	for(char* p = str; *p != 0; p++){
	    if(!isspace(*p))
		allwhite=false;
	}
	if(!allwhite){
	    postMessage(clString("Extraneous text: ")+str+"after node: "+xmlto_string(n.getNodeName())+"(in file "+filename+")");
	}
	delete[] str;
	return;
    }
    postMessage(clString("Do not understand node: ")+xmlto_string(n.getNodeName())+"(in file "+filename+")");
}

static DOMString findText(DOM_Node& node)
{
    for(DOM_Node n = node.getFirstChild();n != 0; n = n.getNextSibling()){
	if(n.getNodeType() == DOM_Node::TEXT_NODE)
	    return n.getNodeValue();
    }
    return 0;
}

static void processDataflowComponent(PackageDB* db, const DOMString& pkgname,
				     const DOMString& catname,
				     LIBRARY_HANDLE so,
				     const DOM_Node& libNode,
				     const clString& filename)
{
    DOM_NamedNodeMap attr = libNode.getAttributes();
    DOM_Node modname = attr.getNamedItem("name");
    if(modname == 0){
	postMessage("Warning: Module does not have a name, skipping (in package "+xmlto_string(pkgname)+", category "+xmlto_string(catname)+")");
	return;
    }
    DOMString modname_str = modname.getNodeValue();
    ModuleMaker create = 0;
    bool havename=false;
    for(DOM_Node n = libNode.getFirstChild();n != 0; n = n.getNextSibling()){
	DOMString name = n.getNodeName();
	if(name.equals("meta")){
	} else if(name.equals("inputs")){
	} else if(name.equals("outputs")){
	} else if(name.equals("parameters")){
	} else if(name.equals("implementation")){
	    for(DOM_Node nn = n.getFirstChild(); nn != 0; nn = nn.getNextSibling()){
		DOMString nname = nn.getNodeName();
		if(nname.equals("creationFunction")){
		    havename=true;
		    if(create)
			postMessage("Warning: Module specified creationFunction twice: "+xmlto_string(modname_str));
		    DOMString createfn_str = findText(nn);
		    char* createfn = createfn_str.transcode();
		    clString cfn = clString(createfn);
		    create = (ModuleMaker)GetHandleSymbolAddress(so, cfn());
		    if(!create)
			postMessage(clString("Warning: creationFunction not found for module: ")+cfn);
		    delete[] createfn;
		} else {
		    invalidNode(nn, filename);
		}
	    }
	} else {
	    invalidNode(n, filename);
	}
    }
    if(!create || !havename){
	postMessage(clString("Warning: Module did not specify a creationFunction, skipping: ")+xmlto_string(modname_str));
	return;
    }
    char* packageName = pkgname.transcode();
    char* categoryName = catname.transcode();
    char* moduleName = modname_str.transcode();
    db->registerModule(packageName, categoryName, moduleName, create, "not currently used");
    delete[] packageName;
    delete[] categoryName;
    delete[] moduleName;
}

static void processLibrary(PackageDB* db, const DOMString& pkgname,
			   const DOM_Node& libNode, const clString& filename)
{
    DOM_NamedNodeMap attr = libNode.getAttributes();
    DOM_Node catname = attr.getNamedItem("category");
    if(catname == 0){
	postMessage("Warning: Category does not have a name, skipping (in package "+xmlto_string(pkgname)+")");
	return;
    }
    DOMString catname_str = catname.getNodeValue();
    vector<DOMString> sonames;
    for(DOM_Node n = libNode.getFirstChild();n != 0; n = n.getNextSibling()){
	DOMString name = n.getNodeName();
	if(name.equals("soNames")){
	    sonames.clear();
	    for(DOM_Node nn = n.getFirstChild(); nn != 0; nn = nn.getNextSibling()){
		DOMString nname = nn.getNodeName();
		if(nname.equals("soName")){
		    sonames.push_back(findText(nn));
		} else {
		    invalidNode(nn, filename);
		}
	    }
	} else if(name.equals("dataflow-component")){
	    LIBRARY_HANDLE so = 0;
	    for(vector<DOMString>::iterator iter = sonames.begin();
		iter != sonames.end(); iter++){
		char* str = iter->transcode();
		so = GetLibraryHandle(str);
		if(so)
		    break;
	    }
	    if(!so){
		clString libs = "";
		for(vector<DOMString>::iterator iter = sonames.begin();
		    iter != sonames.end(); iter++){
		    if(iter != sonames.begin())
			libs += ", ";
		    libs += xmlto_string(*iter);
		}
		postMessage("Warning: library not found, looked in these names: "+libs);
	    } else {
		processDataflowComponent(db, pkgname, catname_str, so, n, filename);
	    }
	} else if(name.equals("alias")){
	    DOM_NamedNodeMap attr = n.getAttributes();
	    DOM_Node fromPackage = attr.getNamedItem("package");
	    DOM_Node fromCategory = attr.getNamedItem("category");
	    DOM_Node fromModule = attr.getNamedItem("module");
	    DOMString toModule = findText(n);
	    if(fromPackage == 0 || fromCategory == 0 || fromModule == 0){
		postMessage("Warning: Alias did not specify package, category and module: "+xmlto_string(toModule));
		continue;
	    }

	    DOMString fromPackageString = fromPackage.getNodeValue();
	    DOMString fromCategoryString = fromCategory.getNodeValue();
	    DOMString fromModuleString = fromModule.getNodeValue();

	    char* toPackageName = pkgname.transcode();
	    char* toCategoryName = catname_str.transcode();
	    char* toModuleName = toModule.transcode();
	    char* fromPackageName = fromPackageString.transcode();
	    char* fromCategoryName = fromCategoryString.transcode();
	    char* fromModuleName = fromModuleString.transcode();
	    db->createAlias(fromPackageName, fromCategoryName, fromModuleName,
			    toPackageName, toCategoryName, toModuleName);
	    TCL::execute(clString("createAlias ")+fromPackageName+" "+fromCategoryName+" "+fromModuleName+" "+toPackageName+" "+toCategoryName+" "+toModuleName);
	} else {
	    invalidNode(n, filename);
	}
    }
}

static void processPackage(PackageDB* db, const DOM_Node& pkgNode,
			   const clString& filename)
{
    DOM_NamedNodeMap attr = pkgNode.getAttributes();
    DOM_Node pkgname = attr.getNamedItem("name");
    if(pkgname == 0){
	postMessage("Warning: Package does not have a name, skipping (in"+filename+")");
	return;
    }
    DOMString pkgname_str = pkgname.getNodeValue();
    for(DOM_Node n = pkgNode.getFirstChild();n != 0; n = n.getNextSibling()){
	DOMString name = n.getNodeName();
	if(name.equals("scirun-library")){
	    processLibrary(db, pkgname_str, n, filename);
	} else if(name.equals("guiPath")){
	    DOMString p = findText(n);
	    clString path = xmlto_string(p);
	    if(path(0) != '/')
		path = pathname(filename)+"/"+path;
	    TCL::execute(clString("lappend auto_path ")+path);
	} else {
	    invalidNode(n, filename);
	}
    }
}

void PackageDB::loadPackage(const clString& packPath)
{
    // Initialize the XML4C system
    try {
        XMLPlatformUtils::Initialize();
    } catch (const XMLException& toCatch) {
	cerr << "Error during initialization! :\n"
	     << StrX(toCatch.getMessage()) << endl;
	return;
    }

    // The format of a package path element is either URL,URL,...
    // Where URL is a filename or a url to an XML file that
    // describes the components in the package.
    clString packagePath = packPath;
    while(packagePath!="") {

	// Strip off the first element, leave the rest in the path for the next
	// iteration.

	clString packageElt;
	int firstComma=packagePath.index(',');
	if(firstComma!=-1) {
	    packageElt=packagePath.substr(0,firstComma);
	    packagePath=packagePath.substr(firstComma+1,-1);
	} else {
	    packageElt=packagePath;
	    packagePath="";
	}

	// Load the package
	postMessage(clString("Loading package '")+packageElt+"'", false);
	
	// Instantiate the DOM parser.
	DOMParser parser;
	parser.setDoValidation(false);

	PackageDBHandler handler;
	parser.setErrorHandler(&handler);

	//
	//  Get the starting time and kick off the parse of the indicated
	//  file. Catch any exceptions that might propogate out of it.
	//
	try {
	    parser.parse(packageElt());
	}  catch (const XMLException& toCatch) {
	    postMessage(clString("Error during parsing: '")+packageElt+"'\nException message is:  "+xmlto_string(toCatch.getMessage()));
	    handler.foundError=true;
	    continue;
	}

	if(handler.foundError){
	    TCL::execute("tk_dialog .errorPopup {Parse Error} {Error parsing package file, see message window for more information} error 0 Ok");
	    continue;
	}
	//
	//  Extract the components from the DOM tree
	//
	DOM_Document doc = parser.getDocument();
	DOM_NodeList list = doc.getElementsByTagName("package");
	int nlist = list.getLength();
	for(int i=0;i<nlist;i++){
	    DOM_Node n = list.item(i);
	    processPackage(this, n, packageElt);
	}
	if(handler.foundError){
	    TCL::execute("tk_dialog .errorPopup {Processing Error} {Error processing package file, see message window for more information} error 0 Ok");
	}
  }

  TCL::execute("createCategoryMenu");
}

void PackageDB::registerModule(const clString& packageName,
                               const clString& categoryName,
                               const clString& moduleName,
                               ModuleMaker moduleMaker,
                               const clString& tclUIFile) {
  Packages* db=(Packages*)d_db;

  Package* package;
  if(!db->lookup(packageName,package))
    {
      db->insert(packageName,package=new Package);
      d_packageList.add( packageName );
    }

  Category* category;
  if(!package->lookup(categoryName,category))
    package->insert(categoryName,category=new Category);

  ModuleInfo* moduleInfo;
  if(!category->lookup(moduleName,moduleInfo)) {
    moduleInfo=new ModuleInfo;
    category->insert(moduleName,moduleInfo);
  } else cerr << "WARNING: Overriding multiply registered module "
              << packageName << "." << categoryName << "."
              << moduleName << "\n";

  moduleInfo->maker=moduleMaker;
  moduleInfo->uiFile=tclUIFile;
}

void PackageDB::createAlias(const clString& fromPackageName,
			    const clString& fromCategoryName,
			    const clString& fromModuleName,
			    const clString& toPackageName,
			    const clString& toCategoryName,
			    const clString& toModuleName)
{
    Packages* db=(Packages*)d_db;

    Package* package;
    if(!db->lookup(fromPackageName,package)) {
	postMessage("Warning: creating an alias from a nonexistant package "+fromPackageName+" (ignored)");
	return;
    }

    Category* category;
    if(!package->lookup(fromCategoryName,category)) {
	postMessage("Warning: creating an alias from a nonexistant category "+fromPackageName+"."+fromCategoryName+" (ignored)");
	return;
    }

    ModuleInfo* moduleInfo;
    if(!category->lookup(fromModuleName,moduleInfo)) {
	postMessage("Warning: creating an alias from a nonexistant module "+fromPackageName+"."+fromCategoryName+"."+fromModuleName+" (ignored)");
	return;
    }
    registerModule(toPackageName, toCategoryName, toModuleName,
		   moduleInfo->maker, moduleInfo->uiFile);
}

Module* PackageDB::instantiateModule(const clString& packageName,
                                     const clString& categoryName,
                                     const clString& moduleName,
                                     const clString& instanceName) const {
  Packages* db=(Packages*)d_db;

  Package* package;
  if(!db->lookup(packageName,package)) {
    cerr << "ERROR: Instantiating from nonexistant package " << packageName 
         << "\n";
    return 0;
  }

  Category* category;
  if(!package->lookup(categoryName,category)) {
    cerr << "ERROR: Instantiating from nonexistant category " << packageName
         << "." << categoryName << "\n";
    return 0;
  }

  ModuleInfo* moduleInfo;
  if(!category->lookup(moduleName,moduleInfo)) {
    cerr << "ERROR: Instantiating nonexistant module " << packageName 
         << "." << categoryName << "." << moduleName << "\n";
    return 0;
  }

#if 0
  // This was McQ's somewhat silly replacement for TCL's tclIndex/auto_path
  // mechanism.  The idea was that there would be a path in the index.cc
  // that pointed to a TCL file to source before instantiating a module
  // of some particular class for the frist time -- sortof a TCL-end class
  // constructor for the module's class.
  //
  // Steve understandably doesn't like new, fragile mechanisms where
  // perfectly good old, traditional ones already exist, so he if0'd this
  // away and added the "lappend auto_path" at package-load-time, above.
  //
  // This code is still here 'cause Some Day it might be nice to allow the
  // source of the TCL files to be stored in the .so (as strings) and eval'd
  // here.  This was the "faraway vision" that drove me to do things this way
  // in the first place, but since that vision seems to have stalled
  // indefinately in lieu of Useful Work, there's no reason not to use
  // auto_path (except that it produces yet one more file to maintain).  And
  // auto_path is useful if you write global f'ns and want to use them in lots
  // of your modules -- auto_path nicely handles this whereas the code below
  // doesn't handle it at all.
  //
  // Some day it might be nice to actually achieve the "package is one .so
  // and that's all" vision, but not today.  :)
  //
  //                                                      -mcq 99/10/6

  if(moduleInfo->uiFile!="") {
    clString result;
    if(!TCL::eval(clString("source ")+moduleInfo->uiFile,result)) {
      cerr << "Can't source UI file " << moduleInfo->uiFile << "...\n";
      cerr << "  TCL Error: " << result << "\n";
    }
    moduleInfo->uiFile="";                       // Don't do it again
  }
#endif

  Module *module = (moduleInfo->maker)(instanceName);
  module->packageName = packageName;
  module->moduleName = moduleName;
  module->categoryName = categoryName;

  return module;
}

Array1<clString> PackageDB::packageNames(void) const {

  // d_packageList is used to keep a list of the packages 
  // that are in this PSE IN THE ORDER THAT THEY ARE SPECIFIED
  // by the user in the Makefile (for main.cc) or in their
  // environment.

  return d_packageList;
}

Array1<clString>
PackageDB::categoryNames(const clString& packageName) const {
  Packages* db=(Packages*)d_db;

  {
    PackagesIter iter(db);
    for(iter.first();iter.ok();++iter) if(iter.get_key()==packageName) {
      Package* package=iter.get_data();
      Array1<clString> result(package->size());
      {
        PackageIter iter(package);
        int i=0;
        for(iter.first();iter.ok();++iter) result[i++]=iter.get_key();
      }
      return result;
    }
  }
  cerr << "WARNING: Unknown package " << packageName << "\n";

  Array1<clString> result(0);
  return result;
}

Array1<clString>
PackageDB::moduleNames(const clString& packageName,
		       const clString& categoryName) const {
  Packages* db=(Packages*)d_db;
  {
    PackagesIter iter(db);
    for(iter.first();iter.ok();++iter) if(iter.get_key()==packageName) {
      Package* package=iter.get_data();
      {
        PackageIter iter(package);
        for(iter.first();iter.ok();++iter) if(iter.get_key()==categoryName) {
          Category* category=iter.get_data();
          Array1<clString> result(category->size());
          {
            CategoryIter iter(category);
            int i=0;
            for(iter.first();iter.ok();++iter) result[i++]=iter.get_key();
          }
          return result;
        }
        cerr << "WARNING: Unknown category " << packageName << "."
             << categoryName << "\n";
      }
    }
  }
  cerr << "WARNING: Unknown package " << packageName << "\n";

  Array1<clString> result(0);
  return result;
}

} // Dataflow namespace
} // PSECore namespace

//
// $Log$
// Revision 1.15  2000/03/17 08:24:52  sparker
// Added XML parser for component repository
//
// Revision 1.14  1999/10/07 02:07:20  sparker
// use standard iostreams and complex type
//
// Revision 1.13  1999/10/06 20:37:37  mcq
// Added memoirs.
//
// Revision 1.12  1999/09/22 23:51:46  dav
// removed debug print
//
// Revision 1.11  1999/09/22 22:39:50  dav
// updated to use tclIndex files
//
// Revision 1.10  1999/09/08 02:26:41  sparker
// Various #include cleanups
//
// Revision 1.9  1999/09/04 06:01:41  sparker
// Updates to .h files, to minimize #includes
// removed .icc files (yeah!)
//
// Revision 1.8  1999/08/31 23:27:53  sparker
// Added Log and Id entries
//
//
