#include "pdtParser.h"

///////////
void List::insert(std::string item) {
  if(m_set.find(item) == m_set.end())
    m_set.insert(item);
}

void List::remove(std::string item) {
  std::set<std::string, ltstr>::iterator iter;
  iter = m_set.find(item);
  if(iter != m_set.end()) {
    m_set.erase(iter);
  }
}

int List::exists(std::string item) {
  return (m_set.find(item) != m_set.end());
}

void List::clear() {
  m_set.clear();
}

/////////
//Parses namespaces and everything underneath them
void pdtParser::parseNamespaces(PDB::namespacevec& nv, IrDefList* deflist) {

  for (PDB::namespacevec::const_iterator n = nv.begin(); n != nv.end() ;++n) {
    IrDefList* pkglist = new IrDefList();

    // iterate over namespace members
    pdbNamespace::memvec mv = (*n)->members();
    for (pdbNamespace::memvec::const_iterator m = mv.begin(); m != mv.end() ;++m) {
        const pdbItem *memitem = static_cast<const pdbItem *>(*m);

	//Cast into class and now output classes
        if (((const pdbGroup *)memitem)->kind() == pdbItem::GR_CLASS) {
	  IrDefinition* def = parseClass((const pdbClass *)memitem,false);
          if(def!=NULL) pkglist->addDef(def);	  
	}
    }
    if(pkglist->getSize() > 0) {
      IrPackage* pkg = new IrPackage((*n)->name(),pkglist);
      deflist->addDef(pkg);
    }
  }
}

/////////
// Parse all non-namespace classes
void pdtParser::parseClasses(PDB::classvec &cv, IrDefList* deflist){
  for (PDB::classvec::const_iterator c = cv.begin(); c != cv.end() ;++c) {
    IrDefinition* def = parseClass((*c),false);
    if(def!=NULL) deflist->addDef(def);
  }
}

//////////
//Parses individual classes
IrPort* pdtParser::parseClass(const pdbClass *c, bool isCStruct, IrMethodList* irmL) {
  string className;
  bool nulify = false;
  className = c->name();

  if(!irmL) {
    //Not a base class
    if(parsedClasses.exists(c->fullName())) return NULL;
    parsedClasses.insert(c->fullName());
    irmL = new IrMethodList();
  }
  else {
    //Base class
    nulify = true;
  }

  //Parse base classes of this class recursively
  const pdbClass::basevec &bv = c->baseClasses();
  if(bv.size() != 0) {
     for(pdbClass::basevec::const_iterator b = bv.begin(); b != bv.end(); ++b) {
        pdbBase *base = static_cast<pdbBase *>(*b);
	parseClass(base->base(),false,irmL);
    }
  }

  //Now parse this class
  if(!(language == PDB::LA_C || isCStruct == true)) {
    const pdbClass::methodvec mv = c->methods();
    for(pdbClass::methodvec::const_iterator m = mv.begin(); m != mv.end(); ++m) {
      const pdbMethod *meth = static_cast<const pdbMethod *>(*m);
      if(!parsedMethods.exists(meth->func()->name())) {
	parsedMethods.insert(meth->func()->name());
	IrMethod* irm = parseCFunction(meth->func());
	if(irm != NULL) irmL->addMethod(irm);
      }
    }
  }

  if(nulify) return NULL;
  parsedMethods.clear();
  return(new IrPort(className,irmL));
}

/////////
// Parse all non-class C/C++ routines
void pdtParser::parseCFunctions(PDB::croutinevec &crv, IrMethodList* mlist) {
  for(PDB::croutinevec::const_iterator c = crv.begin(); c != crv.end(); ++c) {
    //Check if we have already processed this function
    if(!parsedFunctions.exists((*c)->fullName())) {
      parsedFunctions.insert((*c)->fullName());
      IrMethod* irm = parseCFunction(*c);
      if(irm != NULL) mlist->addMethod(irm);
    }  
  }
}

/////////
//Parses C/C++ routines and methods
IrMethod* pdtParser::parseCFunction(const pdbCRoutine *cur) {
  const pdbType *sig = cur->signature();
  const pdbType *ret;
  IrArgumentList* irArgL = new IrArgumentList();

  //Exiting on compiler generated routines
  if(cur->isCompilerGenerated()) return NULL;

  //Exiting on private methods
  if(cur->access() == pdbItem::AC_PRIV) return NULL;

  if (sig != NULL) {
    ret = sig->returnType();
    const pdbType::argvec av = sig->arguments();
    for(pdbType::argvec::const_iterator a = av.begin(); a != av.end(); ++a) {
      const pdbArg cur = static_cast<const pdbArg>(*a);
      IrArgument* irArg = parseArgument(&cur);
      if(irArg != NULL) irArgL->addArg(irArg);
    }
  }
  
  return(new IrMethod(cur->name(),ret->name(),irArgL));
}

/////////
//Parsing arguments
IrArgument* pdtParser::parseArgument(const pdbArg *arg) {
  const pdbType* type = arg->type();
  if (arg->intentOut()) {
    if (arg->intentIn()) {
      return(new IrArgument("inout",type->name(),arg->name()));
    } else {
      return(new IrArgument("out",type->name(),arg->name()));
    }
  } else {
    return(new IrArgument("in",type->name(),arg->name()));
  }
}

////////
//Parses Fortran modules
void pdtParser::parseFModules(PDB::modulevec& mv, IrDefList* deflist) {


} 


////////
//Main function to be called to start the parser; 
IrDefList* pdtParser::pdtParse(const char* filename) {

  PDB pdb(const_cast<char*>(filename)); 
  if(!pdb) return NULL;
  IrDefList* deflist = new IrDefList();
  IrMethodList* stdmlist = new IrMethodList();

  language = pdb.language();

  if((language == PDB::LA_CXX)||(language == PDB::LA_C)||(language == PDB::LA_C_or_CXX)) {
    //Parse namespaces and all classes within them
    parseNamespaces(pdb.getNamespaceVec(),deflist);
    
    //Parse classes and all methods within
    parseClasses(pdb.getClassVec(),deflist);
    
    //Parse functions that don't belong to any container
    parseCFunctions(pdb.getCRoutineVec(),stdmlist);
    if(stdmlist->IrMethods.size() > 0) 
      deflist->addDef(new IrPort("std",stdmlist));  
  }
  else if(language == PDB::LA_FORTRAN) {
    std::cerr << "FORTRAN not finished\n";
  }

  return deflist;
}




