
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Assert.h>
#include <Core/Thread/Mutex.h>
#include <sci_defs.h>
#include <map>
#include <vector>
#include <iostream>

using namespace Uintah;
using namespace std;
using namespace SCIRun;

static Mutex tdLock("TypeDescription::getMPIType lock");

static map<string, const TypeDescription*>* types = 0;
static vector<const TypeDescription*>* typelist=0;
static bool killed=false;

void TypeDescription::deleteAll()
{
  if(!types){
    ASSERT(!killed);
    ASSERT(!typelist);
    return;
  }
  killed=true;
  vector<const TypeDescription*>::iterator iter = typelist->begin();
  for(;iter != typelist->end();iter++) {
    delete *iter;
  }
  delete types;
  types = 0;
  delete typelist;
  typelist = 0;
}

void TypeDescription::register_type()
{
  if(!types){
    ASSERT(!killed);
    ASSERT(!typelist)
    types=scinew map<string, const TypeDescription*>;
    typelist=new vector<const TypeDescription*>;
  }
  
  map<string, const TypeDescription*>::iterator iter = types->find(getName());
  if(iter == types->end()){
    (*types)[getName()]=this;
  }
  typelist->push_back(this);
}

TypeDescription::TypeDescription(Type type, const std::string& name,
				 bool isFlat, MPI_Datatype (*mpitypemaker)())
   : d_type(type), d_subtype(0), d_name(name), d_isFlat(isFlat),
     d_mpitype(MPI_Datatype(-1)), d_mpitypemaker(mpitypemaker), d_maker(0)
{
  register_type();
}

TypeDescription::TypeDescription(Type type, const std::string& name,
				 bool isFlat, MPI_Datatype mpitype)
   : d_type(type), d_subtype(0), d_name(name), d_isFlat(isFlat),
     d_mpitype(mpitype), d_mpitypemaker(0), d_maker(0)
{
  register_type();
}

TypeDescription::TypeDescription(Type type, const std::string& name,
				 Variable* (*maker)(),
				 const TypeDescription* subtype)
   : d_type(type), d_subtype(subtype), d_name(name), d_isFlat(false),
     d_mpitype(MPI_Datatype(-2)), d_mpitypemaker(0), d_maker(maker)
{
  register_type();
}

TypeDescription::~TypeDescription()
{
}

string TypeDescription::getName() const
{
  if(d_subtype) {
    return d_name+"<"+d_subtype->getName()+">";
  } else {
      return d_name;
  }
}

const TypeDescription* TypeDescription::lookupType(const std::string& t)
{
  if(!types){
    types=scinew map<string, const TypeDescription*>;   
    typelist=new vector<const TypeDescription*>;
  }
  map<string, const TypeDescription*>::iterator iter = types->find(t);
  if(iter == types->end())
      return 0;
  return iter->second;
}

TypeDescription::Register::Register(const TypeDescription*/* td*/)
{
  // Registration happens in CTOR
}

TypeDescription::Register::~Register()
{
}

MPI_Datatype TypeDescription::getMPIType() const
{
  if(d_mpitype == MPI_Datatype(-1)){
    tdLock.lock();
    if (d_mpitype == MPI_Datatype(-1)) {
      if(d_mpitypemaker){
	d_mpitype = (*d_mpitypemaker)();
      } else {
	tdLock.unlock();
	throw InternalError("MPI Datatype requested, but do not know how to make it");
      }
    }
    tdLock.unlock();
  }
  ASSERT(d_mpitype != MPI_Datatype(-2));
  return d_mpitype;
}

Variable* TypeDescription::createInstance() const
{
  if(!d_maker)
    throw InternalError("Do not know how to create instance for type: "+getName());
  return (*d_maker)();
}



