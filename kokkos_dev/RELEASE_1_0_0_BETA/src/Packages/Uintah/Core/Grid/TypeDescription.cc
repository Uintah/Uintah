
#include <Packages/Uintah/Core/Grid/TypeDescription.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Assert.h>
#include <sci_defs.h>
#include <map>
#include <iostream>

using namespace Uintah;
using namespace std;
using namespace SCIRun;

static map<string, const TypeDescription*>* types = 0;

TypeDescription::TypeDescription(Type type, const std::string& name,
				 bool isFlat, MPI_Datatype (*mpitypemaker)())
   : d_type(type), d_subtype(0), d_name(name), d_isFlat(isFlat),
     d_mpitype(-1), d_mpitypemaker(mpitypemaker), d_maker(0)
{
}

TypeDescription::TypeDescription(Type type, const std::string& name,
				 bool isFlat, MPI_Datatype mpitype)
   : d_type(type), d_subtype(0), d_name(name), d_isFlat(isFlat),
     d_mpitype(mpitype), d_mpitypemaker(0), d_maker(0)
{
}

TypeDescription::TypeDescription(Type type, const std::string& name,
				 Variable* (*maker)(),
				 const TypeDescription* subtype)
   : d_type(type), d_subtype(subtype), d_name(name), d_isFlat(false),
     d_mpitype(-2), d_mpitypemaker(0), d_maker(maker)
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
  if(!types)
    types=scinew map<string, const TypeDescription*>;   
  
  map<string, const TypeDescription*>::iterator iter = types->find(t);
   if(iter == types->end())
      return 0;
   return iter->second;
}

TypeDescription::Register::Register(const TypeDescription* td)
{
  //  cerr << "Register: td=" << td << ", name=" << td->getName() << '\n';
  if(!types)
    types=scinew map<string, const TypeDescription*>;
  (*types)[td->getName()]=td;
}

TypeDescription::Register::~Register()
{
}

MPI_Datatype TypeDescription::getMPIType() const
{
   if(d_mpitype == -1){
      if(d_mpitypemaker){
	 d_mpitype = (*d_mpitypemaker)();
      } else {
	 throw InternalError("MPI Datatype requested, but do not know how to make it");
      }
   }
   ASSERT(d_mpitype != -2);
   return d_mpitype;
}

Variable* TypeDescription::createInstance() const
{
   if(!d_maker)
      throw InternalError("Do not know how to create instance for type: "+getName());
   return (*d_maker)();
}



