
#include <Uintah/Grid/TypeDescription.h>
#include <map>
#include <iostream>

using namespace Uintah;
using namespace std;

static map<string, const TypeDescription*>* types;

TypeDescription::TypeDescription(Type type, const std::string& name,
				 bool isFlat)
   : d_type(type), d_name(name), d_isFlat(isFlat), d_subtype(0)
{
}

TypeDescription::TypeDescription(Type type, const std::string& name,
				 const TypeDescription* subtype)
   : d_type(type), d_name(name), d_isFlat(false), d_subtype(subtype)
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
   map<string, const TypeDescription*>::iterator iter = types->find(t);
   if(iter == types->end())
      return 0;
   return iter->second;
}

TypeDescription::Register::Register(const TypeDescription* td)
{
   //cerr << "Register: td=" << td << ", name=" << td->getName() << '\n';
   if(!types)
     types=new map<string, const TypeDescription*>;
   (*types)[td->getName()]=td;
}

TypeDescription::Register::~Register()
{
}

//
// $Log$
// Revision 1.5  2000/05/21 08:19:09  sparker
// Implement NCVariable read
// Do not fail if variable type is not known
// Added misc stuff to makefiles to remove warnings
//
// Revision 1.4  2000/05/20 23:08:12  guilkey
// Fixed type database initialization.
//
// Revision 1.3  2000/05/20 08:09:28  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.2  2000/05/18 18:41:14  kuzimmer
// Added Particle to Basis enum, created Type enum with Scalar,Point,Vector,Tensor,& Other
//
// Revision 1.1  2000/05/07 06:02:13  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
//

