
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace SCIRun;

using std::string;
using std::ostringstream;

map<string, VarLabel*> VarLabel::allLabels;
string VarLabel::defaultCompressionMode = "none";

VarLabel::VarLabel(const std::string& name, const TypeDescription* td,
		   VarType vartype)
   : d_name(name), d_td(td), d_vartype(vartype),
     d_compressionMode("default"), d_allowMultipleComputes(false) 
{
   map<string, VarLabel*>::value_type mappair(name, this);
   if (allLabels.insert(mappair).second == false) {
      // two labels with the same name -- make sure they are the same type
      VarLabel* dup = allLabels[name];
      if (d_td != dup->d_td || d_vartype != dup->d_vartype)
	 throw InternalError(string("VarLabel with same name exists, '")
			     + name + "', but with different type");
   }
}

VarLabel::~VarLabel()
{
   allLabels.erase(d_name);
}

VarLabel* VarLabel::find(string name)
{
   map<string, VarLabel*>::iterator found = allLabels.find(name);
   if (found == allLabels.end())
      return NULL;
   else
      return (*found).second;
}


string
VarLabel::getFullName(int matlIndex, const Patch* patch) const
{
   ostringstream out;
        out << d_name << "(matl=" << matlIndex;
   if(patch)
        out << ", patch=" << patch->getID();
   else
        out << ", no patch";
   out << ")";
        return out.str();
}                             

void VarLabel::allowMultipleComputes()
{
   if (!d_td->isReductionVariable())
      throw InternalError(string("Only reduction variables may allow multiple computes.\n'" + d_name + "' is not a reduction variable."));
   d_allowMultipleComputes = true;
}

ostream & 
operator<<( ostream & out, const Uintah::VarLabel & vl )
{
  out << vl.getName();
  return out;
}

