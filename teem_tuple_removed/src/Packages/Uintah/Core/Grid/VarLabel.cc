
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Mutex.h>
#include <map>
#include <iostream>
#include <sstream>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static map<string, VarLabel*> allLabels;
string VarLabel::defaultCompressionMode = "none";
static Mutex lock("VarLabel create/destroy lock");

VarLabel* VarLabel::create(const string& name,
			   const TypeDescription* td,
			   const IntVector& boundaryLayer /*= IntVector(0,0,0) */,
			   VarType vartype /*= Normal*/)
{
  VarLabel* label = 0;
  lock.lock();
  map<string, VarLabel*>::iterator iter = allLabels.find(name);
  if(iter != allLabels.end()){
    // two labels with the same name -- make sure they are the same type
    VarLabel* dup = iter->second;
    if(boundaryLayer != dup->d_boundaryLayer)
      SCI_THROW(InternalError(string("Multiple VarLabels defined with different # of boundary layers")));

#if !defined(_AIX)
    // AIX uses lib.a's, therefore the "same" var labels are different...
    // Need to look into fixing this in a better way...
    if (td != dup->d_td || vartype != dup->d_vartype)
      SCI_THROW(InternalError(string("VarLabel with same name exists, '")
			  + name + "', but with different type"));
#endif
    label = dup;
  }
  else {
    label = scinew VarLabel(name, td, boundaryLayer, vartype);
    allLabels[name]=label;
  }
  label->addReference();
  lock.unlock();  
  return label;
}

bool VarLabel::destroy(const VarLabel* label)
{
  if (label == 0) return false;
  if (label->removeReference()) {
    lock.lock();    
    map<string, VarLabel*>::iterator iter = allLabels.find(label->d_name);
    if(iter != allLabels.end() && iter->second == label)
      allLabels.erase(iter); 
    lock.unlock();
    delete label;
    return true;
  }
  return false;
}

VarLabel::VarLabel(const std::string& name, const TypeDescription* td,
		   const IntVector& boundaryLayer, VarType vartype)
  : d_name(name), d_td(td), d_boundaryLayer(boundaryLayer),
    d_vartype(vartype), d_compressionMode("default"),
    d_allowMultipleComputes(false)
{
}

VarLabel::~VarLabel()
{
}

void VarLabel::printAll()
{
  map<string, VarLabel*>::iterator iter = allLabels.begin();
  for (; iter != allLabels.end(); iter++)
    std::cerr << (*iter).second->d_name << std::endl;
}

VarLabel* VarLabel::find(string name)
{
   map<string, VarLabel*>::iterator found = allLabels.find(name);
   if (found == allLabels.end())
      return NULL;
   else
      return found->second;
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
     SCI_THROW(InternalError(string("Only reduction variables may allow multiple computes.\n'" + d_name + "' is not a reduction variable.")));
   d_allowMultipleComputes = true;
}

ostream & 
operator<<( ostream & out, const Uintah::VarLabel & vl )
{
  out << vl.getName();
  return out;
}

