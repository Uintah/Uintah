
#include <Packages/Uintah/CCA/Components/Schedulers/DeniedAccess.h>
#include <sstream>

using namespace Uintah;
using namespace std;

DeniedAccess::DeniedAccess(const VarLabel* label, const Task* task,
			   int matlIndex, const Patch* patch,
			   string dependency, string accessType)
  : label_(label), task_(task), matlIndex_(matlIndex), patch_(patch)
{
  d_msg = makeMessage(label, task, matlIndex, patch, dependency, accessType);
}

string DeniedAccess::
  makeMessage(const VarLabel* label, const Task* task, int matlIndex,
	      const Patch* patch, string dependency, string accessType)
{
  ostringstream str;
  str << "Task Dependency Warning: " << task->getName() << " has no "
      << dependency
      << " for " << label->getName() << " to " << accessType;
  if (patch)
    str << " on patch " << patch->getID();
  str << " for material " << matlIndex << ".";
  return str.str();
}

DeniedAccess::DeniedAccess(const DeniedAccess& copy)
    : label_(copy.label_), task_(copy.task_), d_msg(copy.d_msg)
{
}

const char* DeniedAccess::message() const
{
    return d_msg.c_str();
}

const char* DeniedAccess::type() const
{
    return "Uintah::Exceptions::DeniedAccess";
}
