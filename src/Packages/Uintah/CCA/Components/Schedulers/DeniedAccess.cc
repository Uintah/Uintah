
#include <Packages/Uintah/CCA/Components/Schedulers/DeniedAccess.h>
#include <sstream>

using namespace Uintah;
using namespace std;

DeniedAccess::DeniedAccess(const VarLabel* label, const Task* task,
			   int matlIndex, const Patch* patch,
			   string dependency, string accessType)
  : label_(label), task_(task), matlIndex_(matlIndex), patch_(patch)
{
  ostringstream str;
  str << "Task Dependency Warning: " << task_->getName() << " has no "
      << dependency
      << " for " << label_->getName() << " to " << accessType;
  if (patch)
    str << " on patch " << patch->getID();
  str << " for material " << matlIndex << ".";
  d_msg = str.str();
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
