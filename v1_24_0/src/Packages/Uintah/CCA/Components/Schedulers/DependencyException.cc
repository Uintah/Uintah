#include <Packages/Uintah/CCA/Components/Schedulers/DependencyException.h>
#include <sstream>

using namespace Uintah;
using namespace std;

DependencyException::DependencyException(const Task* task,
					 const VarLabel* label, int matlIndex,
					 const Patch* patch,
					 string has, string needs)
  : task_(task), label_(label), matlIndex_(matlIndex), patch_(patch)
{
  d_msg = makeMessage( task_, label_, matlIndex_, patch_, has, needs);
}

string DependencyException::makeMessage(const Task* task,
					const VarLabel* label, int matlIndex,
					const Patch* patch,
					string has, string needs)
{
  ostringstream str;
  str << "Task Dependency Error: " << has << " has no corresponding ";
  str << needs << " for " << label->getName();
  if (patch)
    str << " on patch " << patch->getID();
  str << " for material " << matlIndex;
  if (task != 0) {
    str << " in task " << task->getName();
  }
  str << ".";
  return str.str();
}

DependencyException::DependencyException(const DependencyException& copy)
  : task_(copy.task_),
    label_(copy.label_),
    matlIndex_(copy.matlIndex_),
    patch_(copy.patch_),
    d_msg(copy.d_msg)
{
}

const char* DependencyException::message() const
{
  return d_msg.c_str();
}

const char* DependencyException::type() const
{
  return "Uintah::Exceptions::DependencyException";
}

