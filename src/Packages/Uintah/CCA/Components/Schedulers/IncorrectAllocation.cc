#include <Packages/Uintah/CCA/Components/Schedulers/IncorrectAllocation.h>
#include <sstream>

using namespace Uintah;
using namespace std;

IncorrectAllocation::IncorrectAllocation(const VarLabel* expectedLabel,
					 const VarLabel* actualLabel)
  : expectedLabel_(expectedLabel), actualLabel_(actualLabel)
{
  if (actualLabel_ == 0) {
    d_msg = string("Variable Allocation Warning: putting into label ") + expectedLabel_->getName() + " but not allocated for any label";
  }
  else {
    d_msg = string("Variable Allocation Warning: putting into label ") + expectedLabel_->getName() + " but allocated for " + actualLabel_->getName();
  }
}

IncorrectAllocation::IncorrectAllocation(const IncorrectAllocation& copy)
    : expectedLabel_(copy.expectedLabel_),
      actualLabel_(copy.actualLabel_),
      d_msg(copy.d_msg)
{
}

const char* IncorrectAllocation::message() const
{
    return d_msg.c_str();
}

const char* IncorrectAllocation::type() const
{
    return "Uintah::Exceptions::IncorrectAllocation";
}

