#include <Packages/Uintah/CCA/Components/Schedulers/IncorrectAllocation.h>
#include <sstream>

using namespace Uintah;
using namespace std;

IncorrectAllocation::IncorrectAllocation(const VarLabel* expectedLabel,
					 const VarLabel* actualLabel)
  : expectedLabel_(expectedLabel), actualLabel_(actualLabel)
{
  d_msg = makeMessage(expectedLabel, actualLabel);
}

string IncorrectAllocation::makeMessage(const VarLabel* expectedLabel,
					const VarLabel* actualLabel)
{
  if (actualLabel == 0) {
    return string("Variable Allocation Error: putting into label ") + expectedLabel->getName() + " but not allocated for any label";
  }
  else {
    return string("Variable Allocation Error: putting into label ") + expectedLabel->getName() + " but allocated for " + actualLabel->getName();
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

