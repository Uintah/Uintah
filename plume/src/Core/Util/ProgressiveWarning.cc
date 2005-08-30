#include <Core/Util/ProgressiveWarning.h>

namespace SCIRun {

ProgressiveWarning::ProgressiveWarning(std::string message, int multiplier /* =-1 */, 
                                       std::ostream& stream /* =cerr */)
{
  d_message = message;
  d_multiplier = multiplier;
  out = &stream;
  
  d_numOccurences = 0;
  d_nextOccurence = 1;
  d_warned = false;
  
}

bool ProgressiveWarning::invoke(int numTimes /* =-1*/)
{
  bool warning_printed = false;
  d_numOccurences += numTimes;
  if (d_numOccurences >= d_nextOccurence && (!d_warned || d_multiplier != -1)) {
    d_warned = true;

    if (d_multiplier != -1) {
      showWarning();
      warning_printed = true;
      while (d_nextOccurence <= d_numOccurences)
        d_nextOccurence *= d_multiplier;
    }
    else {
      (*out) << d_message << std::endl;
      (*out) << "  This message will only occur once\n";
    }
    return true;
  }
  return false;
}

void ProgressiveWarning::changeMessage(std::string message)
{
  d_message = message;
}

void ProgressiveWarning::showWarning()
{
  (*out) << d_message << std::endl;
  (*out) << "  This warning has occurred " << d_numOccurences << " times\n";
}

} // end namespace SCIRun
