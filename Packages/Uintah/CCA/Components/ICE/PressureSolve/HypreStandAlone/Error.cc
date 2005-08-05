/*#############################################################################
  # Error.cc - error messages handler implementation
  ###########################################################################*/

#include "Error.h"
#include "util.h"

std::string
Error::summary(void) const
  /* Print a summary of the object's properties. */
{
  std::ostringstream out;
  out << "Run-time";
  return out.str();
}
  
void
Error::error(const std::ostringstream& msg) const
  /* Print an error message and exit. */
{
  std::cerr << summary() << " error : " << msg.str() << "\n";
  clean();
  exit(1);
}

void
Error::warning(const std::ostringstream& msg) const
  /* Print an warning message but continue running. */
{
  std::cerr << summary() << " warning : " << msg.str() << "\n";
}
