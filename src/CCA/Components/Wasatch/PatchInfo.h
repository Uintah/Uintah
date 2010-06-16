#ifndef Wasatch_PatchInfo_h
#define Wasatch_PatchInfo_h

#include <map>

#include <Core/Grid/Variables/VarLabel.h>

namespace SpatialOps{ class OperatorDatabase; }

namespace Wasatch{

  /**
   *  \struct PatchInfo
   *  \author James C. Sutherland
   *  \date   June, 2010
   *
   *  \brief Holds information about a patch.  This is useful for
   *  individual nodes in a graph so that they have access to
   *  operators, etc.
   */
  struct PatchInfo
  {
    SpatialOps::OperatorDatabase* operators;
    int patchID;
  };

  typedef std::map< int, PatchInfo > PatchInfoMap;

}

#endif // Wasatch_PatchInfo_h
