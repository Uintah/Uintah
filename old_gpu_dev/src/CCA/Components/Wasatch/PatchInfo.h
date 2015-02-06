#ifndef Wasatch_PatchInfo_h
#define Wasatch_PatchInfo_h

#include <map>

/**
 *  \file PatchInfo.h
 */

namespace SpatialOps{ class OperatorDatabase; }

namespace Wasatch{

  /**
   *  \ingroup WasatchCore
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

  /**
   *  \ingroup WasatchCore
   *
   *  Defines a map between the patch index (Uintah assigns this) and
   *  the PatchInfo object associated with the patch.  This is
   *  generally only required by Wasatch when pairing operators with
   *  their associated patch.
   */
  typedef std::map< int, PatchInfo > PatchInfoMap;

}

#endif // Wasatch_PatchInfo_h
