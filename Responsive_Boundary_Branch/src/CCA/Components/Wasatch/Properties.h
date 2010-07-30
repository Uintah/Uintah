#ifndef Wasatch_Properties_h
#define Wasatch_Properties_h

#include <Core/ProblemSpec/ProblemSpecP.h>

/** \file */

namespace Wasatch{

  class GraphHelper;

  /**
   *  \param params The parser block.  This block will be searched for
   *         one containing the <PropertyEvaluator> tag.
   *
   *  \param gh The GraphHelper object to be used when setting properties.
   */
  void setup_property_evaluation( Uintah::ProblemSpecP& params,
                                  GraphHelper& gh );

} // namespace Wasatch

#endif // Wasatch_Properties_h
