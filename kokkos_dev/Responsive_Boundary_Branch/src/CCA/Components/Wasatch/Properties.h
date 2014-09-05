#ifndef Wasatch_Properties_h
#define Wasatch_Properties_h

#include <Core/ProblemSpec/ProblemSpecP.h>

/**
 *  \file Properties.h
 *
 *  \brief Parser handling for property specification.
 */

namespace Wasatch{

  class GraphHelper;

  /**
   *  \ingroup WasatchParser
   *  \brief handles parsing for the property evaluators.
   *  \param params The parser block.  This block will be searched for
   *         one containing the \verbatim <PropertyEvaluator> \endverbatim tag.
   *  \param gh The GraphHelper object to be used when setting properties.
   */
  void setup_property_evaluation( Uintah::ProblemSpecP& params,
                                  GraphHelper& gh );

} // namespace Wasatch

#endif // Wasatch_Properties_h
