#ifndef RHSTerms_h
#define RHSTerms_h

#include <map>

//-- ExprLib includes --//
#include <expression/Tag.h>

/**
 *  \enum FieldSelector
 *  \brief Use this enum to populate information in the FieldTagInfo type.
 */
enum FieldSelector{
  CONVECTIVE_FLUX_X,
  CONVECTIVE_FLUX_Y,
  CONVECTIVE_FLUX_Z,
  DIFFUSIVE_FLUX_X,
  DIFFUSIVE_FLUX_Y,
  DIFFUSIVE_FLUX_Z,
  SOURCE_TERM
};

/**
 * \todo currently we only allow one of each info type.  But there
 *       are cases where we may want multiple ones.  Example:
 *       diffusive terms in energy equation.  Expand this
 *       capability.
 */
typedef std::map< FieldSelector, Expr::Tag > FieldTagInfo; //< Defines a map to hold information on ExpressionIDs for the RHS.

#endif // RHSTerms_h
