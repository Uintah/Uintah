
/*
 *  HypotenousConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Hypotenous_Constraint_h
#define SCI_project_Hypotenous_Constraint_h 1

#include <Constraints/BaseConstraint.h>


class HypotenousConstraint : public BaseConstraint {
public:
   HypotenousConstraint( const clString& name,
			 const Index numSchemes,
			 Variable* distInX, Variable* hypoInX );
    virtual ~HypotenousConstraint();
   
protected:
   virtual void Satisfy( const Index index );
};

#endif
