
/*
 *  PythagorasConstraint.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCI_project_Pythagoras_Constraint_h
#define SCI_project_Pythagoras_Constraint_h 1

#include <Constraints/BaseConstraint.h>


class PythagorasConstraint : public BaseConstraint {
public:
   PythagorasConstraint( const clString& name,
			 const Index numSchemes,
			 RealVariable* dist1, RealVariable* dist2,
			 RealVariable* hypo );
   virtual ~PythagorasConstraint();
   
protected:
   virtual void Satisfy( const Index index, const Scheme scheme );
};

#endif
