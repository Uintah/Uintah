
#ifndef __CONTACT_H__
#define __CONTACT_H__

#ifdef WONT_COMPILE_YET

class Contact {
public:
  // Basic contact methods
  virtual void exMomInterpolated(const Region* region,
                                 const DataWarehouseP& old_dw,
                                 DataWarehouseP& new_dw) = 0;

  virtual void exMomIntegrated(const Region* region,
                               const DataWarehouseP& old_dw,
                               DataWarehouseP& new_dw) = 0;


  // Auxilliary methods to supply data needed by some of the
  // advanced contact models
  void computeSurfaceNormals();
  void computeTraction();
};

inline bool compare(double num1, double num2)
{
  double EPSILON=1.e-8;

  return (fabs(num1-num2) <= EPSILON);
}

#endif

#endif __CONTACT_H__

// $Log$
// Revision 1.2  2000/03/20 17:17:12  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.1  2000/03/16 01:05:13  guilkey
// Initial commit for Contact base class, as well as a NullContact
// class and SingleVel, a class which reclaims the single velocity
// field result from a multiple velocity field problem.
//
