#ifndef __BECKER_CHECK_H__
#define __BECKER_CHECK_H__

#include "StabilityCheck.h"	
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Math/FastMatrix.h>
#include <Packages/Uintah/Core/Math/TangentModulusTensor.h>

namespace Uintah {

  ///////////////////////////////////////////////////////////////////////////
  /*! 
   \class BeckerCheck
   \brief Checks the loss of ellipticity/hyperbolicity using the 
          the determinant of a two-dimensional form of the acoustic tensor.
   \author  Biswajit Banerjee, \n
            C-SAFE and Department of Mechanical Engineering,\n
            University of Utah.\n
            Copyright (C) 2003 Container Dynamics Group\n

   If the strain is localized in a band with normal \f$ \mathbf{n} \f$, and the
   magnitude of the velocity difference across the band is \f$ \mathbf{g} \f$,
   the bifurcation condition leads to the relation \f$ R_{ij} g_{j} = 0 \f$,
   where
   \f[
     R_{ij} = M_{ikjl} n_k n_l + M_{ilkj} n_k n_l - \sigma_{ik} n_j n_k
   \f]
   \f$ M_{ijkl} \f$ are the components of the co-rotational tangent 
   modulus tensor. \n 
   \f$ \sigma_{ij} \f$ are the components of the co-rotational stress tensor.

   If \f$ \det(R_{ij}) \le 0 \f$, then \f$ g_j \f$ can be arbitrary and there
   is a possibility of strain localization.

   Let \f$ \sigma_1, \sigma_2, \sigma_3 \f$ be the principal components of
   the stress tensor.  If the band normal is assumed to have no component
   in the direction of \f$ \sigma_2 \f$, then we have
   \f[
      R_{11} = (2 M_{1111} - \sigma_1) n_1^2 + 2 M_{3131} n_3^2; R_{12} = 0; 
      R_{13} = (2 M_{1133} + 2 M_{3131} - \sigma_1) n_1 n_3
   \f]
   \f[
      R_{21} = 0; R_{22} = 2 M_{1212} + 2 M_{2323} n_3^2; R_{23} = 0
   \f]
   \f[
      R_{31} = (2 M_{3131} + 2 M_{3311} - \sigma_3) n_1 n_3;
      R_{32} = 0; R_{33} = 2 M_{3131} n_1^2 + (2 M_{3333} - \sigma_3) n_3^2 
   \f]

   Taking the determinant of \f$ \mathbf{R} \f$, dividing by \f$ 2 n_1^2 \f$
   and setting it to zero, we get the quadric equation
   \f[
       A x^4 + B x^2 + C = 0
   \f]
   where \f$ x = n_3/n_1 \f$,
   \f[
       A := ( - M_{3131} \sigma_3 + 2 M_{3131} M_{3333})
   \f]
   \f[
       B := (M_{3131} \sigma_3 - 2 M_{1133} M_{3131} + \sigma_1 M_{3131} 
             - 2 M_{1133} M_{3311} - M_{1111} \sigma_3 + 2 M_{1111} M_{3333} 
             + M_{1133} \sigma_3 - 2 M_{3131} M_{3311} - \sigma_1 M_{3333} 
             + \sigma_1 M_{3311})
   \f]
   \f[
       C := (- \sigma_1 M_{3131} + 2 M1111 M_{3131})
   \f]

    If this equation has no real roots, there is no bifurcation. \n
    If this equation has four real roots, there is localization. \n
    If this equation has two real roots, this is an unlikely situation
    and may be due to errors in data. \n
     
    Reference: Becker, R., (2002), Int. J. Solids Struct., 39, 3555-3580.
  */
  ///////////////////////////////////////////////////////////////////////////

  class BeckerCheck : public StabilityCheck {

  public:
	 
    //! Construct an object that can be used to check stability
    BeckerCheck(ProblemSpecP& ps);

    //! Destructor of stability check
    ~BeckerCheck();
	 
    /*! Check the stability and return the direction of instability.

        \return true if unstable; 
                false if stable
    */
    bool checkStability(const Matrix3& stress,
                        const Matrix3& deformRate,
                        const TangentModulusTensor& tangentModulus,
                        Vector& direction);

  private:

    // Prevent copying of this class and copy constructor
    BeckerCheck(const BeckerCheck &);
    BeckerCheck& operator=(const BeckerCheck &);

  };
} // End namespace Uintah
      
#endif  // __BECKER_CHECK_H__

