/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef __SUVIC_I_H__
#define __SUVIC_I_H__


#include "ViscoPlasticityModel.h"    
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*! 
    \class SuvicI
    \brief SUVIC-Ice
    \author Jonah Lee,
    \author Department of Mechanical Engineering,
    \author University of Alaska Fairbanks,
    Copyright (C) 2008 University of Alaska Fairbanks

    The yield criterion is given by
    
    \f$<(X_{ae}-R)/K>\f$
    
    where \f$X_{ae}=\sqrt{ {3/2} (S_{ij}-B_{ij})(S_{ij}-B_{ij})}\f$=effective reduced stress \n
    \f$ R\f$ is the yield stress, \f$ K \f$ is the drag stress, \f$ B_{ij}\f$ the back stress \n

    The inelastic strain rate is given by

    \f$\dot{\epsilon}^i_{ij}=A<(X_{ae}-R)/K>^N n_{ij}\exp(-Q/RT)\f$

    where \f$ n_{ij} = {3/2} (S_{ij}-B_{ij})/X_{ae}) \f$, \f$Q, R, T\f$
    are the activation energy, Universal gas constant and the absolute temperature,
    respectively.

    The integration scheme is an adaptation of 

    A Tangent Modulus Method for Rate Dependent Solids, 
    D. Peirce, C.F. Shih and A. Needleman, Computers and  Structures
    pp. 875-887, 1984

    See the following and the references therein for details: 
    Plane Strain Indentation of Snow At The Microscale
    Jonah H. Lee, Proceedings of 10th International Conference on Advanced
    Vehicle and Tire Technologies, Paper# DETC2008-49374,  August 2008, ASME,
  */  
  /////////////////////////////////////////////////////////////////////////////

  class SuvicI : public ViscoPlasticityModel {

    // Create datatype for storing model parameters
  public:
    // Create datatype for storing state variable parameters      
    struct StateVariableData {
      double ba1;     /*< coefficient of backstress evol [MPa]  */
      double bq;      /*< exponent of backstress evol */
      double bc;     /*< normalizing back stress [MPa] */
      double b0;     /*< coeff of saturation backstress [MPa] */
      double de0;     /*< ref strain rate [1/sec] */
      double bn;     /*< exponent of backstress */

      double a;      /*< normalizing inelastic strain rate [1/sec] */
      double q;      /*< activation energy [J/Mole] */
      double t;      /*< temperature [K]; TODO- where defined elsewhere*/
      double xn;     /*< exponent of inelastic strain rate [1/sec] */

      double r0;     /*< coef of yield stress saturation [MPa] */
      double rm;     /*< exponent of yield stress */
      double rai;    /*< A3 [MPa] */

      double xmn;    /*< exponent in K [MPa */
      double xai;    /*< A5 [MPa] */
      double rr;      /*< R Universal Gas Constant [8.3144J/(mole . K)] */
      
      double s0;     /*< coeff of saturation of stress [MPa]*/
      
      double initial_yield; /* initial yield stress [MPa] */
      double initial_drag;  /*initial drag stress - never zero! [MPa]*/

      double theta; /*< plasticity integration from 0 (explict) to 1 (implicit) */
    };

    constParticleVariable<double> pYield; //scalar yield stress
    ParticleVariable<double> pYield_new;
    constParticleVariable<double> pDrag; //scalar drag stress
    ParticleVariable<double> pDrag_new;
    constParticleVariable<Matrix3> pBackStress; //tensorial back stress
    ParticleVariable<Matrix3> pBackStress_new;

    const VarLabel* pYieldLabel;  
    const VarLabel* pYieldLabel_preReloc;
    const VarLabel* pDragLabel;  
    const VarLabel* pDragLabel_preReloc;
    const VarLabel* pBackStressLabel;  
    const VarLabel* pBackStressLabel_preReloc;


  private:

        StateVariableData d_SV;
         
    // Prevent copying of this class
    // copy constructor
    //SuvicI(const SuvicI &cm);
    SuvicI& operator=(const SuvicI &cm);

  public:
    // constructors
    SuvicI(ProblemSpecP& ps);
    SuvicI(const SuvicI* cm);
         
    // destructor 
    virtual ~SuvicI();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    // Computes and requires for internal evolution variables
    // Three internal variables for ISUVIC-I :: yield, drag and back stress + plastic strain
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        bool recurse) const;

    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb) const;

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* addset,
                                   map<const VarLabel*, 
                                     ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw);


    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);

    virtual void initializeInternalVars(ParticleSubset* pset,
                                        DataWarehouse* new_dw);

    virtual void getInternalVars(ParticleSubset* pset,
                                 DataWarehouse* old_dw);

    virtual void allocateAndPutInternalVars(ParticleSubset* pset,
                                            DataWarehouse* new_dw); 

    virtual void allocateAndPutRigid(ParticleSubset* pset,
                                     DataWarehouse* new_dw); 

    virtual void updateElastic(const particleIndex idx);

   

    ///////////////////////////////////////////////////////////////////////////
    /*! compute the flow stress */
    ///////////////////////////////////////////////////////////////////////////
//     virtual double computeFlowStress(const PlasticityState* state,
//                                      const double& delT,
//                                      const double& tolerance,
//                                      const MPMMaterial* matl,
//                                      const particleIndex idx);
                                     
//     virtual double computeFlowStress(const PlasticityState* state,
//                                      const double& delT,
//                                      const double& tolerance,
//                                      const MPMMaterial* matl,
//                                      const particleIndex idx,
//                                   const Matrix3 pStress);

    virtual double computeFlowStress(const particleIndex idx,
                                     const Matrix3 pStress,
                                     const Matrix3 tensorR,
                                     const int implicitFlag);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Compute the shear modulus. 
    */
    ///////////////////////////////////////////////////////////////////////////
    double computeShearModulus(const PlasticityState* state);

    ///////////////////////////////////////////////////////////////////////////
    /*!
      \brief Compute the melting temperature
    */
    ///////////////////////////////////////////////////////////////////////////
    double computeMeltingTemp(const PlasticityState* state);
            

   protected:

   void computeNij(Matrix3& nij, 
                            Matrix3& reducedEta, 
                            double& xae, 
                            const particleIndex idx,
                            const Matrix3 pStress,
                            const Matrix3 tensorR,
                            const int implicitFlag);
                            
   void computeStressIncTangent(double& epdot,
                                Matrix3& stressRate, 
                                TangentModulusTensor& Cep,
                                const double delT,
                                const particleIndex idx, 
                                const TangentModulusTensor Ce,
                                const Matrix3 tensorD,
                                const Matrix3 pStress,
                                const int implicitFlag,
                                const Matrix3 tensorR);
                                
   bool checkFailureMaxTensileStress(const Matrix3 pStress);
                                         

   };
} // End namespace Uintah

#endif  // __SUVIC_I_H__ 
