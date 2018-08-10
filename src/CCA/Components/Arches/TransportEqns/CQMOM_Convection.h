#ifndef Uintah_Component_Arches_CQMOM_Convection_h
#define Uintah_Component_Arches_CQMOM_Convection_h

#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <CCA/Components/Arches/Directives.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Patch.h>


//#define cqmom_transport_dbg

//==========================================================================

/**
 * @class Convection_CQMOM
 * @author Alex Abboud
 * @date May 2015
 *
 * @brief A class containing the functions need for calculating convective fluxes specific to CQMOM eqns, when
 *        one or more of the internal coordinates is specified as particle velocity.
 *        These equations require convection to be based on the abscissas rather than the moments directly,
 *        and will be different enough from regular scalar transport to require its own class.
 *        The various helper functions have been reused or adapted from Discretization_new.
 *
 *        The convection term is given by \f$ F_conv = G_{i+1/2} - G_{i-1/2} \f$, with G as
 *        \f$ G_{i+1/2} \eqiv G( M_{i+1/2,l} , M_{i-1/2,r} )
 *        G( M_l, M_r ) = H^+ (M_l) + H^- (M_r) with
 *        H^+ = \sum_i w_i max(u_i,0) u_i^k with k = moment order (expanded for multivariate)
 *        H^- = \sum_i w_i min(u_i,0) u_i^k \f$
 *
 *        The wall boundary conditions - used when cellType = Intrusion or WallBC are given by setting the nodes on the
 *        face fo the wall as (in x-dir) \f$ [w_\alpha U_\alpha V_\alpha W_\alpha]_{wall} =
 *        [w_\alpha/\epsilon_w -\epsilon_w * U_\alpha V_\alpha W_\alpha]_{interior} \f$ where \f$ \epsilon_w \f$ is the
 *        particle-wall restituion coefficient, with default value set to 1 as an elastic wall collision
 *
 *        For walls that are stair-stepped an assumption is made such that the flow cell next to the wall contains a
 *        digonal wall instead, and a rebound vector is calculated in that cell as 
 *        \f$ \bf{r} = -(2/3 (\bf{n} \cdot \bf{v} )\bf{n} - \bf{v} )\f$ for 3D staggered and
 *        \f$ \bf{r} = -( (\bf{n} \cdot \bf{v} )\bf{n} - \bf{v} )\f$ for 2D staggered v = incident vector, n = wall normal
 *        NOTE: These forumla assume constant grid spacing \f$ \Delta x = \Delta y = \Delta z \f$ coefficients would
 *        need to change to support non-uniform mesh, but cases are rarely run that way, if it is needed in the future then
 *        \f$ \bf{r} = -(2 (\bf{n} \cdot \bf{v} )\bf{n} - \bf{v} )\f$ for all walls where n is normalized to |n|=1
 *
 *        The first and second order convection schemes can be found in Vikas et. al. 2011
 *
 *        This is a refactor of previous code to turn it into a Uintah task which calculates the convection for
 *        every moment at the same time, this saves computational time in the interpolation of weights and abscissas
 */


namespace Uintah {
  //---------------------------------------------------------------------------
  // Builder
  class ArchesLabel;

  class CQMOM_Convection {
  
  public:
    
    CQMOM_Convection( ArchesLabel* fieldLabels );
    ~CQMOM_Convection();
    
    typedef std::vector<int> MomentVector;
    
    /** @brief Obtain parameters from input file and process */
    void problemSetup( const ProblemSpecP& params );
    
    /** @brief Schedule the convective solve */
    void sched_solveCQMOMConvection( const LevelP & level,
                                     SchedulerP   & sched,
                                     int      timesubstep);
    
    /** @brief Actuall solve the convective term for each moment equation */
    void solveCQMOMConvection( const ProcessorGroup *,
                               const PatchSubset * patches,
                               const MaterialSubset *,
                               DataWarehouse * old_dw,
                               DataWarehouse * new_dw);
    
    /** @brief Schedule the initialization of the variables */
    void sched_initializeVariables( const LevelP& level,
                                    SchedulerP& sched );
    
    /** @brief Actually initialize the variables at the begining of a time step */
    void initializeVariables( const ProcessorGroup* pc,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse * old_dw,
                              DataWarehouse * new_dw );
    
    /** @brief schedule initialization of a variable to determien wall status */
    void sched_initializeWalls( const LevelP& level,
                                SchedulerP& sched,
                                int timesubstep);
    
    /** @brief actualyl calculate variable for wall status */
    void initializeWalls( const ProcessorGroup *,
                          const PatchSubset * patches,
                          const MaterialSubset *,
                          DataWarehouse * old_dw,
                          DataWarehouse * new_dw);
    
    //---------------------------------------------------------------------------
    // Custom Data Managers
    // --------------------------------------------------------------------------
    
    //  A boolean for marking faces of the cell as touching a boundary
    struct cqFaceBoundaryBool
    {
      bool minus;
      bool plus;
    };
    
    // Stores values in a one-D line on each face for a given cell
    struct cqFaceData1D {
      //plus and minus face values for vars like area frac which are the same
      double minus;
      double plus;
      
      double minus_right; // minus face(interior)
      double minus_left;  // minus face(exterior)
      
      double plus_right;  // plus face(exterior)
      double plus_left;   // plus face(interior)
    };
    
    //---------------------------------------------------------------------------
    // Interpolant Class
    // These interpolate weights and abscissas to the face values
    // --------------------------------------------------------------------------
    
    class Interpolator {
    public:
      Interpolator(){};
      virtual ~Interpolator(){};
      
      //only flow cells touching
      virtual cqFaceData1D no_wall( const IntVector c, const IntVector coord, constCCVariable<double>& phi, constCCVariable<int>& wallInt ) = 0;
      virtual cqFaceData1D bc_no_wall( const IntVector c, const IntVector coord, constCCVariable<double>& phi, const cqFaceBoundaryBool isBoundary) = 0;
      
      //handle 2D wall
      virtual void wall2D( const IntVector c, const IntVector coord, constCCVariable<double>& w, constCCVariable<double>& uVel, constCCVariable<double>& vVel,
                           constCCVariable<double>& wVel, constCCVariable<double>& volFrac, const double epW, const std::vector<double>& wallNorm,
                           cqFaceData1D& faceWeight, cqFaceData1D& uVelFaceAbscissas, cqFaceData1D& vVelFaceAbscissas, cqFaceData1D& wVelFaceAbscissas,
                           const int wVelIndex, const std::vector<double>& epVec, const double fStick) = 0;
      virtual void bc_wall2D( const IntVector c, const IntVector coord, constCCVariable<double>& w, constCCVariable<double>& uVel, constCCVariable<double>& vVel,
                              constCCVariable<double>& wVel, constCCVariable<double>& volFrac, const double epW, const std::vector<double>& wallNorm,
                              cqFaceData1D& faceWeight, cqFaceData1D& uVelFaceAbscissas, cqFaceData1D& vVelFaceAbscissas, cqFaceData1D& wVelFaceAbscissas,
                              const int wVelIndex, const std::vector<double>& epVec, const cqFaceBoundaryBool isBoundary, const double fStick) = 0;
      
      //handle 3D wall
      virtual void wall3D( const IntVector c, const IntVector coord, constCCVariable<double>& w, constCCVariable<double>& uVel, constCCVariable<double>& vVel,
                           constCCVariable<double>& wVel, constCCVariable<double>& volFrac, const double epW, const std::vector<double>& wallNorm,
                           cqFaceData1D& faceWeight, cqFaceData1D& uVelFaceAbscissas, cqFaceData1D& vVelFaceAbscissas, cqFaceData1D& wVelFaceAbscissas,
                           const double fStick ) = 0;
      virtual void bc_wall3D( const IntVector c, const IntVector coord, constCCVariable<double>& w, constCCVariable<double>& uVel, constCCVariable<double>& vVel,
                              constCCVariable<double>& wVel, constCCVariable<double>& volFrac, const double epW, const std::vector<double>& wallNorm,
                              cqFaceData1D& faceWeight, cqFaceData1D& uVelFaceAbscissas, cqFaceData1D& vVelFaceAbscissas, cqFaceData1D& wVelFaceAbscissas,
                              const cqFaceBoundaryBool isBoundary, const double fStick) = 0;
      
      //these all handle 1D wall - (more functions, but simpler ones)
      virtual cqFaceData1D no_bc(const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                                 constCCVariable<double>& volFrac, const double epW) = 0;
      virtual cqFaceData1D no_bc_weight(const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                                         constCCVariable<double>& volFrac, const double epW, const double fStick) = 0;
      virtual cqFaceData1D no_bc_normVel( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                                         constCCVariable<double>& volFrac, const double epW) = 0;
      
      virtual cqFaceData1D with_bc( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                                    constCCVariable<double>& volFrac, const double epW, const cqFaceBoundaryBool isBoundary) = 0;
      virtual cqFaceData1D with_bc_weight( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                                           constCCVariable<double>& volFrac, const double epW, const cqFaceBoundaryBool isBoundary, const double fStick) = 0;
      virtual cqFaceData1D with_bc_normVel( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                                            constCCVariable<double>& volFrac, const double epW, const cqFaceBoundaryBool isBoundary) = 0;
    protected:
      double dotProd ( const std::vector<double>& n, const std::vector <double>& v ) {
        double dotProd = 0;
        for ( unsigned int i = 0; i < n.size(); i++ ) {
          dotProd += n[i] * v[i];
        }
        return dotProd;
      }
      
      std::vector<double> scalarMult ( const std::vector<double>& n, const double& scalar) {
        std::vector<double> mult( n.size() );
        for ( unsigned int i = 0; i < n.size(); i++ ) {
          mult[i] = scalar * n[i];
        }
        return mult;
      }
      
    };
    
    class FirstOrderInterpolant : public Interpolator {
    public:
      FirstOrderInterpolant(){};
      ~FirstOrderInterpolant(){};
      
      //3D corner walls
      void wall3D( const IntVector c, const IntVector coord, constCCVariable<double>& w, constCCVariable<double>& uVel, constCCVariable<double>& vVel,
                   constCCVariable<double>& wVel, constCCVariable<double>& volFrac, const double epW, const std::vector<double>& wallNorm,
                   cqFaceData1D& faceWeight, cqFaceData1D& uVelFaceAbscissas, cqFaceData1D& vVelFaceAbscissas, cqFaceData1D& wVelFaceAbscissas,
                   const double fStick)
      {
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        
        faceWeight.minus_right = w[c]; faceWeight.plus_left = w[c];
        uVelFaceAbscissas.minus_right = uVel[c]; uVelFaceAbscissas.plus_left = uVel[c];
        vVelFaceAbscissas.minus_right = vVel[c]; vVelFaceAbscissas.plus_left = vVel[c];
        wVelFaceAbscissas.minus_right = wVel[c]; wVelFaceAbscissas.plus_left = wVel[c];
        
        if ( volFrac[cxm] == 1.0 ) {
          faceWeight.minus_left = w[cxm];
          uVelFaceAbscissas.minus_left = uVel[cxm];
          vVelFaceAbscissas.minus_left = vVel[cxm];
          wVelFaceAbscissas.minus_left = wVel[cxm];
        } else { //calculate rebound vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c]; incident[2] = wVel[c];
          
          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot*2.0/3.0 );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          rebound[0] = temp[0] * -1.0 * epW;
          rebound[1] = temp[1] * -1.0 * epW;
          rebound[2] = temp[2] * -1.0 * epW;
          
          faceWeight.minus_left = (1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.minus_left = rebound[0];
          vVelFaceAbscissas.minus_left = rebound[1];
          wVelFaceAbscissas.minus_left = rebound[2];
        }
        
        if ( volFrac[cxp] == 1.0 ) {
          faceWeight.plus_right = w[cxp];
          uVelFaceAbscissas.plus_right = uVel[cxp];
          vVelFaceAbscissas.plus_right = vVel[cxp];
          wVelFaceAbscissas.plus_right = wVel[cxp];
        } else { //calculate reboudn vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c]; incident[2] = wVel[c];
          
          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot*2.0/3.0 );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          rebound[0] = temp[0] * -1.0 * epW;
          rebound[1] = temp[1] * -1.0 * epW;
          rebound[2] = temp[2] * -1.0 * epW;
          
          faceWeight.plus_right = (1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.plus_right = rebound[0];
          vVelFaceAbscissas.plus_right = rebound[1];
          wVelFaceAbscissas.plus_right = rebound[2];
        }
      }
      
      //2D corner walls
      void wall2D( const IntVector c, const IntVector coord, constCCVariable<double>& w, constCCVariable<double>& uVel, constCCVariable<double>& vVel,
                   constCCVariable<double>& wVel, constCCVariable<double>& volFrac, const double epW, const std::vector<double>& wallNorm,
                   cqFaceData1D& faceWeight, cqFaceData1D& uVelFaceAbscissas, cqFaceData1D& vVelFaceAbscissas, cqFaceData1D& wVelFaceAbscissas,
                   const int wVelIndex, const std::vector<double>& epVec, const double fStick)
      {
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        
        faceWeight.minus_right = w[c]; faceWeight.plus_left = w[c];
        uVelFaceAbscissas.minus_right = uVel[c]; uVelFaceAbscissas.plus_left = uVel[c];
        vVelFaceAbscissas.minus_right = vVel[c]; vVelFaceAbscissas.plus_left = vVel[c];
        if (wVelIndex > -1) {
          wVelFaceAbscissas.minus_right = wVel[c];
          wVelFaceAbscissas.plus_left = wVel[c];
        }
        
        if ( volFrac[cxm] == 1.0 ) {
          faceWeight.minus_left = w[cxm];
          uVelFaceAbscissas.minus_left = uVel[cxm];
          vVelFaceAbscissas.minus_left = vVel[cxm];
          if (wVelIndex > -1)
            wVelFaceAbscissas.minus_left = wVel[cxm];
        } else { //calculate rebound vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c];
          //check for wvel index to allow testign in 2D x-y space easily
          incident[2] = (wVelIndex > -1 ) ? wVel[c] : 0.0;
          
          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          temp[0] *= epVec[0]; temp[1] *= epVec[1]; temp[2] *= epVec[2];
          rebound[0] = temp[0] * -1.0;
          rebound[1] = temp[1] * -1.0;
          rebound[2] = temp[2] * -1.0;
          
          faceWeight.minus_left = (1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.minus_left = rebound[0];
          vVelFaceAbscissas.minus_left = rebound[1];
          if (wVelIndex > -1)
            wVelFaceAbscissas.minus_left = rebound[2];
        }
        
        if ( volFrac[cxp] == 1.0 ) {
          faceWeight.plus_right = w[cxp];
          uVelFaceAbscissas.plus_right = uVel[cxp];
          vVelFaceAbscissas.plus_right = vVel[cxp];
          if (wVelIndex > -1)
            wVelFaceAbscissas.plus_right = wVel[cxp];
        } else { //calculate reboudn vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c];
          //check for wvel index to allow testign in 2D x-y space easily
          incident[2] = (wVelIndex > -1 ) ? wVel[c] : 0.0;
          
          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          temp[0] *= epVec[0]; temp[1] *= epVec[1]; temp[2] *= epVec[2];
          rebound[0] = temp[0] * -1.0;
          rebound[1] = temp[1] * -1.0;
          rebound[2] = temp[2] * -1.0;
          
          faceWeight.plus_right = (1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.plus_right = rebound[0];
          vVelFaceAbscissas.plus_right = rebound[1];
          if (wVelIndex > -1)
            wVelFaceAbscissas.plus_right = rebound[2];
        }
      }

      //simplest form when no wall cells touching
      cqFaceData1D no_wall( const IntVector c, const IntVector coord, constCCVariable<double>& phi, constCCVariable<int>& wallInt )
      {
        CQMOM_Convection::cqFaceData1D face_values;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        
        double phic = phi[c];
        face_values.minus_right =  phic;
        face_values.plus_left  =  phic;
        
        face_values.minus_left = phi[cxm];
        face_values.plus_right = phi[cxp];

        return face_values;
      }
      
      //breakdown of flat walls for each IC
      cqFaceData1D no_bc( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                          constCCVariable<double>& volFrac, const double epW)
      {
        CQMOM_Convection::cqFaceData1D face_values;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        
        double phic = phi[c];
        face_values.minus_right =  phic;
        face_values.plus_left  =  phic;
        
        if ( volFrac[cxm] == 1.0 || volFrac[c] == 0.0 ) {
          face_values.minus_left = phi[cxm];
        } else {
          face_values.minus_left = phic;
        }
        
        if ( volFrac[cxp] == 1.0 || volFrac[c] == 0.0 ) {
          face_values.plus_right = phi[cxp];
        } else {
          face_values.plus_right = phic;
        }
        
        return face_values;
      }
      
      cqFaceData1D no_bc_weight( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                                 constCCVariable<double>& volFrac, const double epW, const double fStick)
      {
        CQMOM_Convection::cqFaceData1D face_values;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        
        double phic = phi[c];
        face_values.minus_right =  phic;
        face_values.plus_left  =  phic;
        
        if ( volFrac[cxm] == 1.0 || volFrac[c] == 0.0 ) { //check if this is a flow cell has a wall touching it
          face_values.minus_left = phi[cxm];
        } else {
          face_values.minus_left = (1.0 - fStick) * phic/epW;
        }
        
        if ( volFrac[cxp] == 1.0 || volFrac[c] == 0.0 ) {
          face_values.plus_right = phi[cxp];
        } else {
          face_values.plus_right = (1.0 - fStick) * phic/epW;
        }
        
        return face_values;
      }
      
      cqFaceData1D no_bc_normVel( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                                  constCCVariable<double>& volFrac, const double epW)
      {
        CQMOM_Convection::cqFaceData1D face_values;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        
        double phic = phi[c];
        face_values.minus_right =  phic;
        face_values.plus_left  =  phic;
        
        if ( volFrac[cxm] == 1.0 || volFrac[c] == 0.0 ) { //check if this is a flow cell has a wall touching it
          face_values.minus_left = phi[cxm];
        } else {
          face_values.minus_left = -epW*phic;
        }
        
        if ( volFrac[cxp] == 1.0 || volFrac[c] == 0.0 ) {
          face_values.plus_right = phi[cxp];
        } else {
          face_values.plus_right = -epW*phic;
        }
        
        return face_values;
      }
      
      //3D corner walls
      void bc_wall3D( const IntVector c, const IntVector coord, constCCVariable<double>& w, constCCVariable<double>& uVel, constCCVariable<double>& vVel,
                      constCCVariable<double>& wVel, constCCVariable<double>& volFrac, const double epW, const std::vector<double>& wallNorm,
                      cqFaceData1D& faceWeight, cqFaceData1D& uVelFaceAbscissas, cqFaceData1D& vVelFaceAbscissas, cqFaceData1D& wVelFaceAbscissas,
                      const cqFaceBoundaryBool isBoundary, const double fStick )
      {
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        
        faceWeight.minus_right = w[c]; faceWeight.plus_left = w[c];
        uVelFaceAbscissas.minus_right = uVel[c]; uVelFaceAbscissas.plus_left = uVel[c];
        vVelFaceAbscissas.minus_right = vVel[c]; vVelFaceAbscissas.plus_left = vVel[c];
        wVelFaceAbscissas.minus_right = wVel[c]; wVelFaceAbscissas.plus_left = wVel[c];
        
        if ( volFrac[cxm] == 1.0 ) {
          faceWeight.minus_left = w[cxm];
          uVelFaceAbscissas.minus_left = uVel[cxm];
          vVelFaceAbscissas.minus_left = vVel[cxm];
          wVelFaceAbscissas.minus_left = wVel[cxm];
        } else { //calculate rebound vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c]; incident[2] = wVel[c];
          
          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot*2.0/3.0 );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          rebound[0] = temp[0] * -1.0 * epW;
          rebound[1] = temp[1] * -1.0 * epW;
          rebound[2] = temp[2] * -1.0 * epW;
          
          faceWeight.minus_left = (1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.minus_left = rebound[0];
          vVelFaceAbscissas.minus_left = rebound[1];
          wVelFaceAbscissas.minus_left = rebound[2];
        }
        
        if ( volFrac[cxp] == 1.0 ) {
          faceWeight.plus_right = w[cxp];
          uVelFaceAbscissas.plus_right = uVel[cxp];
          vVelFaceAbscissas.plus_right = vVel[cxp];
          wVelFaceAbscissas.plus_right = wVel[cxp];
        } else { //calculate reboudn vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c]; incident[2] = wVel[c];
          
          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot*2.0/3.0 );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          rebound[0] = temp[0] * -1.0 * epW;
          rebound[1] = temp[1] * -1.0 * epW;
          rebound[2] = temp[2] * -1.0 * epW;
          
          faceWeight.plus_right = (1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.plus_right = rebound[0];
          vVelFaceAbscissas.plus_right = rebound[1];
          wVelFaceAbscissas.plus_right = rebound[2];
        }
      }
      
      //2D corner walls
      void bc_wall2D( const IntVector c, const IntVector coord, constCCVariable<double>& w, constCCVariable<double>& uVel, constCCVariable<double>& vVel,
                      constCCVariable<double>& wVel, constCCVariable<double>& volFrac, const double epW, const std::vector<double>& wallNorm,
                      cqFaceData1D& faceWeight, cqFaceData1D& uVelFaceAbscissas, cqFaceData1D& vVelFaceAbscissas, cqFaceData1D& wVelFaceAbscissas,
                      const int wVelIndex, const std::vector<double>& epVec, const cqFaceBoundaryBool isBoundary, const double fStick )
      {
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        
        faceWeight.minus_right = w[c]; faceWeight.plus_left = w[c];
        uVelFaceAbscissas.minus_right = uVel[c]; uVelFaceAbscissas.plus_left = uVel[c];
        vVelFaceAbscissas.minus_right = vVel[c]; vVelFaceAbscissas.plus_left = vVel[c];
        if (wVelIndex > -1) {
          wVelFaceAbscissas.minus_right = wVel[c];
          wVelFaceAbscissas.plus_left = wVel[c];
        }
        
        if ( volFrac[cxm] == 1.0 ) {
          faceWeight.minus_left = w[cxm];
          uVelFaceAbscissas.minus_left = uVel[cxm];
          vVelFaceAbscissas.minus_left = vVel[cxm];
          if (wVelIndex > -1)
            wVelFaceAbscissas.minus_left = wVel[cxm];
        } else { //calculate rebound vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c];
          //check for wvel index to allow testign in 2D x-y space easily
          incident[2] = (wVelIndex > -1 ) ? wVel[c] : 0.0;
          
          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          temp[0] *= epVec[0]; temp[1] *= epVec[1]; temp[2] *= epVec[2];
          rebound[0] = temp[0] * -1.0;
          rebound[1] = temp[1] * -1.0;
          rebound[2] = temp[2] * -1.0;
          
          faceWeight.minus_left = (1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.minus_left = rebound[0];
          vVelFaceAbscissas.minus_left = rebound[1];
          if (wVelIndex > -1)
            wVelFaceAbscissas.minus_left = rebound[2];
        }
        
        if ( volFrac[cxp] == 1.0 ) {
          faceWeight.plus_right = w[cxp];
          uVelFaceAbscissas.plus_right = uVel[cxp];
          vVelFaceAbscissas.plus_right = vVel[cxp];
          if (wVelIndex > -1)
            wVelFaceAbscissas.plus_right = wVel[cxp];
        } else { //calculate reboudn vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c];
          //check for wvel index to allow testign in 2D x-y space easily
          incident[2] = (wVelIndex > -1 ) ? wVel[c] : 0.0;
          
          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          temp[0] *= epVec[0]; temp[1] *= epVec[1]; temp[2] *= epVec[2];
          rebound[0] = temp[0] * -1.0;
          rebound[1] = temp[1] * -1.0;
          rebound[2] = temp[2] * -1.0;
          
          faceWeight.plus_right = (1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.plus_right = rebound[0];
          vVelFaceAbscissas.plus_right = rebound[1];
          if (wVelIndex > -1)
            wVelFaceAbscissas.plus_right = rebound[2];
        }
      }

      //simplest form no walls touching this cell
      cqFaceData1D bc_no_wall( const IntVector c, const IntVector coord, constCCVariable<double>& phi, const cqFaceBoundaryBool isBoundary )
      {
        CQMOM_Convection::cqFaceData1D face_values;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        
        double phic = phi[c];
        face_values.minus_right =  phic;
        face_values.plus_left  =  phic;
        
        face_values.minus_left = phi[cxm];
        face_values.plus_right = phi[cxp];
        
        return face_values;
      }
      
      cqFaceData1D with_bc( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                            constCCVariable<double>& volFrac, const double epW, const cqFaceBoundaryBool isBoundary)
      {
        //For first order treat boundary cells exactly how interior cells are treated
        CQMOM_Convection::cqFaceData1D face_values;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        
        double phic = phi[c];
        face_values.minus_right =  phic;
        face_values.plus_left  =  phic;
        
        if ( volFrac[cxm] == 1.0 || volFrac[c] == 0.0 ) { //check if this is a flow cell has a wall touching it
          face_values.minus_left = phi[cxm];
        } else {
          face_values.minus_left = phic;
        }
        
        if ( volFrac[cxp] == 1.0 || volFrac[c] == 0.0 ) {
          face_values.plus_right = phi[cxp];
        } else {
          face_values.plus_right = phic;
        }
        
        return face_values;
      }
      
      cqFaceData1D with_bc_weight( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                                   constCCVariable<double>& volFrac, const double epW, const cqFaceBoundaryBool isBoundary, const double fStick)
      {
        CQMOM_Convection::cqFaceData1D face_values;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        
        double phic = phi[c];
        face_values.minus_right =  phic;
        face_values.plus_left  =  phic;
        
        if ( volFrac[cxm] == 1.0 || volFrac[c] == 0.0 ) {  //check if this is a flow cell has a wall touching it
          face_values.minus_left = phi[cxm];
        } else {
          face_values.minus_left = (1.0 - fStick) * phic/epW;
        }
        
        if ( volFrac[cxp] == 1.0 || volFrac[c] == 0.0 ) {
          face_values.plus_right = phi[cxp];
        } else {
          face_values.plus_right = (1.0 - fStick) * phic/epW;
        }
        
        return face_values;
      }
      
      cqFaceData1D with_bc_normVel( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                                    constCCVariable<double>& volFrac, const double epW, const cqFaceBoundaryBool isBoundary)
      {
        CQMOM_Convection::cqFaceData1D face_values;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        
        double phic = phi[c];
        face_values.minus_right =  phic;
        face_values.plus_left  =  phic;
        
        if ( volFrac[cxm] == 1.0 || volFrac[c] == 0.0 ) { //check if this is a flow cell has a wall touching it
          face_values.minus_left = phi[cxm];
        } else {
          face_values.minus_left = -epW*phic;
        }
        
        if ( volFrac[cxp] == 1.0 || volFrac[c] == 0.0 ) {
          face_values.plus_right = phi[cxp];
        } else {
          face_values.plus_right = -epW*phic;
        }
        
        return face_values;
      }
      
    }; //end class
    
    class SecondOrderInterpolant : public Interpolator {
    public:
      SecondOrderInterpolant(){};
      ~SecondOrderInterpolant(){};

      //3D corner walls
      //NOTE: 2nd order scheme with walls has some numerical issues that can occur even in simple cases, clipping of velocities in CQMOM xml should be turned on
      void wall3D( const IntVector c, const IntVector coord, constCCVariable<double>& w, constCCVariable<double>& uVel, constCCVariable<double>& vVel,
                   constCCVariable<double>& wVel, constCCVariable<double>& volFrac, const double epW, const std::vector<double>& wallNorm,
                   cqFaceData1D& faceWeight, cqFaceData1D& uVelFaceAbscissas, cqFaceData1D& vVelFaceAbscissas, cqFaceData1D& wVelFaceAbscissas,
                   const double fStick)
      {
        IntVector cxpp = c + coord + coord;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        IntVector cxmm = c - coord - coord;
        
        double wc = w[c], wcxm = w[cxm], wcxp = w[cxp];
        double uVelc = uVel[c], uVelcxm = uVel[cxm], uVelcxp = uVel[cxp];
        double vVelc = vVel[c], vVelcxm = vVel[cxm], vVelcxp = vVel[cxp];
        double wVelc = wVel[c], wVelcxm = wVel[cxm], wVelcxp = wVel[cxp];
        
        //calculate the inside faces
        double nxm = wc - wcxm;
        double nxp = wcxp - wc;
        double delN = minMod(nxm, nxp);
        faceWeight.minus_right = wc - 1.0/2.0 * delN; faceWeight.plus_left = wc + 1.0/2.0 * delN;
        
        nxm = uVelc - uVelcxm;
        nxp = uVelcxp - uVelc;
        delN = minMod(nxm, nxp);
        uVelFaceAbscissas.minus_right = uVelc - 1.0/2.0 * delN; uVelFaceAbscissas.plus_left = uVelc + 1.0/2.0 * delN;
        
        nxm = vVelc - vVelcxm;
        nxp = vVelcxp - vVelc;
        delN = minMod(nxm, nxp);
        vVelFaceAbscissas.minus_right = vVelc - 1.0/2.0 * delN; vVelFaceAbscissas.plus_left = vVelc + 1.0/2.0 * delN;
        
        nxm = wVelc - wVelcxm;
        nxp = wVelcxp - wVelc;
        delN = minMod(nxm, nxp);
        wVelFaceAbscissas.minus_right = wVelc - 1.0/2.0 * delN; wVelFaceAbscissas.plus_left = wVelc + 1.0/2.0 * delN;
        
        if ( volFrac[cxm] == 1.0 ) {
          
          nxm = wcxm - w[cxmm];
          nxp = wc - wcxm;
          delN = minMod(nxm, nxp);
          faceWeight.minus_left = wcxm + 1.0/2.0 * delN;
          
          nxm = uVelcxm - uVel[cxmm];
          nxp = uVelc - uVelcxm;
          delN = minMod(nxm, nxp);
          uVelFaceAbscissas.minus_left = uVelcxm + 1.0/2.0 * delN;
          
          nxm = vVelcxm - vVel[cxmm];
          nxp = vVelc - vVelcxm;
          delN = minMod(nxm, nxp);
          vVelFaceAbscissas.minus_left = vVelcxm + 1.0/2.0 * delN;
          
          nxm = wVelcxm - wVel[cxmm];
          nxp = wVelc - wVelcxm;
          delN = minMod(nxm, nxp);
          wVelFaceAbscissas.minus_left = wVelcxm + 1.0/2.0 * delN;
          
        } else { //calculate rebound vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c]; incident[2] = wVel[c];
          
          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot*2.0/3.0 );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          rebound[0] = temp[0] * -1.0 * epW;
          rebound[1] = temp[1] * -1.0 * epW;
          rebound[2] = temp[2] * -1.0 * epW;
          
          faceWeight.minus_left = ( 1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.minus_left = rebound[0];
          vVelFaceAbscissas.minus_left = rebound[1];
          wVelFaceAbscissas.minus_left = rebound[2];
        }
        
        if ( volFrac[cxp] == 1.0 ) {
          
          nxm = wcxp - wc;
          nxp = w[cxpp] - wcxp;
          delN = minMod(nxm, nxp);
          faceWeight.plus_right = wcxp - 1.0/2.0 * delN;
          
          nxm = uVelcxp - uVelc;
          nxp = uVel[cxpp] - uVelcxp;
          delN = minMod(nxm, nxp);
          uVelFaceAbscissas.plus_right = uVelcxp - 1.0/2.0 * delN;
          
          nxm = vVelcxp - vVelc;
          nxp = vVel[cxpp] - vVelcxp;
          delN = minMod(nxm, nxp);
          vVelFaceAbscissas.plus_right = vVelcxp - 1.0/2.0 * delN;
          
          nxm = wVelcxp - wVelc;
          nxp = wVel[cxpp] - wVelcxp;
          delN = minMod(nxm, nxp);
          wVelFaceAbscissas.plus_right = wVelcxp - 1.0/2.0 * delN;
          
        } else { //calculate rebound vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c]; incident[2] = wVel[c];
          
          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot*2.0/3.0 );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          rebound[0] = temp[0] * -1.0 * epW;
          rebound[1] = temp[1] * -1.0 * epW;
          rebound[2] = temp[2] * -1.0 * epW;
          
          faceWeight.plus_right = (1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.plus_right = rebound[0];
          vVelFaceAbscissas.plus_right = rebound[1];
          wVelFaceAbscissas.plus_right = rebound[2];
        }
      }

      //2D corner walls
      void wall2D( const IntVector c, const IntVector coord, constCCVariable<double>& w, constCCVariable<double>& uVel, constCCVariable<double>& vVel,
                   constCCVariable<double>& wVel, constCCVariable<double>& volFrac, const double epW, const std::vector<double>& wallNorm,
                   cqFaceData1D& faceWeight, cqFaceData1D& uVelFaceAbscissas, cqFaceData1D& vVelFaceAbscissas, cqFaceData1D& wVelFaceAbscissas,
                   const int wVelIndex, const std::vector<double>& epVec, const double fStick)
      {
        IntVector cxpp = c + coord + coord;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        IntVector cxmm = c - coord - coord;
        
        double wc = w[c], wcxm = w[cxm], wcxp = w[cxp];
        double uVelc = uVel[c], uVelcxm = uVel[cxm], uVelcxp = uVel[cxp];
        double vVelc = vVel[c], vVelcxm = vVel[cxm], vVelcxp = vVel[cxp];
        double wVelc, wVelcxm, wVelcxp;
        if (wVelIndex > -1 ) {
          wVelc = wVel[c];
          wVelcxm = wVel[cxm];
          wVelcxp = wVel[cxp];
        }
        
        //calculate the inside faces
        double nxm = wc - wcxm;
        double nxp = wcxp - wc;
        double delN = minMod(nxm, nxp);
        faceWeight.minus_right = wc - 1.0/2.0 * delN; faceWeight.plus_left = wc + 1.0/2.0 * delN;
        
        nxm = uVelc - uVelcxm;
        nxp = uVelcxp - uVelc;
        delN = minMod(nxm, nxp);
        uVelFaceAbscissas.minus_right = uVelc - 1.0/2.0 * delN; uVelFaceAbscissas.plus_left = uVelc + 1.0/2.0 * delN;
        
        nxm = vVelc - vVelcxm;
        nxp = vVelcxp - vVelc;
        delN = minMod(nxm, nxp);
        vVelFaceAbscissas.minus_right = vVelc - 1.0/2.0 * delN; vVelFaceAbscissas.plus_left = vVelc + 1.0/2.0 * delN;
        
        if (wVelIndex > -1) {
          nxm = wVelc - wVelcxm;
          nxp = wVelcxp - wVelc;
          delN = minMod(nxm, nxp);
          wVelFaceAbscissas.minus_right = wVelc - 1.0/2.0 * delN; wVelFaceAbscissas.plus_left = wVelc + 1.0/2.0 * delN;
        }
        
        if ( volFrac[cxm] == 1.0 ) {
          
          nxm = wcxm - w[cxmm];
          nxp = wc - wcxm;
          delN = minMod(nxm, nxp);
          faceWeight.minus_left = wcxm + 1.0/2.0 * delN;
          
          nxm = uVelcxm - uVel[cxmm];
          nxp = uVelc - uVelcxm;
          delN = minMod(nxm, nxp);
          uVelFaceAbscissas.minus_left = uVelcxm + 1.0/2.0 * delN;
          
          nxm = vVelcxm - vVel[cxmm];
          nxp = vVelc - vVelcxm;
          delN = minMod(nxm, nxp);
          vVelFaceAbscissas.minus_left = vVelcxm + 1.0/2.0 * delN;
          
          if (wVelIndex > -1) {
            nxm = wVelcxm - wVel[cxmm];
            nxp = wVelc - wVelcxm;
            delN = minMod(nxm, nxp);
            wVelFaceAbscissas.minus_left = wVelcxm + 1.0/2.0 * delN;
          }

        } else { //calculate rebound vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c];
          //check for wvel index to allow testign in 2D x-y space easily
          incident[2] = (wVelIndex > -1 ) ? wVel[c] : 0.0;
          
          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          temp[0] *= epVec[0]; temp[1] *= epVec[1]; temp[2] *= epVec[2];
          rebound[0] = temp[0] * -1.0;
          rebound[1] = temp[1] * -1.0;
          rebound[2] = temp[2] * -1.0;
          
          faceWeight.minus_left = (1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.minus_left = rebound[0];
          vVelFaceAbscissas.minus_left = rebound[1];
          if (wVelIndex > -1)
            wVelFaceAbscissas.minus_left = rebound[2];
        }
        
        if ( volFrac[cxp] == 1.0 ) {
          
          nxm = wcxp - wc;
          nxp = w[cxpp] - wcxp;
          delN = minMod(nxm, nxp);
          faceWeight.plus_right = wcxp - 1.0/2.0 * delN;
          
          nxm = uVelcxp - uVelc;
          nxp = uVel[cxpp] - uVelcxp;
          delN = minMod(nxm, nxp);
          uVelFaceAbscissas.plus_right = uVelcxp - 1.0/2.0 * delN;
          
          nxm = vVelcxp - vVelc;
          nxp = vVel[cxpp] - vVelcxp;
          delN = minMod(nxm, nxp);
          vVelFaceAbscissas.plus_right = vVelcxp - 1.0/2.0 * delN;
          
          if (wVelIndex > -1) {
            nxm = wVelcxp - wVelc;
            nxp = wVel[cxpp] - wVelcxp;
            delN = minMod(nxm, nxp);
            wVelFaceAbscissas.plus_right = wVelcxp - 1.0/2.0 * delN;
          }
          
        } else { //calculate rebound vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c];
          //check for wvel index to allow testign in 2D x-y space easily
          incident[2] = (wVelIndex > -1 ) ? wVel[c] : 0.0;

          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          temp[0] *= epVec[0]; temp[1] *= epVec[1]; temp[2] *= epVec[2];
          rebound[0] = temp[0] * -1.0;
          rebound[1] = temp[1] * -1.0;
          rebound[2] = temp[2] * -1.0;
          
          faceWeight.plus_right = (1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.plus_right = rebound[0];
          vVelFaceAbscissas.plus_right = rebound[1];
          if (wVelIndex > -1)
            wVelFaceAbscissas.plus_right = rebound[2];
        }
      }

      //simplest form when no wall cells touching
      cqFaceData1D no_wall( const IntVector c, const IntVector coord, constCCVariable<double>& phi, constCCVariable<int>& wallInt )
      {
        CQMOM_Convection::cqFaceData1D face_values;
        
        IntVector cxpp = c + coord + coord;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        IntVector cxmm = c - coord - coord;
        
        double phic = phi[c];
        double phicxm = phi[cxm];
        double phicxp = phi[cxp];
        //calculate the inside faces
        double delN;
        double nxm = (phic - phicxm);
        double nxp = (phicxp - phic);
        delN = minMod(nxm, nxp);
      
        face_values.minus_right = phic - 1.0/2.0 * delN;
        face_values.plus_left = phic + 1.0/2.0 * delN;
       
        //calculate outside faces
        nxm = (phicxm - phi[cxmm]);
        nxp = (phic - phicxm);
        delN = minMod(nxm, nxp);
        face_values.minus_left = phicxm + 1.0/2.0 * delN;

        nxm = (phicxp - phic);
        nxp = (phi[cxpp] - phicxp);
        delN = minMod(nxm, nxp);
        face_values.plus_right = phicxp - 1.0/2.0 * delN;
        
        return face_values;
      }
      
      //functions to deal with flat wall
      cqFaceData1D no_bc( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                          constCCVariable<double>& volFrac, const double epW)
      {
        CQMOM_Convection::cqFaceData1D face_values;
        
        IntVector cxpp = c + coord + coord;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        IntVector cxmm = c - coord - coord;
        
        double phic = phi[c];
        double phicxm = phi[cxm];
        double phicxp = phi[cxp];
        //calculate the inside faces
        double delN;
        double nxm = (phic - phicxm);
        double nxp = (phicxp - phic);
        delN = minMod(nxm, nxp);
        
        face_values.minus_right = phic - 1.0/2.0 * delN;
        face_values.plus_left = phic + 1.0/2.0 * delN;
        
        if ( volFrac[cxm] == 1.0 || volFrac[c] == 0.0 ) { //check if wall is present
          nxm = (phicxm - phi[cxmm]);
          nxp = (phic - phicxm);
          delN = minMod(nxm, nxp);
          face_values.minus_left = phicxm + 1.0/2.0 * delN;
        } else {
          face_values.minus_left = face_values.minus_right;
        }
        
        if ( volFrac[cxp] == 1.0 || volFrac[c] == 0.0 ) {  //check if wall is present
          nxm = (phicxp - phic);
          nxp = (phi[cxpp] - phicxp);
          delN = minMod(nxm, nxp);
          face_values.plus_right = phicxp - 1.0/2.0 * delN;
        } else {
          face_values.plus_right = face_values.plus_left;
        }
        
#ifdef cqmom_transport_dbg
        if (phi[c] > 0.0) {
          std::cout << "Cell c " << c << std::endl;
          std::cout << "inside -r: " << face_values.minus_right << std::endl;
          std::cout << "inside +l: " << face_values.plus_left << std::endl;
          std::cout << "plus +r: " << face_values.plus_right << std::endl;
          std::cout << "minus -l: " << face_values.minus_left << std::endl;
        }
#endif
        return face_values;
      }
      
      cqFaceData1D no_bc_weight( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                                 constCCVariable<double>& volFrac, const double epW, const double fStick)
      {
        CQMOM_Convection::cqFaceData1D face_values;
        
        IntVector cxpp = c + coord + coord;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        IntVector cxmm = c - coord - coord;
        
        double phic = phi[c];
        double phicxm = phi[cxm];
        double phicxp = phi[cxp];
        //calculate the inside faces
        double delN;
        double nxm = (phic - phicxm);
        double nxp = (phicxp - phic);
        delN = minMod(nxm, nxp);
        
        face_values.minus_right = phic - 1.0/2.0 * delN;
        face_values.plus_left = phic + 1.0/2.0 * delN;
        
        if ( volFrac[cxm] == 1.0 || volFrac[c] == 0.0 ) { //check if wall is present
          nxm = (phicxm - phi[cxmm]);
          nxp = (phic - phicxm);
          delN = minMod(nxm, nxp);
          face_values.minus_left = phicxm + 1.0/2.0 * delN;
        } else {
          face_values.minus_left = (1.0 - fStick) * face_values.minus_right/epW;
        }
        
        if ( volFrac[cxp] == 1.0 || volFrac[c] == 0.0 ) { //check if wall is present
          nxm = (phicxp - phic);
          nxp = (phi[cxpp] - phicxp);
          delN = minMod(nxm, nxp);
          face_values.plus_right = phicxp - 1.0/2.0 * delN;
        } else {
          face_values.plus_right = (1.0 - fStick) * face_values.plus_left/epW;
        }
        
#ifdef cqmom_transport_dbg
        if (phi[c] > 0.0) {
          std::cout << "Cell c " << c << std::endl;
          std::cout << "inside -r: " << face_values.minus_right << std::endl;
          std::cout << "inside +l: " << face_values.plus_left << std::endl;
          std::cout << "plus +r: " << face_values.plus_right << std::endl;
          std::cout << "minus -l: " << face_values.minus_left << std::endl;
        }
#endif
        return face_values;
      }
      
      cqFaceData1D no_bc_normVel( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                                  constCCVariable<double>& volFrac, const double epW)
      {
        CQMOM_Convection::cqFaceData1D face_values;
        
        IntVector cxpp = c + coord + coord;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        IntVector cxmm = c - coord - coord;
        
        double phic = phi[c];
        double phicxm = phi[cxm];
        double phicxp = phi[cxp];
        //calculate the inside faces
        double delN;
        double nxm = (phic - phicxm);
        double nxp = (phicxp - phic);
        delN = minMod(nxm, nxp);
        
        face_values.minus_right = phic - 1.0/2.0 * delN;
        face_values.plus_left = phic + 1.0/2.0 * delN;
        
        if ( volFrac[cxm] == 1.0 || volFrac[c] == 0.0 ) { //check if wall is present
          nxm = (phicxm - phi[cxmm]);
          nxp = (phic - phicxm);
          delN = minMod(nxm, nxp);
          face_values.minus_left = phi[cxm] + 1.0/2.0 * delN;
        } else {
          face_values.minus_left = -epW*face_values.minus_right;
        }
        
        if ( volFrac[cxp] == 1.0 || volFrac[c] == 0.0 ) { //check if wall is present
          nxm = (phicxp - phic);
          nxp = (phi[cxpp] - phicxp);
          delN = minMod(nxm, nxp);
          face_values.plus_right = phicxp - 1.0/2.0 * delN;
        } else {
          face_values.plus_right = -epW*face_values.plus_left;
        }
        
#ifdef cqmom_transport_dbg
        if (phi[c] > 0.0) {
          std::cout << "Cell c " << c << std::endl;
          std::cout << "inside -r: " << face_values.minus_right << std::endl;
          std::cout << "inside +l: " << face_values.plus_left << std::endl;
          std::cout << "plus +r: " << face_values.plus_right << std::endl;
          std::cout << "minus -l: " << face_values.minus_left << std::endl;
        }
#endif
        return face_values;
      }
      
      //3D corner walls
      void bc_wall3D( const IntVector c, const IntVector coord, constCCVariable<double>& w, constCCVariable<double>& uVel, constCCVariable<double>& vVel,
                      constCCVariable<double>& wVel, constCCVariable<double>& volFrac, const double epW, const std::vector<double>& wallNorm,
                      cqFaceData1D& faceWeight, cqFaceData1D& uVelFaceAbscissas, cqFaceData1D& vVelFaceAbscissas, cqFaceData1D& wVelFaceAbscissas,
                      const cqFaceBoundaryBool isBoundary, const double fStick)
      {
        IntVector cxpp = c + coord + coord;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        IntVector cxmm = c - coord - coord;
        
        double wc = w[c], wcxm = w[cxm], wcxp = w[cxp];
        double uVelc = uVel[c], uVelcxm = uVel[cxm], uVelcxp = uVel[cxp];
        double vVelc = vVel[c], vVelcxm = vVel[cxm], vVelcxp = uVel[cxp];
        double wVelc = wVel[c], wVelcxm = wVel[cxm], wVelcxp = uVel[cxp];
        
        //calculate the inside faces
        double nxm = wc - wcxm;
        double nxp = wcxp - wc;
        double delN = minMod(nxm, nxp);
        faceWeight.minus_right = wc - 1.0/2.0 * delN; faceWeight.plus_left = wc + 1.0/2.0 * delN;
        
        nxm = uVelc - uVelcxm;
        nxp = uVelcxp - uVelc;
        delN = minMod(nxm, nxp);
        uVelFaceAbscissas.minus_right = uVelc - 1.0/2.0 * delN; uVelFaceAbscissas.plus_left = uVelc + 1.0/2.0 * delN;
        
        nxm = vVelc - vVelcxm;
        nxp = vVelcxp - vVelc;
        delN = minMod(nxm, nxp);
        vVelFaceAbscissas.minus_right = vVelc - 1.0/2.0 * delN; vVelFaceAbscissas.plus_left = vVelc + 1.0/2.0 * delN;
        
        nxm = wVelc - wVelcxm;
        nxp = wVelcxp - wVelc;
        delN = minMod(nxm, nxp);
        wVelFaceAbscissas.minus_right = wVelc - 1.0/2.0 * delN; wVelFaceAbscissas.plus_left = wVelc + 1.0/2.0 * delN;
        
        if ( volFrac[cxm] == 1.0 ) {
          if ( isBoundary.minus ) {
            faceWeight.minus_left = wcxm;
            uVelFaceAbscissas.minus_left = uVelcxm;
            vVelFaceAbscissas.minus_left = vVelcxm;
            wVelFaceAbscissas.minus_left = wVelcxm;
          } else {
            nxm = wcxm - w[cxmm];
            nxp = wc - wcxm;
            delN = minMod(nxm, nxp);
            faceWeight.minus_left = wcxm + 1.0/2.0 * delN;
           
            nxm = uVelcxm - uVel[cxmm];
            nxp = uVelc - uVelcxm;
            delN = minMod(nxm, nxp);
            uVelFaceAbscissas.minus_left = uVelcxm + 1.0/2.0 * delN;
          
            nxm = vVelcxm - vVel[cxmm];
            nxp = vVelc - vVelcxm;
            delN = minMod(nxm, nxp);
            vVelFaceAbscissas.minus_left = vVelcxm + 1.0/2.0 * delN;
          
            nxm = wVelcxm - wVel[cxmm];
            nxp = wVelc - wVelcxm;
            delN = minMod(nxm, nxp);
            wVelFaceAbscissas.minus_left = wVelcxm + 1.0/2.0 * delN;
          }
          
        } else { //calculate rebound vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c]; incident[2] = wVel[c];
          
          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot*2.0/3.0 );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          rebound[0] = temp[0] * -1.0 * epW;
          rebound[1] = temp[1] * -1.0 * epW;
          rebound[2] = temp[2] * -1.0 * epW;
          
          faceWeight.minus_left = (1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.minus_left = rebound[0];
          vVelFaceAbscissas.minus_left = rebound[1];
          wVelFaceAbscissas.minus_left = rebound[2];
        }
        
        if ( volFrac[cxp] == 1.0 ) {
          if ( isBoundary.plus ) {
            faceWeight.plus_right = wcxp;
            uVelFaceAbscissas.plus_right = uVelcxp;
            vVelFaceAbscissas.plus_right = vVelcxp;
            wVelFaceAbscissas.plus_right = wVelcxp;
          } else {
            nxm = wcxp - wc;
            nxp = w[cxpp] - wcxp;
            delN = minMod(nxm, nxp);
            faceWeight.plus_right = wcxp - 1.0/2.0 * delN;
          
            nxm = uVelcxp - uVelc;
            nxp = uVel[cxpp] - uVelcxp;
            delN = minMod(nxm, nxp);
            uVelFaceAbscissas.plus_right = uVelcxp - 1.0/2.0 * delN;
          
            nxm = vVelcxp - vVelc;
            nxp = vVel[cxpp] - vVelcxp;
            delN = minMod(nxm, nxp);
            vVelFaceAbscissas.plus_right = vVelcxp - 1.0/2.0 * delN;
          
            nxm = wVelcxp - wVelc;
            nxp = wVel[cxpp] - wVelcxp;
            delN = minMod(nxm, nxp);
            wVelFaceAbscissas.plus_right = wVelcxp - 1.0/2.0 * delN;
          }
          
        } else { //calculate rebound vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c]; incident[2] = wVel[c];
          
          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot*2.0/3.0 );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          rebound[0] = temp[0] * -1.0 * epW;
          rebound[1] = temp[1] * -1.0 * epW;
          rebound[2] = temp[2] * -1.0 * epW;
          
          faceWeight.plus_right = (1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.plus_right = rebound[0];
          vVelFaceAbscissas.plus_right = rebound[1];
          wVelFaceAbscissas.plus_right = rebound[2];
        }
      }

      //2D corner walls
      void bc_wall2D( const IntVector c, const IntVector coord, constCCVariable<double>& w, constCCVariable<double>& uVel, constCCVariable<double>& vVel,
                      constCCVariable<double>& wVel, constCCVariable<double>& volFrac, const double epW, const std::vector<double>& wallNorm,
                      cqFaceData1D& faceWeight, cqFaceData1D& uVelFaceAbscissas, cqFaceData1D& vVelFaceAbscissas, cqFaceData1D& wVelFaceAbscissas,
                      const int wVelIndex, const std::vector<double>& epVec, const cqFaceBoundaryBool isBoundary, const double fStick)
      {
        IntVector cxpp = c + coord + coord;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        IntVector cxmm = c - coord - coord;
        
        double wc = w[c], wcxm = w[cxm], wcxp = w[cxp];
        double uVelc = uVel[c], uVelcxm = uVel[cxm], uVelcxp = uVel[cxp];
        double vVelc = vVel[c], vVelcxm = vVel[cxm], vVelcxp = uVel[cxp];
        double wVelc, wVelcxm, wVelcxp;
        if (wVelIndex > -1) {
          wVelc = wVel[c];
          wVelcxm = wVel[cxm];
          wVelcxp = uVel[cxp];
        }
        
        //calculate the inside faces
        double nxm = wc - wcxm;
        double nxp = wcxp - wc;
        double delN = minMod(nxm, nxp);
        faceWeight.minus_right = wc - 1.0/2.0 * delN; faceWeight.plus_left = wc + 1.0/2.0 * delN;
        
        nxm = uVelc - uVelcxm;
        nxp = uVelcxp - uVelc;
        delN = minMod(nxm, nxp);
        uVelFaceAbscissas.minus_right = uVelc - 1.0/2.0 * delN; uVelFaceAbscissas.plus_left = uVelc + 1.0/2.0 * delN;
        
        nxm = vVelc - vVelcxm;
        nxp = vVelcxp - vVelc;
        delN = minMod(nxm, nxp);
        vVelFaceAbscissas.minus_right = vVelc - 1.0/2.0 * delN; vVelFaceAbscissas.plus_left = vVelc + 1.0/2.0 * delN;
        
        if (wVelIndex > -1) {
          nxm = wVelc - wVelcxm;
          nxp = wVelcxp - wVelc;
          delN = minMod(nxm, nxp);
          wVelFaceAbscissas.minus_right = wVelc - 1.0/2.0 * delN; wVelFaceAbscissas.plus_left = wVelc + 1.0/2.0 * delN;
        }
        
        if ( volFrac[cxm] == 1.0 ) {
          if ( isBoundary.minus ) {
            faceWeight.minus_left = wcxm;
            uVelFaceAbscissas.minus_left = uVelcxm;
            vVelFaceAbscissas.minus_left = vVelcxm;
            if (wVelIndex > -1)
              wVelFaceAbscissas.minus_left = wVelcxm;
          } else {
            nxm = wcxm - w[cxmm];
            nxp = wc - wcxm;
            delN = minMod(nxm, nxp);
            faceWeight.minus_left = wcxm + 1.0/2.0 * delN;
          
            nxm = uVelcxm - uVel[cxmm];
            nxp = uVelc - uVelcxm;
            delN = minMod(nxm, nxp);
            uVelFaceAbscissas.minus_left = uVelcxm + 1.0/2.0 * delN;
          
            nxm = vVelcxm - vVel[cxmm];
            nxp = vVelc - vVelcxm;
            delN = minMod(nxm, nxp);
            vVelFaceAbscissas.minus_left = vVelcxm + 1.0/2.0 * delN;
          
            if (wVelIndex > -1) {
              nxm = wVelcxm - wVel[cxmm];
              nxp = wVelc - wVelcxm;
              delN = minMod(nxm, nxp);
              wVelFaceAbscissas.minus_left = wVelcxm + 1.0/2.0 * delN;
            }
          }
          
        } else { //calculate rebound vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c];
          //check for wvel index to allow testign in 2D x-y space easily
          incident[2] = (wVelIndex > -1 ) ? wVel[c] : 0.0;
          
          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          temp[0] *= epVec[0]; temp[1] *= epVec[1]; temp[2] *= epVec[2];
          rebound[0] = temp[0] * -1.0;
          rebound[1] = temp[1] * -1.0;
          rebound[2] = temp[2] * -1.0;
          
          faceWeight.minus_left = (1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.minus_left = rebound[0];
          vVelFaceAbscissas.minus_left = rebound[1];
          if (wVelIndex > -1)
            wVelFaceAbscissas.minus_left = rebound[2];
        }
        
        if ( volFrac[cxp] == 1.0 ) {
          if ( isBoundary.plus ) {
            faceWeight.plus_right = wcxp;
            uVelFaceAbscissas.plus_right = uVelcxp;
            vVelFaceAbscissas.plus_right = vVelcxp;
            if (wVelIndex > -1)
              wVelFaceAbscissas.plus_right = wVelcxp;
          } else {
            nxm = wcxp - wc;
            nxp = w[cxpp] - wcxp;
            delN = minMod(nxm, nxp);
            faceWeight.plus_right = wcxp - 1.0/2.0 * delN;
          
            nxm = uVelcxp - uVelc;
            nxp = uVel[cxpp] - uVelcxp;
            delN = minMod(nxm, nxp);
            uVelFaceAbscissas.plus_right = uVelcxp - 1.0/2.0 * delN;
          
            nxm = vVelcxp - vVelc;
            nxp = vVel[cxpp] - vVelcxp;
            delN = minMod(nxm, nxp);
            vVelFaceAbscissas.plus_right = vVelcxp - 1.0/2.0 * delN;
          
            if (wVelIndex > -1) {
              nxm = wVelcxp - wVelc;
              nxp = wVel[cxpp] - wVelcxp;
              delN = minMod(nxm, nxp);
              wVelFaceAbscissas.plus_right = wVelcxp - 1.0/2.0 * delN;
            }
          }
          
        } else { //calculate reboudn vector
          std::vector<double> rebound (3);
          std::vector<double> incident (3);
          std::vector<double> temp (3);
          incident[0] = uVel[c]; incident[1] = vVel[c];
          //check for wvel index to allow testign in 2D x-y space easily
          incident[2] = (wVelIndex > -1 ) ? wVel[c] : 0.0;
          
          double dot = dotProd( wallNorm, incident );
          temp = scalarMult( wallNorm, dot );
          temp[0] -= incident[0]; temp[1] -= incident[1]; temp[2] -= incident[2];
          temp[0] *= epVec[0]; temp[1] *= epVec[1]; temp[2] *= epVec[2];
          rebound[0] = temp[0] * -1.0;
          rebound[1] = temp[1] * -1.0;
          rebound[2] = temp[2] * -1.0;
          
          faceWeight.plus_right = (1.0 - fStick) * w[c]/epW;
          uVelFaceAbscissas.plus_right = rebound[0];
          vVelFaceAbscissas.plus_right = rebound[1];
          if (wVelIndex > -1)
            wVelFaceAbscissas.plus_right = rebound[2];
        }
      }

      //simplest form when no wall cells touching
      cqFaceData1D bc_no_wall( const IntVector c, const IntVector coord, constCCVariable<double>& phi, const cqFaceBoundaryBool isBoundary )
      {
        CQMOM_Convection::cqFaceData1D face_values;
        
        IntVector cxpp = c + coord + coord;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        IntVector cxmm = c - coord - coord;
        
        double phic = phi[c];
        double phicxm = phi[cxm];
        double phicxp = phi[cxp];
        //calculate the inside faces
        double delN;
        double nxm = (phic - phicxm);
        double nxp = (phicxp - phic);
        delN = minMod(nxm, nxp);
        
        face_values.minus_right = phic - 1.0/2.0 * delN;
        face_values.plus_left = phic + 1.0/2.0 * delN;
        
        if ( isBoundary.minus ) {
          face_values.minus_left = phicxm;
        } else {
          nxm = (phicxm - phi[cxmm]);
          nxp = (phic - phicxm);
          delN = minMod(nxm, nxp);
          face_values.minus_left = phicxm + 1.0/2.0 * delN;
        }
        
        if ( isBoundary.plus ) {
          face_values.plus_right = phicxp;
        } else {
          nxm = (phicxp - phic);
          nxp = (phi[cxpp] - phicxp);
          delN = minMod(nxm, nxp);
          face_values.plus_right = phicxp - 1.0/2.0 * delN;
        }
        
        return face_values;
      }
      
      //functions to handle flat wall
      cqFaceData1D with_bc( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                            constCCVariable<double>& volFrac, const double epW, const cqFaceBoundaryBool isBoundary)
      {
        CQMOM_Convection::cqFaceData1D face_values;
        
        IntVector cxpp = c + coord + coord;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        IntVector cxmm = c - coord - coord;
        
        double phic = phi[c];
        double phicxm = phi[cxm];
        double phicxp = phi[cxp];
        //calculate the inside faces
        double delN;
        double nxm = (phic - phicxm);
        double nxp = (phicxp - phic);
        delN = minMod(nxm, nxp);
        
        face_values.minus_right = phic - 1.0/2.0 * delN;
        face_values.plus_left = phic + 1.0/2.0 * delN;
        
        if ( volFrac[cxm] == 1.0 || volFrac[c] == 0.0 ) {  //check if wall is present
          if (isBoundary.minus ) {
            face_values.minus_left = phicxm;
          } else {
            nxm = (phicxm - phi[cxmm]);
            nxp = (phic - phicxm);
            delN = minMod(nxm, nxp);
            face_values.minus_left = phicxm + 1.0/2.0 * delN;
          }
        } else {
          face_values.minus_left = face_values.minus_right;
        }
        
        if ( volFrac[cxp] == 1.0 || volFrac[c] == 0.0 ) {  //check if wall is present
          if (isBoundary.plus ) {
            face_values.plus_right = phicxp;
          } else {
            nxm = (phicxp - phic);
            nxp = (phi[cxpp] - phicxp);
            delN = minMod(nxm, nxp);
            face_values.plus_right = phicxp - 1.0/2.0 * delN;
          }
        } else {
          face_values.plus_right = face_values.plus_left;
        }
        
#ifdef cqmom_transport_dbg
        if (phi[c] > 0.0) {
          std::cout << "Cell c " << c << std::endl;
          std::cout << "inside -r: " << face_values.minus_right << std::endl;
          std::cout << "inside +l: " << face_values.plus_left << std::endl;
          std::cout << "plus +r: " << face_values.plus_right << std::endl;
          std::cout << "minus -l: " << face_values.minus_left << std::endl;
        }
#endif
        return face_values;
      }
      
      cqFaceData1D with_bc_weight( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                                   constCCVariable<double>& volFrac, const double epW, const cqFaceBoundaryBool isBoundary, const double fStick)
      {
        CQMOM_Convection::cqFaceData1D face_values;
        
        IntVector cxpp = c + coord + coord;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        IntVector cxmm = c - coord - coord;
        
        double phic = phi[c];
        double phicxm = phi[cxm];
        double phicxp = phi[cxp];
        //calculate the inside faces
        double delN;
        double nxm = (phic - phicxm);
        double nxp = (phicxp - phic);
        delN = minMod(nxm, nxp);
        
        face_values.minus_right = phic - 1.0/2.0 * delN;
        face_values.plus_left = phic + 1.0/2.0 * delN;
        
        if ( volFrac[cxm] == 1.0 || volFrac[c] == 0.0 ) { //check if wall is present
          if (isBoundary.minus ) {
            face_values.minus_left = phicxm;
          } else {
            nxm = (phicxm - phi[cxmm]);
            nxp = (phic - phicxm);
            delN = minMod(nxm, nxp);
            face_values.minus_left = phicxm + 1.0/2.0 * delN;
          }
        } else {
          face_values.minus_left = (1.0 - fStick) * face_values.minus_right/epW;
        }
        
        if ( volFrac[cxp] == 1.0 || volFrac[c] == 0.0 ) { //check if wall is present
          if (isBoundary.plus ) {
            face_values.plus_right = phicxp;
          } else {
            nxm = (phicxp - phic);
            nxp = (phi[cxpp] - phicxp);
            delN = minMod(nxm, nxp);
            face_values.plus_right = phicxp - 1.0/2.0 * delN;
          }
        } else {
          face_values.plus_right = (1.0 - fStick) * face_values.plus_left/epW;
        }
        
#ifdef cqmom_transport_dbg
        if (phi[c] > 0.0) {
          std::cout << "Cell c " << c << std::endl;
          std::cout << "inside -r: " << face_values.minus_right << std::endl;
          std::cout << "inside +l: " << face_values.plus_left << std::endl;
          std::cout << "plus +r: " << face_values.plus_right << std::endl;
          std::cout << "minus -l: " << face_values.minus_left << std::endl;
        }
#endif
        return face_values;
      }
      
      cqFaceData1D with_bc_normVel( const IntVector c, const IntVector coord, constCCVariable<double>& phi,
                                    constCCVariable<double>& volFrac, const double epW, const cqFaceBoundaryBool isBoundary)
      {
        CQMOM_Convection::cqFaceData1D face_values;
        
        IntVector cxpp = c + coord + coord;
        IntVector cxp = c + coord;
        IntVector cxm = c - coord;
        IntVector cxmm = c - coord - coord;
        
        double phic = phi[c];
        double phicxm = phi[cxm];
        double phicxp = phi[cxp];
        //calculate the inside faces
        double delN;
        double nxm = (phic - phicxm);
        double nxp = (phicxp - phic);
        delN = minMod(nxm, nxp);
        
        face_values.minus_right = phic - 1.0/2.0 * delN;
        face_values.plus_left = phic + 1.0/2.0 * delN;
        
        if ( volFrac[cxm] == 1.0 || volFrac[c] == 0.0 ) { //check if wall is present
          if (isBoundary.minus ) {
            face_values.minus_left = phicxm;
          } else {
            nxm = (phicxm - phi[cxmm]);
            nxp = (phic - phicxm);
            delN = minMod(nxm, nxp);
            face_values.minus_left = phicxm + 1.0/2.0 * delN;
          }
        } else {
          face_values.minus_left = -epW*face_values.minus_right;
        }
        
        if ( volFrac[cxp] == 1.0 || volFrac[c] == 0.0 ) { //check if wall is present
          if (isBoundary.plus ) {
            face_values.plus_right = phicxp;
          } else {
            nxm = (phicxp - phic);
            nxp = (phi[cxpp] - phicxp);
            delN = minMod(nxm, nxp);
            face_values.plus_right = phicxp - 1.0/2.0 * delN;
          }
        } else {
          face_values.plus_right = -epW*face_values.plus_left;
        }
        
#ifdef cqmom_transport_dbg
        if (phi[c] > 0.0) {
          std::cout << "Cell c " << c << std::endl;
          std::cout << "inside -r: " << face_values.minus_right << std::endl;
          std::cout << "inside +l: " << face_values.plus_left << std::endl;
          std::cout << "plus +r: " << face_values.plus_right << std::endl;
          std::cout << "minus -l: " << face_values.minus_left << std::endl;
        }
#endif
        return face_values;
      }
      
    private:
      double minMod( const double x, const double y)
      {
        double sgn = 0.0;
        if ( x < 0.0 ) {
          sgn = -1.0;
        } else if ( x > 0.0 ) {
          sgn = 1.0;
        }
        //        if (x == 0.0 )
        //          sgn = 0.0;
        
        double sgn2 = 0.0;
        if ( x*y < 0.0 ) {
          sgn2 = -1.0;
        } else if ( x*y > 0.0 ) {
          sgn2 = 1.0;
        }
        //        if (x*y == 0.0 )
        //          sgn2 = 0.0;
        
        double delN;
        if ( fabs(x) < fabs(y) ) {
          delN = sgn * ( 1.0 + sgn2 )/2.0 * fabs(x);
        } else {
          delN = sgn * ( 1.0 + sgn2 )/2.0 * fabs(y);
        }
        return delN;
      }
    }; //end class
    
  private:
    //---------------------------------------------------------------------------
    // Various varibles needed
    // --------------------------------------------------------------------------
    
    std::vector<MomentVector> momentIndexes; // Vector with all moment indexes
    
    ArchesLabel * d_fieldLabels;
    
    Interpolator * _opr;
    
    int M;
    int nNodes;
    int nMoments;
    int uVelIndex;
    int vVelIndex;
    int wVelIndex;
    std::vector<int> N_i;
    double epW;
    double convWeightLimit;
    bool partVel;
    bool d_deposition;
    
    std::vector<const VarLabel *> convLabels;
    std::vector<const VarLabel *> xConvLabels;
    std::vector<const VarLabel *> yConvLabels;
    std::vector<const VarLabel *> zConvLabels;
    std::vector<const VarLabel *> fStickLabels;
    const VarLabel * d_wallIntegerLabel;

    std::string d_convScheme; //first or second order convection
    
    //---------------------------------------------------------------------------
    // Helper/Utility type functions
    // --------------------------------------------------------------------------
    
    /** @brief Returns an iterator for all cells not touching the domain boundaries */
    inline CellIterator getInteriorCellIterator( const Patch* p ) const
    {
      bool xminus = p->getBCType(Patch::xminus) == Patch::None;
      bool xplus =  p->getBCType(Patch::xplus)  == Patch::None;
      bool yminus = p->getBCType(Patch::yminus) == Patch::None;
      bool yplus =  p->getBCType(Patch::yplus)  == Patch::None;
      bool zminus = p->getBCType(Patch::zminus) == Patch::None;
      bool zplus =  p->getBCType(Patch::zplus)  == Patch::None;
      
      IntVector clow  = p->getCellLowIndex();
      IntVector chigh = p->getCellHighIndex();
      IntVector clow_mod = clow;
      IntVector chigh_mod = chigh;
      
      if (xminus)
        clow_mod = clow_mod + IntVector(1,0,0);
      if (xplus)
        chigh_mod = chigh_mod - IntVector(1,0,0);
      if (yminus)
        clow_mod = clow_mod + IntVector(0,1,0);
      if (yplus)
        chigh_mod = chigh_mod - IntVector(0,1,0);
      if (zminus)
        clow_mod = clow_mod + IntVector(0,0,1);
      if (zplus)
        chigh_mod = chigh_mod - IntVector(0,0,1);
      
      CellIterator the_iterator = CellIterator(clow_mod, chigh_mod);
      
      the_iterator.reset();
      return the_iterator;
    }
    
    /** @brief Returns an iterator for all cell touching a domain boundary */
    inline CellIterator getInteriorBoundaryCellIterator( const Patch* p,
                                                        const std::vector<Patch::FaceType>::const_iterator bf_iter ) const
    {
      Patch::FaceType face = *bf_iter;
      IntVector l,h;
      p->getFaceCells( face, 0, l, h );
      
      if ( face == Patch::xminus || face == Patch::xplus ){
        
        bool yminus = p->getBCType(Patch::yminus) == Patch::None;
        bool yplus =  p->getBCType(Patch::yplus)  == Patch::None;
        bool zminus = p->getBCType(Patch::zminus) == Patch::None;
        bool zplus =  p->getBCType(Patch::zplus)  == Patch::None;
        
        if (yminus)
          l[1] += 1;
        if (yplus)
          h[1] -= 1;
        if (zminus)
          l[2] += 1;
        if (zplus)
          h[2] -= 1;
        
      } else if ( face == Patch::yminus || face == Patch::yplus ){
        
        bool xminus = p->getBCType(Patch::xminus) == Patch::None;
        bool xplus =  p->getBCType(Patch::xplus)  == Patch::None;
        bool zminus = p->getBCType(Patch::zminus) == Patch::None;
        bool zplus =  p->getBCType(Patch::zplus)  == Patch::None;
        
        if (xminus)
          l[0] += 2;
        if (xplus)
          h[0] -= 2;
        if (zminus)
          l[2] += 1;
        if (zplus)
          h[2] -= 1;
      } else if ( face == Patch::zminus || face == Patch::zplus ){
        
        bool yminus = p->getBCType(Patch::yminus) == Patch::None;
        bool yplus =  p->getBCType(Patch::yplus)  == Patch::None;
        bool xminus = p->getBCType(Patch::xminus) == Patch::None;
        bool xplus =  p->getBCType(Patch::xplus)  == Patch::None;
        
        if (yminus)
          l[1] += 2;
        if (yplus)
          h[1] -= 2;
        if (xminus)
          l[0] += 2;
        if (xplus)
          h[0] -= 2;
      }
      
      CellIterator the_iterator = CellIterator( l, h );
      return the_iterator;
    }
    
    /** @brief Checks if cell has a face on a boundary in a given absolute normal direction (coord) */
    inline cqFaceBoundaryBool checkFacesForBoundaries( const Patch* p, const IntVector c, const IntVector coord )
    {
      cqFaceBoundaryBool b;
      b.minus = false;
      b.plus  = false;
      
      IntVector l = p->getCellLowIndex();
      IntVector h = p->getCellHighIndex();
      
      bool fplus  = false;
      bool fminus = false;
      
      int I = -1;
      if ( coord[0] == 1 ) {
        fminus = p->getBCType(Patch::xminus) != Patch::Neighbor;
        fplus  = p->getBCType(Patch::xplus ) != Patch::Neighbor;
        I = 0;
      } else if ( coord[1] == 1 ){
        fminus = p->getBCType(Patch::yminus) != Patch::Neighbor;
        fplus  = p->getBCType(Patch::yplus ) != Patch::Neighbor;
        I = 1;
      } else if ( coord[2] == 1 ){
        fminus = p->getBCType(Patch::zminus) != Patch::Neighbor;
        fplus  = p->getBCType(Patch::zplus ) != Patch::Neighbor;
        I = 2;
      }
      
      if ( fminus && c[I] == l[I] ) {
        b.minus = true;
      }
      if ( fplus  && c[I] == h[I]-1 ) {
        b.plus = true;
      }
      return b;
    }

    //---------------------------------------------------------------------------
    // Assemblers
    // These functions assemble other terms
    // --------------------------------------------------------------------------
    /** @brief Computes the flux term of the cell based on summatino of face values */
    inline double getFlux( const double area, cqFaceData1D GPhi, const IntVector c, constCCVariable<double>& volFrac )
    {
      //Not using the areafraction here allows for flux to come from an intrusion cell into the domain as
      //the particles bounce, but requires checking celltype to prevent flux into intrusion cells
      if ( volFrac[c] == 1.0 ) {
        return area * ( GPhi.plus - GPhi.minus );
      }
      else {
        return 0.0;
      }
    }
    
    /** @brief Use the face interpolated weight and abscissa values to calculate the \f$ H^+ \f$ and \f$ H^- \f$ summation term at faces*/
    inline cqFaceData1D sumNodes( const std::vector<cqFaceData1D>& w, const std::vector<cqFaceData1D>& a, const int& nNodes, const int& M,
                                  const int& velIndex, const std::vector<int>& momentIndex, const double& weightLimit )
    {
      cqFaceData1D gPhi;
      double hPlus = 0.0, hMinus = 0.0;
      //plus cell face
      for (int i = 0; i < nNodes; i++) {
        if (a[i + nNodes*velIndex].plus_left > 0.0 && w[i].plus_left > weightLimit) {
          double nodeVal = 1.0;
          for (int m = 0 ; m < M ; m++) {
            for ( int n = 0; n < momentIndex[m]; n++ ) {
              nodeVal *= a[i + nNodes*m].plus_left;
            }
          }
          hPlus += w[i].plus_left * a[i + nNodes*velIndex].plus_left * nodeVal; //add hplus
        }
        if ( a[i + nNodes*velIndex].plus_right < 0.0 && w[i].plus_right > weightLimit) {
          double nodeVal = 1.0;
          for (int m = 0 ; m < M ; m++) {
            for (int n = 0; n < momentIndex[m]; n++ ) {
              nodeVal *= a[i + nNodes*m].plus_right;
            }
          }
          hMinus += w[i].plus_right * a[i + nNodes*velIndex].plus_right * nodeVal; //add hminus
        }
      }
      gPhi.plus = hPlus + hMinus;
      
      //minus cell face
      hPlus = 0.0; hMinus = 0.0;
      for (int i = 0; i < nNodes; i++) {
        if (a[i + nNodes*velIndex].minus_left > 0.0 && w[i].minus_left > weightLimit) {
          double nodeVal = 1.0;
          for (int m = 0 ; m < M ; m++) {
            for (int n = 0; n < momentIndex[m]; n++ ) {
              nodeVal *= a[i + nNodes*m].minus_left;
            }
          }
          hPlus += w[i].minus_left * a[i + nNodes*velIndex].minus_left * nodeVal; //add hplus
        }
        if ( a[i + nNodes*velIndex].minus_right < 0.0 && w[i].minus_right > weightLimit) {
          double nodeVal = 1.0;
          for (int m = 0 ; m < M ; m++) {
            for (int n = 0; n < momentIndex[m]; n++ ) {
              nodeVal *= a[i + nNodes*m].minus_right;
            }
          }
          hMinus += w[i].minus_right * a[i + nNodes*velIndex].minus_right * nodeVal; //add hminus
        }
      }
      gPhi.minus = hPlus + hMinus;
      return gPhi;
    }

   }; //end class
  
} //namespace Uintah
#endif
