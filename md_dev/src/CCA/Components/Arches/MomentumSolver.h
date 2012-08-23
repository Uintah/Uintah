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


//----- MomentumSolver.h -----------------------------------------------

#ifndef Uintah_Components_Arches_MomentumSolver_h
#define Uintah_Components_Arches_MomentumSolver_h

/**************************************
CLASS
   MomentumSolver
   
   Class MomentumSolver linearizes and solves momentum
   equation on a grid hierarchy


GENERAL INFORMATION
   MomentumSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class MomentumSolver linearizes and solves momentum
   equation on a grid hierarchy


WARNING
   none

************************************************************************/
#include <CCA/Ports/SchedulerP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Components/Arches/Discretization.h>
namespace Uintah {
class ArchesLabel;
class MPMArchesLabel;
class ProcessorGroup;
class TurbulenceModel;
class PhysicalConstants;
class Source;
#ifdef PetscFilter
class Filter;
#endif
class BoundaryCondition;
class RHSSolver;
class TimeIntegratorLabel;

class MomentumSolver {

public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Construct an instance of the Momentum solver.
  // PRECONDITIONS
  // POSTCONDITIONS
  //   A linear level solver is partially constructed.  
  MomentumSolver(const ArchesLabel* label, 
                 const MPMArchesLabel* MAlb,
                 TurbulenceModel* turb_model, 
                 BoundaryCondition* bndry_cond,
                 PhysicalConstants* physConst);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
  ~MomentumSolver();

  // GROUP: Problem Setup :
  ///////////////////////////////////////////////////////////////////////
  // Set up the problem specification database
  void problemSetup(const ProblemSpecP& params);

  // GROUP: Schedule Action :
  ///////////////////////////////////////////////////////////////////////
  // Schedule Solve of the linearized momentum equation. 
  void solve(SchedulerP& sched,
             const PatchSet* patches,
             const MaterialSet* matls,
             const TimeIntegratorLabel* timelabels,
             bool extraProjection);

  ///////////////////////////////////////////////////////////////////////
  // Schedule the build of the linearized momentum matrix
  void sched_buildLinearMatrix(SchedulerP& sched, 
                               const PatchSet* patches,
                               const MaterialSet* matls,
                               const TimeIntegratorLabel* timelabels,
                               bool extraProjection );


  void solveVelHat(const LevelP& level,
                   SchedulerP&,
                   const TimeIntegratorLabel* timelabels );

  ///////////////////////////////////////////////////////////////////////
  // Schedule the build of the linearized eqn
  void sched_buildLinearMatrixVelHat(SchedulerP&, 
                                     const PatchSet* patches,
                                     const MaterialSet* matls,
                                     const TimeIntegratorLabel* timelabels );

  void sched_averageRKHatVelocities(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls,
                                    const TimeIntegratorLabel* timelabels );

  void sched_prepareExtraProjection(SchedulerP& sched, 
                                    const PatchSet* patches,
                                    const MaterialSet* matls,
                                    const TimeIntegratorLabel* timelabels,
                                    bool set_BC);

  void setInitVelCondition( const Patch* patch, 
                            SFCXVariable<double>& uvel, 
                            SFCYVariable<double>& vvel, 
                            SFCZVariable<double>& wvel );

#ifdef PetscFilter
  inline void setDiscretizationFilter(Filter* filter) {
    d_discretize->setFilter(filter);
  }
#endif
  inline void setMMS(bool doMMS) {
    d_doMMS=doMMS;
  }
  inline void setMomentumCoupling(bool doMC) {
    d_momentum_coupling = doMC;
  }

private:

  // GROUP: Constructors (private):
  ////////////////////////////////////////////////////////////////////////
  // Default constructor.
  MomentumSolver();

  // GROUP: Action Methods (private):
  ///////////////////////////////////////////////////////////////////////
  // Actually build the linearized momentum matrix
  void buildLinearMatrix(const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* /*matls*/,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw,
                         const TimeIntegratorLabel* timelabels,
                         bool extraProjection );


  void buildLinearMatrixVelHat(const ProcessorGroup* pc,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse*,
                               DataWarehouse*,
                               const TimeIntegratorLabel* timelabels );

  void averageRKHatVelocities(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset* matls,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw,
                              const TimeIntegratorLabel* timelabels );

  void prepareExtraProjection(const ProcessorGroup* pc,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse*,
                               DataWarehouse*,
                               const TimeIntegratorLabel* timelabels,
                               bool set_BC);


  // const VarLabel* (required)
  const ArchesLabel* d_lab;
  const MPMArchesLabel* d_MAlab;
  bool d_momentum_coupling;
  bool d_central;

  // computes coefficients
  Discretization* d_discretize;
  // computes sources
  Source* d_source;
  // linear solver
  RHSSolver* d_rhsSolver;
  // turbulence model
  TurbulenceModel* d_turbModel;
  // boundary condition
  BoundaryCondition* d_boundaryCondition;
  // physical constants
  PhysicalConstants* d_physicalConsts;
  bool d_3d_periodic;
  bool d_filter_divergence_constraint;
  bool d_mixedModel;
  bool d_doMMS;
  vector<string> d_new_sources; 


  //--------------------- for initialization -----------
  class VelocityInitBase { 
    
    public:

      VelocityInitBase(){}; 
      virtual ~VelocityInitBase(){}; 

      virtual void problemSetup( ProblemSpecP db ) = 0; 
      virtual void setXVel( const Patch* patch, SFCXVariable<double>& vel ) = 0;
      virtual void setYVel( const Patch* patch, SFCYVariable<double>& vel ) = 0;
      virtual void setZVel( const Patch* patch, SFCZVariable<double>& vel ) = 0;

    protected: 

      std::string _init_type; 

  }; 

  VelocityInitBase* _init_function; 
  std::string _init_type; 

  // constant initialization ------------------------
  class ConstantVel : public VelocityInitBase { 

    public: 

      ConstantVel(){ 
        _const_u = 0.0;
        _const_v = 0.0;
        _const_w = 0.0;
      };
      ~ConstantVel(){}; 

      void problemSetup( ProblemSpecP db ){ 

        db->getWithDefault( "const_u", _const_u, 0.0 ); 
        db->getWithDefault( "const_v", _const_v, 0.0 ); 
        db->getWithDefault( "const_w", _const_w, 0.0 ); 

      }; 

      void setXVel( const Patch* patch, SFCXVariable<double>& uvel ){ 

        uvel.initialize( _const_u ); 

      }; 
      
      void setYVel( const Patch* patch, SFCYVariable<double>& vvel ){ 

        vvel.initialize( _const_v ); 

      }; 

      void setZVel( const Patch* patch, SFCZVariable<double>& wvel ){ 

        wvel.initialize( _const_w ); 

      }; 

    private: 

      double _const_u;
      double _const_v; 
      double _const_w;  

  };  

  // almgren mms initialization ------------------------
  class AlmgrenVel : public VelocityInitBase { 

    public: 

      AlmgrenVel(){ 
        _pi = acos(-1.0); 
      };
      ~AlmgrenVel(){}; 

      void problemSetup( ProblemSpecP db ){ 

        db->getWithDefault( "A", _A, 1.0 ); 
        db->getWithDefault( "B", _B, 1.0 ); 
        db->getWithDefault( "plane", _plane, "x-y"); 
        //valid options are x-y, y-z, z-x

      }; 

      void setXVel( const Patch* patch, SFCXVariable<double>& uvel ){ 

        for (CellIterator iter=patch->getSFCXIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          Point p = patch->cellPosition(c);

          if ( _plane == "x-y" ){ 

            uvel[c] = 1.0 - _A * cos( 2.0*_pi*p.x() ) 
                               * sin( 2.0*_pi*p.y() ); 

          } else if ( _plane == "z-x" ){ 

            uvel[c] = 1.0 + _B * sin( 2.0*_pi*p.x() ) 
                               * cos( 2.0*_pi*p.z() ); 

          } else { 
            uvel[c] = 0.0; 
          }
 
        }

      }; 
      
      void setYVel( const Patch* patch, SFCYVariable<double>& vvel ){ 

        for (CellIterator iter=patch->getSFCXIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          Point p = patch->cellPosition(c);

          if ( _plane == "y-z" ){ 

            vvel[c] = 1.0 - _A * cos( 2.0*_pi*p.y() ) 
                               * sin( 2.0*_pi*p.z() ); 

          } else if ( _plane == "x-y" ){ 

            vvel[c] = 1.0 + _B * sin( 2.0*_pi*p.x() ) 
                               * cos( 2.0*_pi*p.y() ); 

          } else { 
            vvel[c] = 0.0; 
          }
 
        }

      }; 

      void setZVel( const Patch* patch, SFCZVariable<double>& wvel ){ 

        for (CellIterator iter=patch->getSFCXIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          Point p = patch->cellPosition(c);

          if ( _plane == "z-x" ){ 

            wvel[c] = 1.0 - _A * cos( 2.0*_pi*p.z() ) 
                               * sin( 2.0*_pi*p.x() ); 

          } else if ( _plane == "y-z" ){ 

            wvel[c] = 1.0 + _B * sin( 2.0*_pi*p.y() ) 
                               * cos( 2.0*_pi*p.z() ); 

          } else { 
            wvel[c] = 0.0; 
          }
 
        }

      }; 

    private: 

      double _A;
      double _B; 
      double _pi; 
      std::string _plane;  

  };  

  // taylor-green initialization ------------------------
  class TaylorGreen3D : public VelocityInitBase { 

    public: 

      TaylorGreen3D(){ 
        _pi = acos(-1.0); 
      };
      ~TaylorGreen3D(){}; 

      void problemSetup( ProblemSpecP db ){ 

        db->getWithDefault( "c", _c, 2.0 ); 

      }; 

      void setXVel( const Patch* patch, SFCXVariable<double>& uvel ){ 

        double x,y,z;
        Vector Dx = patch->dCell(); 

        for (CellIterator iter=patch->getSFCXIterator(); !iter.done(); iter++){

          IntVector c = *iter;

          x = c.x() * Dx.x(); 
          y = c.y() * Dx.y() + Dx.y()/2.0;
          z = c.z() * Dx.z() + Dx.z()/2.0;

          uvel[c] = 2.0/sqrt(3.0) * sin( _c + 2.0*_pi/3.0 ) 
            * sin( 2 * _pi * x ) 
            * cos( 2 * _pi * y )
            * cos( 2 * _pi * z ); 
        }    
      };

      
      void setYVel( const Patch* patch, SFCYVariable<double>& vvel ){ 

        double x,y,z;
        Vector Dx = patch->dCell(); 

        for (CellIterator iter=patch->getSFCYIterator(); !iter.done(); iter++){

          IntVector c = *iter;

          x = c.x() * Dx.x() + Dx.x()/2.0; 
          y = c.y() * Dx.y();
          z = c.z() * Dx.z() + Dx.z()/2.0;

          vvel[c] = 2.0/sqrt(3.0) * sin( _c - 2.0*_pi/3.0 ) 
            * sin( 2 * _pi * y ) 
            * cos( 2 * _pi * x )
            * cos( 2 * _pi * z ); 
        }    
      }; 

      void setZVel( const Patch* patch, SFCZVariable<double>& wvel ){ 

        double x,y,z;
        Vector Dx = patch->dCell(); 

        for (CellIterator iter=patch->getSFCZIterator(); !iter.done(); iter++){

          IntVector c = *iter;

          x = c.x() * Dx.x() + Dx.x()/2.0; 
          y = c.y() * Dx.y() + Dx.y()/2.0;
          z = c.z() * Dx.z();

          wvel[c] = 2.0/sqrt(3.0) * sin( _c ) 
            * sin( 2 * _pi * z ) 
            * cos( 2 * _pi * x )
            * cos( 2 * _pi * y ); 
        }    
      }; 
 
    private: 

      double _c; 
      double _pi;

  };  
  
  
  // ExponentialVortex initialization ------------------------
  // Tony Saad (just in case you need to blame someone)
  class ExponentialVortex : public VelocityInitBase { 
    
  public: 
    
    ExponentialVortex(){};
    ~ExponentialVortex(){}; 
    
    void problemSetup( ProblemSpecP db ){ 
      
      db->getWithDefault( "x0", x0_, 0.0 ); 
      db->getWithDefault( "y0", y0_, 0.0 ); 
      db->getWithDefault( "z0", z0_, 0.0 ); 
      db->getWithDefault( "G",  G_,  0.01 ); 
      db->getWithDefault( "R",  R_,  0.1 );       
      db->getWithDefault( "U",  U_,  1.0 );             
      db->getWithDefault( "V",  V_,  0.0 );
      db->getWithDefault( "plane", plane_, "x-y"); 
      //valid options are x-y, y-z, z-x      
      GR_ = G_/(R_*R_);
    }; 
    
    void setXVel( const Patch* patch, SFCXVariable<double>& uvel ){ 
      
      double x,y,z,r;
      Vector Dx = patch->dCell(); 
      const double dx_2 = Dx.x()/2.0;
      
      for (CellIterator iter=patch->getSFCXIterator(); !iter.done(); iter++){          
        
        IntVector c = *iter;
        Uintah::Point position = patch->getCellPosition(c);        
        x = position.x() - dx_2; 
        y = position.y();
        z = position.z();
        
        if (plane_ == "x-y") {
          r = (x - x0_)*(x - x0_) + (y - y0_)*(y - y0_);
          uvel[c] = U_ - GR_*(y-y0_)*exp(-r/(2.0*R_*R_));          
        } 
        
        else if (plane_ == "z-x") {
          r = (x - x0_)*(x - x0_) + (z - z0_)*(z - z0_);
          uvel[c] = V_ + GR_*(z-z0_)*exp(-r/(2.0*R_*R_));
        } 
        
        else {
          uvel[c] = 0.0;                             
        }
      }
      
    };
    
    
    void setYVel( const Patch* patch, SFCYVariable<double>& vvel ){ 
      
      double x,y,z,r;
      Vector Dx = patch->dCell(); 
      const double dy_2 = Dx.y()/2.0;
      
      for (CellIterator iter=patch->getSFCYIterator(); !iter.done(); iter++){      
        
        IntVector c = *iter;
        Uintah::Point position = patch->getCellPosition(c);        
        x = position.x(); 
        y = position.y() - dy_2;
        z = position.z();

        if (plane_ == "x-y") {
          r = (x - x0_)*(x - x0_) + (y - y0_)*(y - y0_);
          vvel[c] =  V_ + GR_*(x-x0_)*exp(-r/(2.0*R_*R_));
        } 
        
        else if (plane_ == "y-z") {
          r = (y - y0_)*(y - y0_) + (z - z0_)*(z - z0_);
          vvel[c] =  U_ - GR_*(z-z0_)*exp(-r/(2.0*R_*R_));
        } 
        
        else {
          vvel[c] = 0.0;                             
        }        
        
      }
    }; 
    
    void setZVel( const Patch* patch, SFCZVariable<double>& wvel ){     
      
      double x,y,z, r;
      Vector Dx = patch->dCell(); 
      const double dz_2 = Dx.z()/2.0;
      
      for (CellIterator iter=patch->getSFCZIterator(); !iter.done(); iter++){      
        
        IntVector c = *iter;
        Uintah::Point position = patch->getCellPosition(c);        
        x = position.x();
        y = position.y(); 
        z = position.z() - dz_2;
        
        if (plane_ == "z-x") {
          r = (x - x0_)*(x - x0_) + (z - z0_)*(z - z0_);
          wvel[c] =  U_ - GR_*(x-x0_)*exp(-r/(2.0*R_*R_));
        } 
        
        else if (plane_ == "y-z") {
          r = (y - y0_)*(y - y0_) + (z - z0_)*(z - z0_);
          wvel[c] =  V_ + GR_*(y-y0_)*exp(-r/(2.0*R_*R_));
        } 
        
        else {
          wvel[c] = 0.0;                             
        }        
        
      }
    }; 
    
  private: 
    double x0_, y0_, z0_, G_, R_, U_, V_, GR_;
    std::string plane_;
  };  
  
  

}; // End class MomentumSolver
} // End namespace Uintah


#endif

