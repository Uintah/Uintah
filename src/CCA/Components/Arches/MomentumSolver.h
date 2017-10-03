/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
#include <CCA/Components/Arches/Task/TaskFactoryBase.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <Core/IO/UintahZlibUtil.h>

namespace Uintah {

class ArchesLabel;
class MPMArchesLabel;
class ProcessorGroup;
class TurbulenceModel;
class PhysicalConstants;
class Source;
class Filter;
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
                 PhysicalConstants* physConst,
                 std::map<std::string, std::shared_ptr<TaskFactoryBase> >* task_factory_map
                 );

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
  ~MomentumSolver();

  // GROUP: Problem Setup :
  ///////////////////////////////////////////////////////////////////////
  // Set up the problem specification database
  void problemSetup(const ProblemSpecP& params,
		    SimulationStateP & sharedState);

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
                   const TimeIntegratorLabel* timelabels,
                   const int curr_level );

  ///////////////////////////////////////////////////////////////////////
  // Schedule the build of the linearized eqn
  void sched_buildLinearMatrixVelHat(SchedulerP&,
                                     const PatchSet* patches,
                                     const MaterialSet* matls,
                                     const TimeIntegratorLabel* timelabels,
                                     const int curr_level );

  void sched_averageRKHatVelocities(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls,
                                    const TimeIntegratorLabel* timelabels );

  void sched_prepareExtraProjection(SchedulerP& sched,
                                    const PatchSet* patches,
                                    const MaterialSet* matls,
                                    const TimeIntegratorLabel* timelabels,
                                    bool set_BC);

  void sched_computeMomentum( const LevelP& level,
                                SchedulerP& sched,
                                const int timesubstep,
                                const bool isInitialization=false );

  void setInitVelCondition( const Patch* patch,
                            SFCXVariable<double>& uvel,
                            SFCYVariable<double>& vvel,
                            SFCZVariable<double>& wvel,
                            constCCVariable<double>& rho );

  inline void setDiscretizationFilter(Filter* filter) {
    d_discretize->setFilter(filter);
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
                               const TimeIntegratorLabel* timelabels,
                               const int curr_level );

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

  void computeMomentum( const ProcessorGroup* pc,
                          const PatchSubset* patches,
                          const MaterialSubset*,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          const int timesubstep,
                          const bool isInitialization=false);

  // const VarLabel* (required)
  const ArchesLabel* d_lab;
  const MPMArchesLabel* d_MAlab;
  bool d_momentum_coupling;
  bool d_central;

  const VarLabel* d_denRefArrayLabel;

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
  std::vector<std::string> d_new_sources;

  const VarLabel* _u_mom;
  const VarLabel* _v_mom;
  const VarLabel* _w_mom;

  std::map<std::string, std::shared_ptr<TaskFactoryBase> >* _task_factory_map;

  std::string d_wall_closure;
  double d_wall_const_smag_C;
  int d_standoff_index; 

  //--------------------- for initialization -----------
  class VelocityInitBase {

    public:

      VelocityInitBase(){};
      virtual ~VelocityInitBase(){};

      virtual void problemSetup( ProblemSpecP db ) = 0;
      virtual void setXVel( const Patch* patch, SFCXVariable<double>& vel, constCCVariable<double>& rho ) = 0;
      virtual void setYVel( const Patch* patch, SFCYVariable<double>& vel, constCCVariable<double>& rho ) = 0;
      virtual void setZVel( const Patch* patch, SFCZVariable<double>& vel, constCCVariable<double>& rho ) = 0;

    protected:

      std::string _init_type;

  };

  VelocityInitBase* _init_function;
  std::string _init_type;
  Discretization::MOMCONV d_conv_scheme;
  double d_re_limit;
  double d_re_limit_wall_upwind;

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

      void setXVel( const Patch* patch, SFCXVariable<double>& uvel, constCCVariable<double>& rho ){

        uvel.initialize( _const_u );

      };

      void setYVel( const Patch* patch, SFCYVariable<double>& vvel, constCCVariable<double>& rho ){

        vvel.initialize( _const_v );

      };

      void setZVel( const Patch* patch, SFCZVariable<double>& wvel, constCCVariable<double>& rho ){

        wvel.initialize( _const_w );

      };

    private:

      double _const_u;
      double _const_v;
      double _const_w;

  };

  // read in a CC velocity field and interpolate it to the staggered positions.
  class InputfileInit : public VelocityInitBase {

    public:

      InputfileInit(){
      };
      ~InputfileInit(){};

      void problemSetup( ProblemSpecP db ){

        db->require("input_file",_input_file);


      };

      void setXVel( const Patch* patch, SFCXVariable<double>& uvel, constCCVariable<double>& rho ){

        gzFile file = gzopen( _input_file.c_str(), "r" );

        if ( file == nullptr ) {
          proc0cout << "Error opening file: " << _input_file << " for velocity initialization." << std::endl;
          throw ProblemSetupException("Unable to open the given input file: " + _input_file, __FILE__, __LINE__);
        }

        int nx,ny,nz;
        nx = getInt(file);
        ny = getInt(file);
        nz = getInt(file);

        const Level* level = patch->getLevel();
        IntVector low, high;
        level->findCellIndexRange(low, high);
        IntVector range = high-low;

        if (!(range == IntVector(nx,ny,nz))) {
          std::ostringstream warn;
          warn << "ERROR: \n Wrong grid size in input file for velocities." << range;
          throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
        }

        int size = 0;
        double uvel_in,tmp;
        IntVector idxLo = patch->getFortranCellLowIndex();
        IntVector idxHi = patch->getFortranCellHighIndex();
        for (int colZ = 1; colZ <= nz; colZ ++) {
          for (int colY = 1; colY <= ny; colY ++) {
            for (int colX = 1; colX <= nx; colX ++) {

              IntVector currCell(colX-1, colY-1, colZ-1);
              uvel_in = getDouble(file);
              tmp =  getDouble(file);
              (void)tmp; //to silence the unused variable warning.
              tmp =  getDouble(file);
              (void)tmp; //to silence the unused variable warning.
              if ((currCell.x() <= idxHi.x() && currCell.y() <= idxHi.y() && currCell.z() <= idxHi.z()) &&
                  (currCell.x() >= idxLo.x() && currCell.y() >= idxLo.y() && currCell.z() >= idxLo.z())) {
                uvel[currCell] = uvel_in;
                size++;
              }
            }
          }
        }
        gzclose( file );

      };

      void setYVel( const Patch* patch, SFCYVariable<double>& vvel, constCCVariable<double>& rho ){

        gzFile file = gzopen( _input_file.c_str(), "r" );

        if ( file == nullptr ) {
          proc0cout << "Error opening file: " << _input_file << " for velocity initialization." << std::endl;
          throw ProblemSetupException("Unable to open the given input file: " + _input_file, __FILE__, __LINE__);
        }

        int nx,ny,nz;
        nx = getInt(file);
        ny = getInt(file);
        nz = getInt(file);

        const Level* level = patch->getLevel();
        IntVector low, high;
        level->findCellIndexRange(low, high);
        IntVector range = high-low;

        if (!(range == IntVector(nx,ny,nz))) {
          std::ostringstream warn;
          warn << "ERROR: \n Wrong grid size in input file for velocities." << range;
          throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
        }

        int size = 0;
        double vvel_in,tmp;
        IntVector idxLo = patch->getFortranCellLowIndex();
        IntVector idxHi = patch->getFortranCellHighIndex();
        for (int colZ = 1; colZ <= nz; colZ ++) {
          for (int colY = 1; colY <= ny; colY ++) {
            for (int colX = 1; colX <= nx; colX ++) {

              IntVector currCell(colX-1, colY-1, colZ-1);
              tmp = getDouble(file);
              (void)tmp; //to silence the unused variable warning.
              vvel_in =  getDouble(file);
              tmp =  getDouble(file);
              (void)tmp; //to silence the unused variable warning.
              if ((currCell.x() <= idxHi.x() && currCell.y() <= idxHi.y() && currCell.z() <= idxHi.z()) &&
                  (currCell.x() >= idxLo.x() && currCell.y() >= idxLo.y() && currCell.z() >= idxLo.z())) {
                vvel[currCell] = vvel_in;
                size++;
              }
            }
          }
        }
        gzclose( file );


      };

      void setZVel( const Patch* patch, SFCZVariable<double>& wvel, constCCVariable<double>& rho ){

        gzFile file = gzopen( _input_file.c_str(), "r" );

        if ( file == nullptr ) {
          proc0cout << "Error opening file: " << _input_file << " for velocity initialization." << std::endl;
          throw ProblemSetupException("Unable to open the given input file: " + _input_file, __FILE__, __LINE__);
        }

        int nx,ny,nz;
        nx = getInt(file);
        ny = getInt(file);
        nz = getInt(file);

        const Level* level = patch->getLevel();
        IntVector low, high;
        level->findCellIndexRange(low, high);
        IntVector range = high-low;

        if (!(range == IntVector(nx,ny,nz))) {
          std::ostringstream warn;
          warn << "ERROR: \n Wrong grid size in input file for velocities." << range;
          throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
        }

        int size = 0;
        double wvel_in,tmp;
        IntVector idxLo = patch->getFortranCellLowIndex();
        IntVector idxHi = patch->getFortranCellHighIndex();
        for (int colZ = 1; colZ <= nz; colZ ++) {
          for (int colY = 1; colY <= ny; colY ++) {
            for (int colX = 1; colX <= nx; colX ++) {

              IntVector currCell(colX-1, colY-1, colZ-1);
              tmp = getDouble(file);
              (void)tmp; //to silence the unused variable warning.
              tmp =  getDouble(file);
              (void)tmp; //to silence the unused variable warning.
              wvel_in =  getDouble(file);
              if ((currCell.x() <= idxHi.x() && currCell.y() <= idxHi.y() && currCell.z() <= idxHi.z()) &&
                  (currCell.x() >= idxLo.x() && currCell.y() >= idxLo.y() && currCell.z() >= idxLo.z())) {
                wvel[currCell] = wvel_in;
                size++;
              }
            }
          }
        }
        gzclose( file );


      };

    private:

      std::string _input_file;

  };

  // Atmospheric Boundary Layer initialization ------------------------
  class StABLVel : public VelocityInitBase {

    public:

      StABLVel(){
      };
      ~StABLVel(){};

      void problemSetup( ProblemSpecP db ){

        std::string which_bc;
        db->require( "which_bc", which_bc );

        Vector grav;
        db->getRootNode()->findBlock("PhysicalConstants")->require("gravity",grav);
        if ( grav.x() != 0 ){
          _dir_grav = 0;
        }
        else if ( grav.y() != 0 ){
          _dir_grav = 1;
        }
        else if ( grav.z() != 0 ){
          _dir_grav = 2;
        }
        else {
          throw ProblemSetupException("Error: The specified gravity doesnt indicate a clear up-down direction", __FILE__, __LINE__);
        }

        ProblemSpecP db_bc   = db->getRootNode()->findBlock("Grid")->findBlock("BoundaryConditions");
        bool found_boundary = false;
        _do_u = false;
        _do_v = false;
        _do_w = false;
        _sign = 1.0;

        if ( db_bc ) {

          for ( ProblemSpecP db_face = db_bc->findBlock("Face"); db_face != nullptr; db_face = db_face->findNextBlock("Face") ){

            std::string which_face;
            int v_index=999;
            if ( db_face->getAttribute("side", which_face)){
              db_face->getAttribute("side",which_face);
            }
            else if ( db_face->getAttribute( "circle", which_face)){
              db_face->getAttribute("circle",which_face);
            }
            else if ( db_face->getAttribute( "rectangle", which_face)){
              db_face->getAttribute("rectangle",which_face);
            }
            else if ( db_face->getAttribute( "annulus", which_face)){
              db_face->getAttribute("annulus",which_face);
            }
            else if ( db_face->getAttribute( "ellipse", which_face)){
              db_face->getAttribute("ellipse",which_face);
            }

            //avoid the "or" in case I want to add more logic
            //re: the face normal.
            if ( which_face =="x-"){
              v_index = 0;
            }
            else if ( which_face =="x+"){
              v_index = 0;
            }
            else if ( which_face =="y-"){
              v_index = 1;
            }
            else if ( which_face =="y+"){
              v_index = 1;
            }
            else if ( which_face =="z-"){
              v_index = 2;
            }
            else if ( which_face =="z+"){
              v_index = 2;
            }
            else {
              throw InvalidValue("Error: Could not identify the boundary face direction.", __FILE__, __LINE__);
            }

            for( ProblemSpecP db_BCType = db_face->findBlock("BCType"); db_BCType != nullptr; db_BCType = db_BCType->findNextBlock("BCType") ) {

              std::string name = "NA";
              std::string type;
              db_BCType->getAttribute("label", name);
              db_BCType->getAttribute("var", type);

              if ( name == which_bc && found_boundary == false ){

                if ( found_boundary == false ){
                  found_boundary = true;
                  db_BCType->require("roughness",_zo);
                  db_BCType->require("freestream_h",_zh);
                  db_BCType->require("value",_u_inf);  // Using <value> as the infinite velocity
                  db_BCType->getWithDefault("k",_k,0.41);

                  _kappa = pow( _k / log( _zh / _zo ), 2.0);
                  _ustar = pow( (_kappa * pow(_u_inf[v_index],2.0)), 0.5 );

                  if ( which_face == "x-" ){
                    _do_u = true;
                  } else if ( which_face =="x+" ){
                    _sign = -1.0;
                    _do_u = true;
                  } else if ( which_face == "y-" ){
                    _do_v = true;
                  } else if ( which_face == "y+" ){
                    _sign = -1.0;
                    _do_v = true;
                  } else if ( which_face == "z-" ){
                    _do_w = true;
                  } else if ( which_face == "z+" ){
                    _sign = -1.0;
                    _do_w = true;
                  }

                } else {
                  throw ProblemSetupException("Error: You have two BCs with the same name using StABL: "+name, __FILE__, __LINE__);
                }
              }

            }
          }
        }

      };

      void setXVel( const Patch* patch, SFCXVariable<double>& uvel, constCCVariable<double>& rho ){

        if ( _do_u ){
          for ( CellIterator iter=patch->getCellIterator(); !iter.done(); iter++ ){
            IntVector c = *iter;
            Point p = patch->getCellPosition(c);
            uvel[c] = _sign * _ustar / _k * log( p(_dir_grav) / _zo );
          }
        }

      };

      void setYVel( const Patch* patch, SFCYVariable<double>& vvel, constCCVariable<double>& rho ){

        if ( _do_v ){
          for ( CellIterator iter=patch->getCellIterator(); !iter.done(); iter++ ){
            IntVector c = *iter;
            Point p = patch->getCellPosition(c);
            vvel[c] = _sign * _ustar / _k * log( p(_dir_grav) / _zo );
          }
        }

      };

      void setZVel( const Patch* patch, SFCZVariable<double>& wvel, constCCVariable<double>& rho ){

        if ( _do_w ){
          for ( CellIterator iter=patch->getCellIterator(); !iter.done(); iter++ ){
            IntVector c = *iter;
            Point p = patch->getCellPosition(c);
            wvel[c] = _sign * _ustar / _k * log( p(_dir_grav) / _zo );
          }
        }

      };

    private:

      unsigned int _dir_grav;
      double _zo;
      double _zh;
      double _k;
      double _kappa;
      Vector _u_inf;
      double _ustar;
      double _sign;
      bool _do_u;
      bool _do_v;
      bool _do_w;

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

      void setXVel( const Patch* patch, SFCXVariable<double>& uvel, constCCVariable<double>& rho ){

        Vector DX = patch->dCell();

        double dx2 = DX.x()/2.;

        for (CellIterator iter=patch->getSFCXIterator(); !iter.done(); iter++){

          IntVector c = *iter;
          Point p = patch->cellPosition(c);

          if ( _plane == "x-y" ){

            uvel[c] = 1.0 - _A * cos( 2.0*_pi*(p.x() - dx2) )
                               * sin( 2.0*_pi*p.y() );

          } else if ( _plane == "z-x" ){

            uvel[c] = 1.0 + _B * sin( 2.0*_pi*p.z() )
                               * cos( 2.0*_pi*(p.x() - dx2) );

          } else {
            uvel[c] = 0.0;
          }

        }

      };

      void setYVel( const Patch* patch, SFCYVariable<double>& vvel, constCCVariable<double>& rho ){

        Vector DX = patch->dCell();

        double dy2 = DX.y()/2.;

        for (CellIterator iter=patch->getSFCXIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          Point p = patch->cellPosition(c);

          if ( _plane == "y-z" ){

            vvel[c] = 1.0 - _A * cos( 2.0*_pi*(p.y() - dy2) )
                               * sin( 2.0*_pi*p.z() );

          } else if ( _plane == "x-y" ){

            vvel[c] = 1.0 + _B * sin( 2.0*_pi*p.x() )
                               * cos( 2.0*_pi*(p.y() - dy2) );

          } else {
            vvel[c] = 0.0;
          }

        }

      };

      void setZVel( const Patch* patch, SFCZVariable<double>& wvel, constCCVariable<double>& rho ){

        Vector DX = patch->dCell();

        double dz2 = DX.z()/2.;


        for (CellIterator iter=patch->getSFCXIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          Point p = patch->cellPosition(c);

          if ( _plane == "z-x" ){

            wvel[c] = 1.0 - _A * cos( 2.0*_pi*(p.z() - dz2) )
                               * sin( 2.0*_pi*p.x() );

          } else if ( _plane == "y-z" ){

            wvel[c] = 1.0 + _B * sin( 2.0*_pi*p.y() )
                               * cos( 2.0*_pi*(p.z() - dz2) );

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

      void setXVel( const Patch* patch, SFCXVariable<double>& uvel, constCCVariable<double>& rho ){

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


      void setYVel( const Patch* patch, SFCYVariable<double>& vvel, constCCVariable<double>& rho ){

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

      void setZVel( const Patch* patch, SFCZVariable<double>& wvel, constCCVariable<double>& rho ){

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

  // L. Shunn and P. Moin variable den MMS  ------------------------
  // Tony Saad (just in case you need to blame someone)
  class ShunnMoin : public VelocityInitBase {

  public:

    ShunnMoin(){}
    ~ShunnMoin(){}

    void problemSetup( ProblemSpecP db ){

      ProblemSpecP db_prop = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("Properties")->findBlock("ColdFlow");
      db_prop->findBlock("stream_0")->getAttribute("density",_rho0);
      db_prop->findBlock("stream_1")->getAttribute("density",_rho1);
      db->require("w", _w);
      db->require("k", _k);
      db->require("plane",_plane);
      db->getWithDefault("uf",_uf,0.0);
      db->getWithDefault("vf",_vf,0.0);

    }

    void setXVel( const Patch* patch, SFCXVariable<double>& uvel, constCCVariable<double>& rho ){

      double x,y,z;
      Vector Dx = patch->dCell();
      double dx_2 = Dx.x()/2.;
      const double pi = acos(-1.);

      for (CellIterator iter=patch->getSFCXIterator(); !iter.done(); iter++){

        IntVector c = *iter;
        Uintah::Point position = patch->getCellPosition(c);
        x = position.x() - dx_2;
        y = position.y();
        z = position.z();
        double rho_face = (rho[c] + rho[c-IntVector(1,0,0)])/2.0;

        if ( _plane == "x-y" ){

          double xhat = _k * pi * x;
          double yhat = _k * pi * y;
          double that = _w * pi * _time;

          uvel[c] = get_u_vel(xhat, yhat, that, rho_face);

        } else if ( _plane == "z-x" ){

          double xhat = _k * pi * z;
          double yhat = _k * pi * x;
          double that = _w * pi * _time;

          uvel[c] = get_v_vel(xhat, yhat, that, rho_face);

        } else {

          uvel[c] = 0.;

        }

      }

    }

    void setYVel( const Patch* patch, SFCYVariable<double>& vvel, constCCVariable<double>& rho ){

      double x,y,z;
      Vector Dx = patch->dCell();
      const double dy_2 = Dx.y()/2.0;

      const double pi = acos(-1.);

      for (CellIterator iter=patch->getSFCYIterator(); !iter.done(); iter++){

        IntVector c = *iter;
        Uintah::Point position = patch->getCellPosition(c);
        x = position.x();
        y = position.y() - dy_2;
        z = position.z();

        double rho_face = (rho[c]+rho[c-IntVector(0,1,0)])/2.;

        if ( _plane == "x-y" ){

          double xhat = _k * pi * x;
          double yhat = _k * pi * y;
          double that = _w * pi * _time;

          vvel[c] = get_v_vel(xhat, yhat, that, rho_face);

        } else if ( _plane == "y-z" ){

          double xhat = _k * pi * y;
          double yhat = _k * pi * z;
          double that = _w * pi * _time;

          vvel[c] = get_u_vel(xhat, yhat, that, rho_face);

        } else {

          vvel[c] = 0.;

        }


      }
    }

    void setZVel( const Patch* patch, SFCZVariable<double>& wvel, constCCVariable<double>& rho ){

      double x,y,z;
      Vector Dx = patch->dCell();
      const double dz_2 = Dx.z()/2.0;
      const double pi = acos(-1.);

      for (CellIterator iter=patch->getSFCZIterator(); !iter.done(); iter++){

        IntVector c = *iter;
        Uintah::Point position = patch->getCellPosition(c);
        x = position.x();
        y = position.y();
        z = position.z() - dz_2;

        double rho_face = (rho[c]+rho[c-IntVector(0,0,1)])/2.;

        if ( _plane == "y-z" ){

          double xhat = _k * pi * y;
          double yhat = _k * pi * z;
          double that = _w * pi * _time;

          wvel[c] = get_v_vel(xhat, yhat, that, rho_face);

        } else if ( _plane == "z-x" ){

          double xhat = _k * pi * z;
          double yhat = _k * pi * x;
          double that = _w * pi * _time;

          wvel[c] = get_u_vel(xhat, yhat, that, rho_face);

        } else {

          wvel[c] = 0.;

        }


      }
    }

  private:

    double _rho0;  //when f = 1 (different than Shunn's paper)
    double _rho1;
    double _uf, _vf;

    double _w;
    double _k;
    double _time;

    std::string _plane;

    inline double get_u_vel( const double xhat, const double yhat, const double that, const double rho_face ){
      return ( _rho0 - _rho1 )/rho_face * (-_w/(4.*_k))*cos(xhat)*sin(yhat)*sin(that);
    }

    inline double get_v_vel( const double xhat, const double yhat, const double that, const double rho_face ){
      return ( _rho0 - _rho1 )/rho_face  * (-_w/(4.*_k))*sin(xhat)*cos(yhat)*sin(that);
    }


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

    void setXVel( const Patch* patch, SFCXVariable<double>& uvel, constCCVariable<double>& rho ){

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


    void setYVel( const Patch* patch, SFCYVariable<double>& vvel, constCCVariable<double>& rho ){

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

    void setZVel( const Patch* patch, SFCZVariable<double>& wvel, constCCVariable<double>& rho ){

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
