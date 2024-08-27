/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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

#include <CCA/Components/OnTheFlyAnalysis/SGS_ReynoldsStress.h>

#include <CCA/Components/ICE/Core/ICELabel.h>
#include <CCA/Components/ICE/Materials/ICEMaterial.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Util/DebugStream.h>

#include <sci_defs/visit_defs.h>

#include <sys/stat.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <cstdio>


using namespace Uintah;
using namespace std;
//__________________________________
//  To turn on the output

Dout dout_OTF_SGSRS("SGS_ReynoldsStress",     "OnTheFlyAnalysis", "Task scheduling and execution.", false);

namespace Uintah {
//______________________________________________________________________
SGS_ReynoldsStress::SGS_ReynoldsStress(const ProcessorGroup   * myworld,
                                       const MaterialManagerP   materialManager,
                                       const ProblemSpecP     & module_spec)
  : AnalysisModule(myworld, materialManager, module_spec)
{
  I_lb  = scinew ICELabel();

  SGS_ReynoldsStressLabel = VarLabel::create("SGS_ReynoldsStress", CCVariable<Matrix3>::getTypeDescription());
}

//______________________________________________________________________
//
SGS_ReynoldsStress::~SGS_ReynoldsStress()
{
  DOUTR(dout_OTF_SGSRS, "Doing destructor SGS_ReynoldsStress");

  if(m_matl_set && m_matl_set->removeReference()) {
    delete m_matl_set;
  }

  VarLabel::destroy(SGS_ReynoldsStressLabel);
  delete I_lb;
}

//______________________________________________________________________
//     P R O B L E M   S E T U P
void SGS_ReynoldsStress::problemSetup(const ProblemSpecP& ,
                                      const ProblemSpecP& ,
                                      GridP             & grid,
                                      std::vector<std::vector<const VarLabel* > > &PState,
                                      std::vector<std::vector<const VarLabel* > > &PState_preReloc)
{
  DOUTR(dout_OTF_SGSRS, "Doing SGS_ReynoldsStress::problemSetup");

  // determine which material index to compute
  if(m_module_spec->findBlock("material") ){
    m_matl = m_materialManager->parseAndLookupMaterial(m_module_spec, "material");
  }
  else {
    throw ProblemSetupException("ERROR:AnalysisModule:SGS_ReynoldsStress: Missing <material> tag. \n", __FILE__, __LINE__);
  }

  vector<int> m(1);
  m[0] = m_matl->getDWIndex();
  m_matl_set = scinew MaterialSet();
  m_matl_set->addAll(m);
  m_matl_set->addReference();
  m_matl_sub = m_matl_set->getUnion();

}

//______________________________________________________________________
//
void SGS_ReynoldsStress::scheduleDoAnalysis(SchedulerP  & sched,
                                            const LevelP& level)
{
  printSchedule( level, dout_OTF_SGSRS,"SGS_ReynoldsStress::scheduleDoAnalysis" );

  Task* t = scinew Task("SGS_ReynoldsStress::doAnalysis",
                   this,&SGS_ReynoldsStress::doAnalysis);

  Ghost::GhostType m_gac = Ghost::AroundCells;

  t->requires( Task::NewDW, I_lb->tau_X_FCLabel, m_matl_sub, m_gac,1);
  t->requires( Task::NewDW, I_lb->tau_Y_FCLabel, m_matl_sub, m_gac,1);
  t->requires( Task::NewDW, I_lb->tau_Z_FCLabel, m_matl_sub, m_gac,1);


  t->computes( SGS_ReynoldsStressLabel, m_matl_sub);
  sched->addTask(t, level->eachPatch(), m_matl_set);
}

//______________________________________________________________________
// Compute the SGS_ReynoldsStress field by interpolating the
// face-centered shear stress terms to the cell center
void SGS_ReynoldsStress::doAnalysis(const ProcessorGroup * pg,
                                    const PatchSubset    * patches,
                                    const MaterialSubset * matl_sub ,
                                    DataWarehouse        * old_dw,
                                    DataWarehouse        * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dout_OTF_SGSRS,"Doing SGS_ReynoldsStress::doAnalysis");

    constSFCXVariable<Vector> tau_X_FC;
    constSFCYVariable<Vector> tau_Y_FC;
    constSFCZVariable<Vector> tau_Z_FC;
    CCVariable<Matrix3> SGS_ReynoldsStress;

    int indx = m_matl->getDWIndex();
    new_dw->get(tau_X_FC,  I_lb->tau_X_FCLabel, indx, patch, m_gac, 1);
    new_dw->get(tau_Y_FC,  I_lb->tau_Y_FCLabel, indx, patch, m_gac, 1);
    new_dw->get(tau_Z_FC,  I_lb->tau_Z_FCLabel, indx, patch, m_gac, 1);

    new_dw->allocateAndPut( SGS_ReynoldsStress, SGS_ReynoldsStressLabel, indx, patch );

    SGS_ReynoldsStress.initialize( Matrix3(0.0) );

    interpolateTauComponents( patch, tau_X_FC, tau_Y_FC, tau_Z_FC, SGS_ReynoldsStress);
  }  // patches
}


/*---------------------------------------------------------------------
 Purpose:   This function interpolates the shear stress
            tau_xx, ta_xy, tau_xz
            tau_yx, ta_yy, tau_yz
            tau_zx, ta_zy, tau_zz

            to the cell center

 ---------------------------------------------------------------------  */
void SGS_ReynoldsStress::interpolateTauComponents( const Patch* patch,
                                                   constSFCXVariable<Vector>& tau_X_FC,
                                                   constSFCYVariable<Vector>& tau_Y_FC,
                                                   constSFCZVariable<Vector>& tau_Z_FC,
                                                   CCVariable<Matrix3>      & SGS_ReynoldsStress )
{
  //__________________________________
  //  bullet proofing against AMR
  const Level* level = patch->getLevel();
  if (level->getIndex() > 0) {
    throw InternalError("AMRICE:computeTauX, computational footprint "
                        " has not been tested ", __FILE__, __LINE__ );
  }

  Vector dx = patch->dCell();
  //__________________________________
  //  Compute over the entire domain the different components
  CellIterator hi_lo = patch->getSFCXIterator();
  IntVector low = hi_lo.begin();
  IntVector hi  = hi_lo.end();
  hi[0] += patch->getBCType(patch->xplus) ==Patch::Neighbor?1:0;
  CellIterator X_iterLimits( low,hi );
  interpolateTauX_driver( X_iterLimits, dx, tau_X_FC, SGS_ReynoldsStress);


  hi_lo = patch->getSFCYIterator();
  low   = hi_lo.begin();
  hi    = hi_lo.end();
  hi[1] += patch->getBCType(patch->yplus) ==Patch::Neighbor?1:0;
  CellIterator Y_iterLimits( low,hi );
  interpolateTauY_driver( Y_iterLimits, dx, tau_Y_FC, SGS_ReynoldsStress);


  hi_lo = patch->getSFCZIterator();
  low   = hi_lo.begin();
  hi    = hi_lo.end();
  hi[2] += patch->getBCType(patch->zplus) ==Patch::Neighbor?1:0;
  CellIterator Z_iterLimits( low,hi );
  interpolateTauZ_driver( Z_iterLimits, dx, tau_Z_FC, SGS_ReynoldsStress);
}


/*---------------------------------------------------------------------
 Purpose:  Computes shear stress tau_xx, ta_xy, tau_xz

 ---------------------------------------------------------------------  */
void SGS_ReynoldsStress::interpolateTauX_driver( CellIterator iterLimits,
                                                 const Vector dx,
                                                 constSFCXVariable<Vector>& tau_X,
                                                 CCVariable<Matrix3>      & SGS_ReynoldsStress)
{
  //__________________________________
  //  Loop over the left cell faces and perform linear interpolation
  //  to the cell center. This assumes constant dx:
  //
  //  f_CC = f_FC[L] + (( f_FC[R] - f_FC[L] )/dx ) * dx/2
  //
  //       = f_FC[L] + (( f_FC[R] - f_FC[L] )/2 )
  
  for(CellIterator iter = iterLimits;!iter.done();iter++){
    IntVector c = *iter;
    
    IntVector R   = c + IntVector(1,0,0);   // right
    IntVector L   = c;                      // left

    Vector tau_X_R = tau_X[R];              // access the array once
    Vector tau_X_L = tau_X[L];
    
    Vector Delta = ( tau_X_R - tau_X_L )/2.0;
    
    //__________________________________
    //  tau_XX_CC
    double tau_XX_CC = tau_X_L.x() + Delta.x();
    
    //__________________________________
    //  tau_XY_CC
    double tau_XY_CC = tau_X_L.y() + Delta.y();

    //__________________________________
    //  tau_XZ_CC
    double tau_XZ_CC = tau_X_L.z() + Delta.z();

    SGS_ReynoldsStress[c].set( 0,0, tau_XX_CC );
    SGS_ReynoldsStress[c].set( 0,1, tau_XY_CC );
    SGS_ReynoldsStress[c].set( 0,2, tau_XZ_CC );
  }
}


/*---------------------------------------------------------------------
 Purpose:   This function computes shear stress tau_YY, ta_yx, tau_yz

 ---------------------------------------------------------------------  */
void SGS_ReynoldsStress::interpolateTauY_driver( CellIterator iterLimits,
                                                 const Vector dx,
                                                 constSFCYVariable<Vector>& tau_Y,
                                                 CCVariable<Matrix3>      & SGS_ReynoldsStress)
{
  //__________________________________
  //  Loop over the bottom cell faces and perform linear interpolation
  //  to the cell center. This assumes constant dx:
  //
  //  f_CC = f_FC[B] + (( f_FC[T] - f_FC[B] )/dx ) * dx/2
  //
  //       = f_FC[B] + (( f_FC[T] - f_FC[B] )/2 )

  for(CellIterator iter = iterLimits;!iter.done();iter++){
    IntVector c = *iter;

    IntVector T   = c + IntVector(0,1,0);   // top
    IntVector B   = c;                      // bottom
    
    Vector tau_Y_T = tau_Y[T];              // access the array once
    Vector tau_Y_B = tau_Y[B];

    Vector Delta = ( tau_Y_T - tau_Y_B )/2.0;
    
    //__________________________________
    //  tau_YX_CC
    double tau_YX_CC = tau_Y_B.x() + Delta.x();
    
    //__________________________________
    //  tau_YY_CC
    double tau_YY_CC = tau_Y_B.y() + Delta.y();

    //__________________________________
    //  tau_YZ_CC
    double tau_YZ_CC = tau_Y_B.z() + Delta.z();

    SGS_ReynoldsStress[c].set( 1,0, tau_YX_CC );
    SGS_ReynoldsStress[c].set( 1,1, tau_YY_CC );
    SGS_ReynoldsStress[c].set( 1,2, tau_YZ_CC );
  }
}

/*---------------------------------------------------------------------
 Purpose:   This function computes shear stress tau_zx, ta_zy, tau_zz

 ---------------------------------------------------------------------  */
void SGS_ReynoldsStress::interpolateTauZ_driver( CellIterator iterLimits,
                                                 const Vector dx,
                                                 constSFCZVariable<Vector>& tau_Z,
                                                 CCVariable<Matrix3>      & SGS_ReynoldsStress )
{
   //__________________________________
  //  Loop over the back cell faces and perform linear interpolation
  //  to the cell center. This assumes constant dx:
  //
  //  f_CC = f_FC[BCK] + (( f_FC[FRT] - f_FC[BCK] )/dx ) * dx/2
  //
  //       = f_FC[BCK] + (( f_FC[FRT] - f_FC[BCK] )/2 )

  for(CellIterator iter = iterLimits;!iter.done();iter++){
    IntVector c = *iter;

    IntVector frt = c + IntVector(0,0,1);   // front
    IntVector bck = c;                      // back
    
    Vector tau_Z_FRT = tau_Z[frt];          // access the array once
    Vector tau_Z_BCK = tau_Z[bck];

    Vector Delta = ( tau_Z_FRT - tau_Z_BCK )/2.0;
    
    //__________________________________
    //  tau_ZX_CC
    double tau_ZX_CC = tau_Z_BCK.x() + Delta.x();
    
    //__________________________________
    //  tau_ZY_CC
    double tau_ZY_CC = tau_Z_BCK.y() + Delta.y();

    //__________________________________
    //  tau_ZZ_CC
    double tau_ZZ_CC = tau_Z_BCK.z() + Delta.z();


    SGS_ReynoldsStress[c].set( 2,0, tau_ZX_CC );
    SGS_ReynoldsStress[c].set( 2,1, tau_ZY_CC );
    SGS_ReynoldsStress[c].set( 2,2, tau_ZZ_CC );
//  cout<<"tau_ZX: "<<tau_Z_FC[cell].x()<<
//        " tau_ZY: "<<tau_Z_FC[cell].y()<<
//        " tau_ZZ: "<<tau_Z_FC[cell].z()<<endl;
  }
}
}  // using namespace Uintah
