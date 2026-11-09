/*
 * The MIT License
 *
 * Copyright (c) 1997-2026 The University of Utah
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


//______________________________________________________________________
//  SPICE: Semi-implicit Pressure Iterated Compressible EOS.  See SPICE.h.
//
//  The task bodies below marked "Copied from ICE::<name>" started as the
//  ICE code of the PIMPLE/KT development (commit 0cb7e2c071 plus the pure
//  ICE parts of 4bd76cca21 and c7ccb2c5a6).  Keep them diffable against
//  their ICE originals so ICE fixes can be ported.

#include <CCA/Components/SPICE/SPICE.h>
#include <CCA/Components/SPICE/Advection/CentralUpwindAdvector.h>
#include <CCA/Components/SPICE/Core/SPICELabel.h>

#include <CCA/Components/ICE/ICE.h>
#include <CCA/Components/ICE/Advection/AdvectionFactory.h>
#include <CCA/Components/ICE/Core/ConservationTest.h>
#include <CCA/Components/ICE/Core/Diffusion.h>
#include <CCA/Components/ICE/Core/ICELabel.h>
#include <CCA/Components/ICE/CustomBCs/BoundaryCond.h>
#include <CCA/Components/ICE/CustomBCs/C_BC_driver.h>
#include <CCA/Components/ICE/EOS/EquationOfState.h>
#include <CCA/Components/ICE/Materials/ICEMaterial.h>
#include <CCA/Components/ICE/TurbulenceModel/Turbulence.h>
#include <CCA/Components/Models/FluidsBased/FluidsBasedModel.h>
#include <CCA/Components/Models/MultiMatlExchange/ExchangeModel.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/LoadBalancer.h>
#include <CCA/Ports/Output.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/Utils.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MiscMath.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DOUT.hpp>

#include <cfloat>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

using namespace std;
using namespace Uintah;
using namespace ExchangeModels;

namespace {
  Dout m_spice_tasks(  "SPICE_tasks",  "SPICE", "Print task scheduling & execution", false );
  Dout m_spice_pimple( "SPICE_PIMPLE", "SPICE", "Print PIMPLE outer-loop iteration residual", false );
}

//______________________________________________________________________
//
SPICE::SPICE( const ProcessorGroup  * myworld,
              const MaterialManagerP  materialManager,
              const bool              /*doAMR*/ )
  : ApplicationCommon( myworld, materialManager )
{
  //  ICE shares SPICE's material manager, as it does with MPMICE
  d_ice = scinew ICE( myworld, m_materialManager );
  lb    = d_ice->lb;
  Slb   = scinew SPICELabel();
}

//______________________________________________________________________
//
SPICE::~SPICE()
{
  //  the PIMPLE sub-graph holds tasks bound to both SPICE and d_ice
  d_pimpleSched = nullptr;

  if( d_ktICE_matl && d_ktICE_matl->removeReference() ){
    delete d_ktICE_matl;
  }

  for( auto label : d_pimpleTvarAdv0Labels ){
    VarLabel::destroy( label );
  }

  d_ice->releaseComponents();
  delete d_ice;
  delete Slb;
}

//______________________________________________________________________
//
double SPICE::recomputeDelT( const double delT )
{
  return d_ice->recomputeDelT( delT );
}

//______________________________________________________________________
//  Advection operator factory handed to ICE.  ICE calls it from its own
//  problemSetup with the <CFD><SPICE> block.
Advector* SPICE::createAdvector( ProblemSpecP & cfd_ps,
                                 bool         & useCompatibleFluxes,
                                 int          & orderOfAdvection )
{
  ProblemSpecP advect_ps = cfd_ps->findBlock("advection");
  if( !advect_ps ){
    throw ProblemSetupException("Cannot find advection tag", __FILE__, __LINE__);
  }

  map<string,string> advect_options;
  advect_ps->getAttributes( advect_options );

  if( advect_options["type"] != "CentralUpwind" ){
    if( advect_options.count("limiter") ){
      throw ProblemSetupException(
        "\n ERROR: <advection limiter=\"...\"> only applies to type=\"CentralUpwind\"\n",
        __FILE__, __LINE__);
    }
    return AdvectionFactory::create( cfd_ps, useCompatibleFluxes, orderOfAdvection );
  }

  //__________________________________
  //  CentralUpwind.  Same attribute handling as ICE's AdvectionFactory,
  //  plus the optional limiter.
  if( advect_options["useCompatibleFluxes"] == "false" ||
      advect_options["compatibleFluxes"]    == "false" ){
    useCompatibleFluxes = false;
    proc0cout << "\n--------SPICE::Warning:  You've turned off compatible fluxes.\n"<< endl;
  }

  for( auto & opt : advect_options ){
    const string & name = opt.first;
    if( name != "type" && name != "limiter" &&
        name != "useCompatibleFluxes" && name != "compatibleFluxes" ){
      throw ProblemSetupException(
        "\n ERROR: unknown <advection> attribute (" + name + ")\n", __FILE__, __LINE__);
    }
  }

  string limName = "vanAlbada";
  if( advect_options.count("limiter") ){
    limName = advect_options["limiter"];
  }

  centralUpwind::Limiter lim;
  if( !centralUpwind::limiterFromString( limName, lim ) ){
    throw ProblemSetupException(
      "\n ERROR: <advection type=\"CentralUpwind\"> limiter '" + limName +
      "' is not recognized.  Valid values: vanAlbada, vanLeer, minmod, MC, upwind\n",
      __FILE__, __LINE__);
  }

  orderOfAdvection = 2;
  return scinew CentralUpwindAdvector( lim );
}

//______________________________________________________________________
//
void SPICE::problemSetup( const ProblemSpecP & prob_spec,
                          const ProblemSpecP & restart_prob_spec,
                          GridP              & grid )
{
  if( isAMR() || grid->numLevels() > 1 ){
    throw ProblemSetupException(
      "\n ERROR: SPICE does not support AMR yet (an <AMR> block or more than one level)\n",
      __FILE__, __LINE__);
  }

  //__________________________________
  //  Set up ICE the way MPMICE does: hand it this component's ports, let
  //  the base class read the <Time>/<AMR> settings into it, then point it
  //  at the <CFD><SPICE> block and at SPICE's advection factory.
  d_ice->setComponents( this );
  dynamic_cast<ApplicationCommon*>( d_ice )->problemSetup( prob_spec );

  d_ice->setCFDBlockName( "SPICE" );
  d_ice->setAdvectorFactory(
    [this]( ProblemSpecP & ps, bool & useCompatibleFluxes, int & order ){
      return createAdvector( ps, useCompatibleFluxes, order );
    } );

  d_ice->problemSetup( prob_spec, restart_prob_spec, grid );

  ProblemSpecP cfd_spice_ps = prob_spec->findBlock("CFD")->findBlock("SPICE");

  //__________________________________
  //  CentralUpwind (Kurganov-Tadmor) options.  createAdvector() already
  //  validated the <advection> attributes.
  {
    ProblemSpecP adv_ps = cfd_spice_ps->findBlock("advection");
    string advType;
    adv_ps->getAttribute( "type", advType );

    if( advType == "CentralUpwind" ){
      d_useCentralUpwind = true;

      string limiter;
      if( adv_ps->getAttribute( "limiter", limiter ) ){
        d_ktLimiterName = limiter;
      }
    }

    //__________________________________
    //  Optional <CentralUpwind> block with the Kurganov-Tadmor scheme details
    ProblemSpecP cu_ps = cfd_spice_ps->findBlock("CentralUpwind");
    if( cu_ps ){
      if( !d_useCentralUpwind ){
        throw ProblemSetupException(
          "\n ERROR: <CentralUpwind> requires <advection type=\"CentralUpwind\">\n",
          __FILE__, __LINE__);
      }

      cu_ps->getWithDefault( "fluxScheme",     d_ktFluxScheme,     string("Kurganov") );
      cu_ps->getWithDefault( "kappaFunction",  d_ktKappaFunction,  string("MachToAcCourant") );
      cu_ps->getWithDefault( "faceVelocity",   d_ktFaceVel,      true );
      cu_ps->getWithDefault( "facePressure",   d_ktFacePress,    true );
      cu_ps->getWithDefault( "ktPressureFlux", d_ktPressureFlux, true );
    }

    //  Convert the strings to enums once here so the per-face math never
    //  re-parses strings.
    if( d_useCentralUpwind ){
      if( !centralUpwind::limiterFromString( d_ktLimiterName, d_ktLimiter ) ){
        throw ProblemSetupException(
          "\n ERROR: <advection type=\"CentralUpwind\"> unrecognized limiter\n",
          __FILE__, __LINE__);
      }
      if( !centralUpwind::fluxSchemeFromString( d_ktFluxScheme, d_ktScheme ) ){
        throw ProblemSetupException(
          "\n ERROR: <CentralUpwind><fluxScheme> must be 'Kurganov' or 'Tadmor'\n",
          __FILE__, __LINE__);
      }
      if( !centralUpwind::kappaModeFromString( d_ktKappaFunction, d_ktKappaMode ) ){
        throw ProblemSetupException(
          "\n ERROR: <CentralUpwind><kappaFunction> must be 'MachToAcCourant' or 'one'\n",
          __FILE__, __LINE__);
      }
    }
  }

  //__________________________________
  //  Optional <PIMPLE> block
  ProblemSpecP pimple_ps = cfd_spice_ps->findBlock("PIMPLE");
  if( pimple_ps ){
    pimple_ps->getWithDefault( "nOuterCorrectors", d_pimpleOuter, 1 );
    if( d_pimpleOuter < 1 ){
      throw ProblemSetupException(
        "\n ERROR: <PIMPLE><nOuterCorrectors> must be >= 1\n", __FILE__, __LINE__);
    }

    pimple_ps->getWithDefault( "nCorrectors", d_pimpleNCorr, 1 );
    if( d_pimpleNCorr < 1 ){
      throw ProblemSetupException(
        "\n ERROR: <PIMPLE><nCorrectors> must be >= 1\n", __FILE__, __LINE__);
    }

    pimple_ps->getWithDefault( "theta", d_pimpleTheta, 0.5 );
    if( d_pimpleTheta <= 0.0 || d_pimpleTheta > 1.0 ){
      throw ProblemSetupException(
        "\n ERROR: <PIMPLE><theta> must satisfy 0 < theta <= 1\n", __FILE__, __LINE__);
    }

    pimple_ps->getWithDefault( "outerTolerance", d_pimpleTol, 0.0 );
    if( d_pimpleTol < 0.0 ){
      throw ProblemSetupException(
        "\n ERROR: <PIMPLE><outerTolerance> must be >= 0\n", __FILE__, __LINE__);
    }
  }

  //__________________________________
  //  Bulletproofing that needs ICE's setup (impICE, materials, models)
  if( m_materialManager->getNumMatls( "MPM" ) > 0 ){
    throw ProblemSetupException(
      "\n ERROR: SPICE does not support MPM materials\n", __FILE__, __LINE__);
  }

  //  SPICE is a single-fluid code: one ICE material, with any species carried
  //  as transported scalars (e.g. gasCombustion), as in reactingPimpleCentralFoam.
  //  The KT face reconstruction and the PIMPLE blend are single-material, and
  //  the multi-material f_theta split of the pressure work is untested here.
  //  Use ICE for multi-material problems.
  if( m_materialManager->getNumMatls( "ICE" ) != 1 ){
    ostringstream msg;
    msg << "\n ERROR: SPICE requires exactly one ICE material (found "
        << m_materialManager->getNumMatls( "ICE" ) << ").  Use ICE for multi-material problems.\n";
    throw ProblemSetupException( msg.str(), __FILE__, __LINE__ );
  }

  if( d_pimpleOuter > 1 ){
    if( d_ice->d_impICE ){
      throw ProblemSetupException(
        "\n ERROR: PIMPLE outer loop with the implicit pressure solve is not supported yet\n",
        __FILE__, __LINE__);
    }

    // The PIMPLE sub-graph re-runs scheduleFluidChain() and recomputes the
    // LODI maxMach_<face> values, but not the turbulence variance, which
    // scheduleTimeAdvance computes outside the fluid chain.
    if( d_ice->d_turbulence ){
      throw ProblemSetupException(
        "\n ERROR: PIMPLE outer loop with a turbulence model is not "
        "supported yet (the sub-graph never recomputes the scalar variance "
        "that scheduleViscousShearStress reads)\n", __FILE__, __LINE__);
    }
  }

  if( d_useCentralUpwind ){
    if( d_ice->d_impICE ){
      throw ProblemSetupException(
        "\n ERROR: <advection type=\"CentralUpwind\"> (Kurganov-Tadmor) with the implicit pressure solve is not supported yet\n",
        __FILE__, __LINE__);
    }

    d_ktICE_matl = scinew MaterialSubset();
    d_ktICE_matl->add( m_materialManager->getMaterial( "ICE", 0 )->getDWIndex() );
    d_ktICE_matl->addReference();
  }

  //__________________________________
  //  Startup banner
  proc0cout << "__________________________________ \n"
            << " SPICE: new pressure work formulation (always on)\n";
  if( d_useCentralUpwind ){
    proc0cout << " CentralUpwind advection: "
              << " limiter: "        << d_ktLimiterName
              << " fluxScheme: "     << d_ktFluxScheme
              << " kappaFunction: "  << d_ktKappaFunction
              << " faceVelocity: "   << d_ktFaceVel
              << " facePressure: "   << d_ktFacePress
              << " ktPressureFlux: " << d_ktPressureFlux << "\n";
  }
  if( d_pimpleOuter > 1 ){
    proc0cout << " PIMPLE outer loop: "
              << " nOuterCorrectors: " << d_pimpleOuter
              << " nCorrectors: "      << d_pimpleNCorr
              << " theta: "            << d_pimpleTheta
              << " outerTolerance: "   << d_pimpleTol << "\n";
  }
  proc0cout << "__________________________________ \n";

  //__________________________________
  //  PIMPLE outer-loop sub-scheduler, created exactly like ICE's implicit
  //  pressure sub-scheduler.
  if( d_pimpleOuter > 1 ){
    d_pimpleSched = m_scheduler->createSubScheduler();
    d_pimpleSched->initialize(3,1);
    d_pimpleSched->clearMappings();
    d_pimpleSched->mapDataWarehouse( Task::ParentOldDW, 0 );
    d_pimpleSched->mapDataWarehouse( Task::ParentNewDW, 1 );
    d_pimpleSched->mapDataWarehouse( Task::OldDW,       2 );
    d_pimpleSched->mapDataWarehouse( Task::NewDW,       3 );
    // single-level convention (matches SchedulerCommon's own default
    // mapping): Coarse == Old/New so getSubCycleProgress() degenerates
    // safely.
    d_pimpleSched->mapDataWarehouse( Task::CoarseOldDW, 2 );
    d_pimpleSched->mapDataWarehouse( Task::CoarseNewDW, 3 );

    d_recompilePimpleSched = true;

    //  One "*_adv0" capture label per transported variable, in the nested
    //  (model, transported variable) order every loop over d_models uses.
    for( auto& model : d_ice->d_models ){
      FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( model );
      if( fb_model ){
        for( auto& tvar : fb_model->d_transVars ){
          d_pimpleTvarAdv0Labels.push_back(
            VarLabel::create( tvar->var->getName() + "_adv0",
                              tvar->var->typeDescription() ) );
        }
      }
    }
  }
}

//______________________________________________________________________
//  Forwarded to ICE unchanged
void SPICE::outputProblemSpec( ProblemSpecP & root_ps )
{
  d_ice->outputProblemSpec( root_ps );
}

void SPICE::scheduleInitialize( const LevelP & level,
                                SchedulerP   & sched )
{
  d_ice->scheduleInitialize( level, sched );
}

void SPICE::scheduleRestartInitialize( const LevelP & level,
                                       SchedulerP   & sched )
{
  d_ice->scheduleRestartInitialize( level, sched );
}

void SPICE::scheduleComputeStableTimeStep( const LevelP & level,
                                           SchedulerP   & sched )
{
  d_ice->scheduleComputeStableTimeStep( level, sched );
}

void SPICE::scheduleFinalizeTimestep( const LevelP & level,
                                      SchedulerP   & sched )
{
  d_ice->scheduleFinalizeTimestep( level, sched );
}

void SPICE::scheduleAnalysis( const LevelP & level,
                              SchedulerP   & sched )
{
  d_ice->scheduleAnalysis( level, sched );
}

/* _____________________________________________________________________
 Task:  SPICE::scheduleTimeAdvance--
 Purpose:  Schedule all tasks for computing a single timestep.
           Copied from ICE::scheduleTimeAdvance (keep the two in step).
_____________________________________________________________________*/
void
SPICE::scheduleTimeAdvance( const LevelP & level,
                            SchedulerP   & sched)
{
  printSchedule( level, m_spice_tasks, " --------------------------------------------------------");
  printSchedule( level, m_spice_tasks, " SPICE::scheduleTimeAdvance" );

  const PatchSet* patches = level->eachPatch();
  const MaterialSet* ice_matls = m_materialManager->allMaterials( "ICE" );
  const MaterialSet* all_matls = m_materialManager->allMaterials();

  const MaterialSubset* ice_mss = (ice_matls ? ice_matls->getUnion() : nullptr);

  if(d_ice->d_turbulence){
    // The turblence model is also called directly from
    // accumlateMomentumSourceSinks.
    d_ice->d_turbulence->scheduleComputeVariance(sched, patches, ice_matls);
  }

  d_ice->scheduleMaxMach_on_Lodi_BC_Faces( sched, level,   ice_matls);

  scheduleFluidChain( sched, level, false );

  if( d_pimpleOuter > 1 ){
    // snapshot the k=0 (main graph) advected conserved state so the outer
    // loop's sub-graph can read it from ParentNewDW without also modifying
    // the same label (mass_adv/mom_adv/eng_adv/sp_vol_adv/tvar->var_adv)
    // in its own NewDW
    scheduleCapturePimpleAdv0(             sched, level,   ice_matls);
  }

  d_ice->scheduleConservedtoPrimitive_Vars(sched, patches, ice_mss,
                                                           all_matls,
                                                           "afterAdvection");

  if( d_pimpleOuter > 1 ){
    schedulePimpleOuterLoop(               sched, level );
  }
}
/* _____________________________________________________________________
 Task:  SPICE::scheduleFluidChain--
 Purpose:  Schedule the fluid-dynamics portion of a single timestep, from
           the thermo/transport properties through the advection step,
           so it can be scheduled again for the PIMPLE outer loop.

           ICE tasks are scheduled through d_ice.  The tasks SPICE owns are
           scheduled here; computeVel_FC, computePressFC and
           computeDelPressAndUpdatePressCC use the SPICE copy only when the
           Kurganov-Tadmor option that changes them is on, and ICE's
           otherwise.  The flags are fixed at problemSetup, so the main
           graph and the PIMPLE sub-graph always choose the same way.
_____________________________________________________________________*/
void
SPICE::scheduleFluidChain( SchedulerP   & sched,
                           const LevelP & level,
                           const bool     inOuterLoop )
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* ice_matls = m_materialManager->allMaterials( "ICE" );
  const MaterialSet* mpm_matls = m_materialManager->allMaterials( "MPM" );
  const MaterialSet* all_matls = m_materialManager->allMaterials();

  MaterialSubset* press_matl = d_ice->d_press_matl;
  MaterialSubset* one_matl   = press_matl;
  const MaterialSubset* ice_mss = (ice_matls ? ice_matls->getUnion() : nullptr);
  const MaterialSubset* mpm_mss = (mpm_matls ? mpm_matls->getUnion() : nullptr);

  const bool ktFaceVel   = d_useCentralUpwind && d_ktFaceVel;
  const bool ktFacePress = d_useCentralUpwind && d_ktFacePress;

  d_ice->scheduleComputeThermoTransportProperties(sched, level,  ice_matls);

  d_ice->scheduleComputePressure(                sched, patches, press_matl,
                                                                 all_matls);

  // KT own/nei weights (alphaOwn, aSf, kappa): pure functions of
  // rho_CC/speedSound_CC/vel_CC, so they can be computed as soon as the
  // equilibration pressure step has produced rho_CC/speedSound_CC.
  scheduleComputeCentralWeights(                 sched, patches, ice_matls);

  d_ice->scheduleComputeTempFC(                  sched, patches, ice_mss,
                                                                 mpm_mss,
                                                                 all_matls);

  if( !inOuterLoop ){
    // chemistry/model sources are evaluated once per timestep, before the
    // PIMPLE outer loop (reactingPimpleCentralFoam: reaction->correct()
    // precedes pimple.loop())
    d_ice->scheduleComputeModelSources(          sched, level,   all_matls);
  }
  else {
    // PIMPLE outer iterations k>=1: model sources are FROZEN at the k=0
    // value -- copy them from ParentNewDW instead of recomputing them.
    schedulePimpleCopyModelSources(              sched, level,   all_matls);
  }

  d_ice->scheduleUpdateVolumeFraction(           sched, level,   press_matl,
                                                                 all_matls);

  if( ktFaceVel ){
    scheduleComputeVel_FC(                       sched, patches, ice_mss,
                                                                 mpm_mss,
                                                                 press_matl,
                                                                 all_matls);
  }
  else {
    d_ice->scheduleComputeVel_FC(                sched, patches, ice_mss,
                                                                 mpm_mss,
                                                                 press_matl,
                                                                 all_matls);
  }

  d_ice->d_exchModel->sched_PreExchangeTasks(    sched, patches, ice_mss,
                                                                 mpm_mss,
                                                                 all_matls);

  d_ice->d_exchModel->sched_AddExch_VelFC(       sched, patches, ice_mss,
                                                                 mpm_mss,
                                                                 all_matls,
                                                                 d_ice->d_BC_globalVars,
                                                                 false);

  // KT per-face mass fluxes (phiOwn, phiNei) and the KT volumetric face
  // velocity: need uvel_FCME (just computed above) and the weights from
  // scheduleComputeCentralWeights.
  scheduleComputeCentralFluxes(                  sched, patches, ice_matls);

  if(d_ice->d_impICE) {        //  I M P L I C I T  (KT and PIMPLE refuse impICE)

    d_ice->scheduleSetupRHS(                     sched, patches, one_matl,
                                                                 all_matls,
                                                                 false,
                                                                 "computes");

    d_ice->scheduleCompute_maxRHS(               sched, level,   one_matl,
                                                                 all_matls);

    d_ice->scheduleImplicitPressureSolve(        sched, level,   patches,
                                                                 one_matl,
                                                                 press_matl,
                                                                 ice_mss,
                                                                 mpm_mss,
                                                                 all_matls);

    d_ice->scheduleComputeDel_P(                 sched, level,   patches,
                                                                 one_matl,
                                                                 press_matl,
                                                                 all_matls);
  }

  if(!d_ice->d_impICE){         //  E X P L I C I T
    if( d_useCentralUpwind ){
      scheduleComputeDelPressAndUpdatePressCC(   sched, patches, press_matl,
                                                                 ice_mss,
                                                                 mpm_mss,
                                                                 all_matls);
    }
    else {
      d_ice->scheduleComputeDelPressAndUpdatePressCC( sched, patches, press_matl,
                                                                 ice_mss,
                                                                 mpm_mss,
                                                                 all_matls);
    }
  }

  if( ktFacePress ){
    scheduleComputePressFC(                      sched, patches, press_matl,
                                                                 all_matls);
  }
  else {
    d_ice->scheduleComputePressFC(               sched, patches, press_matl,
                                                                 all_matls);
  }

  d_ice->scheduleVelTau_CC(                      sched, patches, ice_matls);

  d_ice->scheduleViscousShearStress(             sched, patches, ice_matls);

  d_ice->scheduleAccumulateMomentumSourceSinks(  sched, patches, press_matl,
                                                                 ice_mss,
                                                                 mpm_mss,
                                                                 all_matls);

  scheduleAccumulateEnergySourceSinks(           sched, patches, ice_mss,
                                                                 mpm_mss,
                                                                 press_matl,
                                                                 all_matls);

  d_ice->scheduleComputeLagrangianValues(        sched, patches, all_matls);

  if( d_ice->d_anyModelOwnsCaloricEOS ){
    // must be scheduled before the exchange task: a model-owned caloric EOS
    // makes the exchange require the Lagrangian transported scalars, and a
    // NewDW requires only binds to computes of earlier-scheduled tasks
    d_ice->scheduleComputeLagrangian_Transported_Vars(sched, patches,
                                                                 all_matls);
  }

  d_ice->d_exchModel->sched_AddExch_Vel_Temp_CC( sched, patches, ice_mss,
                                                                 mpm_mss,
                                                                 all_matls,
                                                                 d_ice->d_BC_globalVars);

  if( !d_ice->d_anyModelOwnsCaloricEOS ){
    // original ordering: no model needs the exchange task to see
    // Lagrangian transported scalars, so run this after exchange as before
    d_ice->scheduleComputeLagrangian_Transported_Vars(sched, patches,
                                                                 all_matls);
  }

  d_ice->scheduleComputeLagrangianSpecificVolume(sched, patches, ice_mss,
                                                                 mpm_mss,
                                                                 press_matl,
                                                                 all_matls);

  scheduleAdvectAndAdvanceInTime(                sched, patches, ice_mss,
                                                                 all_matls);
}

/* _____________________________________________________________________
Task:    SPICE::scheduleComputeVel_FC--
Purpose: Compute the face centered velocities
_____________________________________________________________________*/
//  Copied from ICE::scheduleComputeVel_FC and modified for SPICE.  Keep it diffable against the
//  ICE original so ICE fixes can be ported.
void SPICE::scheduleComputeVel_FC(SchedulerP            & sched,
                                const PatchSet        * patches,
                                const MaterialSubset  * ice_mss,
                                const MaterialSubset  * mpm_mss,
                                const MaterialSubset  * press_mss,
                                const MaterialSet     * all_matls)
{
  printSchedule( patches, m_spice_tasks, " SPICE::scheduleComputeVel_FC" );

  Task* t = scinew Task("SPICE::computeVel_FC",
                  this, &SPICE::computeVel_FC);

  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.

  t->requiresVar( Task::OldDW, lb->delTLabel, getLevel(patches) );
  t->requiresVar( Task::NewDW, lb->press_equil_CCLabel, press_mss, oims, m_gac,1 );
  t->requiresVar( Task::NewDW, lb->sp_vol_CCLabel,    /*all_matls*/ m_gac,1 );
  t->requiresVar( Task::NewDW, lb->rho_CCLabel,       /*all_matls*/ m_gac,1 );

  // KT face velocity: the own/nei stencil reads vel_CC two cells out, so
  // use 2 ghost cells when it is on.  Must match the get() in
  // SPICE::computeVel_FC.
  int velCCGhostCells = ( d_useCentralUpwind && d_ktFaceVel ) ? 2 : 1;
  t->requiresVar( Task::OldDW, lb->vel_CCLabel,         ice_mss,    m_gac,velCCGhostCells );

  if( d_useCentralUpwind && d_ktFaceVel ){
    t->requiresVar( Task::NewDW, Slb->ktAlphaOwn_FCXLabel, ice_mss, m_gn,0 );
    t->requiresVar( Task::NewDW, Slb->ktAlphaOwn_FCYLabel, ice_mss, m_gn,0 );
    t->requiresVar( Task::NewDW, Slb->ktAlphaOwn_FCZLabel, ice_mss, m_gn,0 );
  }

  if( mpm_mss ){
    t->requiresVar( Task::NewDW,lb->vel_CCLabel,       mpm_mss,  m_gac,1 );
  }

  t->computesVar( lb->uvel_FCLabel );
  t->computesVar( lb->vvel_FCLabel );
  t->computesVar( lb->wvel_FCLabel );
  t->computesVar( lb->grad_P_XFCLabel );
  t->computesVar( lb->grad_P_YFCLabel );
  t->computesVar( lb->grad_P_ZFCLabel );
  sched->addTask(t, patches, all_matls);
}

/* _____________________________________________________________________
Task:     SPICE::scheduleComputeCentralWeights--
Purpose:  Compute the per-face Kurganov-Tadmor weights (alphaOwn, aSf)
          and the blending function kappa (PDF Section 4).  Scheduled
          after the equilibration pressure and before computeVel_FC,
          which uses alphaOwn.
_____________________________________________________________________*/
void SPICE::scheduleComputeCentralWeights(SchedulerP        & sched,
                                        const PatchSet    * patches,
                                        const MaterialSet * ice_matls)
{
  if( !d_useCentralUpwind ){
    return;
  }

  printSchedule( patches, m_spice_tasks, " SPICE::scheduleComputeCentralWeights" );

  Task* t = scinew Task("SPICE::computeCentralWeights",
                  this, &SPICE::computeCentralWeights);

  // same vel_CC source (OldDW) that computeVel_FC uses for ICE matls
  t->requiresVar( Task::OldDW, lb->delTLabel, getLevel(patches) );
  t->requiresVar( Task::NewDW, lb->rho_CCLabel,        m_gac,2 );
  t->requiresVar( Task::NewDW, lb->speedSound_CCLabel, m_gac,2 );
  t->requiresVar( Task::OldDW, lb->vel_CCLabel,        m_gac,2 );

  t->computesVar( Slb->ktAlphaOwn_FCXLabel );
  t->computesVar( Slb->ktAlphaOwn_FCYLabel );
  t->computesVar( Slb->ktAlphaOwn_FCZLabel );
  t->computesVar( Slb->ktaSf_FCXLabel );
  t->computesVar( Slb->ktaSf_FCYLabel );
  t->computesVar( Slb->ktaSf_FCZLabel );
  t->computesVar( Slb->ktKappa_FCXLabel );
  t->computesVar( Slb->ktKappa_FCYLabel );
  t->computesVar( Slb->ktKappa_FCZLabel );

  sched->addTask(t, patches, ice_matls);
}

/* _____________________________________________________________________
Task:     SPICE::scheduleComputeCentralFluxes--
Purpose:  Compute the KT per-face mass fluxes (phiOwn, phiNei) and the
          KT face velocity ktUvol_FC (PDF Section 5) from uvel_FCME and
          the computeCentralWeights output.  Scheduled after
          sched_AddExch_VelFC and before the pressure update.
_____________________________________________________________________*/
void SPICE::scheduleComputeCentralFluxes(SchedulerP        & sched,
                                       const PatchSet    * patches,
                                       const MaterialSet * ice_matls)
{
  if( !d_useCentralUpwind ){
    return;
  }

  printSchedule( patches, m_spice_tasks, " SPICE::scheduleComputeCentralFluxes" );

  Task* t = scinew Task("SPICE::computeCentralFluxes",
                  this, &SPICE::computeCentralFluxes);

  t->requiresVar( Task::NewDW, lb->uvel_FCMELabel,     m_gn,0 );
  t->requiresVar( Task::NewDW, lb->vvel_FCMELabel,     m_gn,0 );
  t->requiresVar( Task::NewDW, lb->wvel_FCMELabel,     m_gn,0 );
  t->requiresVar( Task::NewDW, Slb->ktAlphaOwn_FCXLabel,m_gn,0 );
  t->requiresVar( Task::NewDW, Slb->ktAlphaOwn_FCYLabel,m_gn,0 );
  t->requiresVar( Task::NewDW, Slb->ktAlphaOwn_FCZLabel,m_gn,0 );
  t->requiresVar( Task::NewDW, Slb->ktaSf_FCXLabel,     m_gn,0 );
  t->requiresVar( Task::NewDW, Slb->ktaSf_FCYLabel,     m_gn,0 );
  t->requiresVar( Task::NewDW, Slb->ktaSf_FCZLabel,     m_gn,0 );
  t->requiresVar( Task::NewDW, Slb->ktKappa_FCXLabel,   m_gn,0 );
  t->requiresVar( Task::NewDW, Slb->ktKappa_FCYLabel,   m_gn,0 );
  t->requiresVar( Task::NewDW, Slb->ktKappa_FCZLabel,   m_gn,0 );
  t->requiresVar( Task::NewDW, lb->rho_CCLabel,        m_gac,2 );

  t->computesVar( Slb->ktPhiOwn_FCXLabel );
  t->computesVar( Slb->ktPhiOwn_FCYLabel );
  t->computesVar( Slb->ktPhiOwn_FCZLabel );
  t->computesVar( Slb->ktPhiNei_FCXLabel );
  t->computesVar( Slb->ktPhiNei_FCYLabel );
  t->computesVar( Slb->ktPhiNei_FCZLabel );
  t->computesVar( Slb->ktUvol_FCXLabel );
  t->computesVar( Slb->ktUvol_FCYLabel );
  t->computesVar( Slb->ktUvol_FCZLabel );

  sched->addTask(t, patches, ice_matls);
}

/* _____________________________________________________________________
 Task:      SPICE::scheduleComputeDelPressAndUpdatePressCC--
 Purpose:   Compute the change in pressure (delP) and add it to the
            equilibration pressure.  Note there could be a change in pressure
            due to changes in mass.
_____________________________________________________________________*/
//  Copied from ICE::scheduleComputeDelPressAndUpdatePressCC and modified for SPICE.  Keep it diffable against the
//  ICE original so ICE fixes can be ported.
void SPICE::scheduleComputeDelPressAndUpdatePressCC(SchedulerP            & sched,
                                                  const PatchSet        * patches,
                                                  const MaterialSubset  * press_matl,
                                                  const MaterialSubset  * ice_matls,
                                                  const MaterialSubset  * /*mpm_matls*/,
                                                  const MaterialSet     * matls)
{
  printSchedule( patches, m_spice_tasks, " SPICE::scheduleComputeDelPressAndUpdatePressCC" );

  Task *task = scinew Task("SPICE::computeDelPressAndUpdatePressCC",this,
                           &SPICE::computeDelPressAndUpdatePressCC);


  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.

  task->requiresVar( Task::OldDW, lb->timeStepLabel );
  task->requiresVar( Task::OldDW, lb->simulationTimeLabel );
  task->requiresVar( Task::OldDW, lb->delTLabel,getLevel(patches) );
  task->requiresVar( Task::NewDW, lb->vol_frac_CCLabel,   m_gac,2 );
  task->requiresVar( Task::NewDW, lb->uvel_FCMELabel,     m_gac,2 );
  task->requiresVar( Task::NewDW, lb->vvel_FCMELabel,     m_gac,2 );
  task->requiresVar( Task::NewDW, lb->wvel_FCMELabel,     m_gac,2 );

  if( d_useCentralUpwind && d_ktPressureFlux ){
    // KT acoustic-dissipation term for the pEqn's convective term: same
    // ghost as uvel_FCME above, since computeCentralFluxes fills
    // ktUvol_FC over the same face range as computeVel_FC/uvel_FCME.
    task->requiresVar( Task::NewDW, Slb->ktUvol_FCXLabel, m_gac,2 );
    task->requiresVar( Task::NewDW, Slb->ktUvol_FCYLabel, m_gac,2 );
    task->requiresVar( Task::NewDW, Slb->ktUvol_FCZLabel, m_gac,2 );
  }

  task->requiresVar( Task::NewDW, lb->sp_vol_CCLabel,     m_gn );
  task->requiresVar( Task::NewDW, lb->rho_CCLabel,        m_gn );
  task->requiresVar( Task::NewDW, lb->speedSound_CCLabel, m_gn );
  task->requiresVar( Task::NewDW, lb->sumKappaLabel,      press_matl,oims,m_gn );
  task->requiresVar( Task::NewDW, lb->press_equil_CCLabel,press_matl,oims,m_gn );
  //__________________________________
  if(d_ice->d_models.size() > 0){
    task->requiresVar( Task::NewDW, lb->modelMass_srcLabel, m_gn );
  }

  computesRequires_CustomBCs(task, "update_press_CC", lb, ice_matls,
                             d_ice->d_BC_globalVars);

  task->computesVar( lb->press_CCLabel,        press_matl, oims );
  task->computesVar( lb->delP_DilatateLabel,   press_matl, oims );
  task->computesVar( lb->delP_MassXLabel,      press_matl, oims );
  task->computesVar( lb->term2Label,           press_matl, oims );
  task->computesVar( lb->sum_rho_CCLabel,      press_matl, oims );
  task->computesVar( lb->vol_fracX_FCLabel );
  task->computesVar( lb->vol_fracY_FCLabel );
  task->computesVar( lb->vol_fracZ_FCLabel );

  task->computesVar( VarLabel::find(abortTimeStep_name) );
  task->computesVar( VarLabel::find(recomputeTimeStep_name) );

  sched->addTask(task, patches, matls);
}

/* _____________________________________________________________________
Task:   SPICE::scheduleComputePressFC--
Purpose: Using the pressure compute the pressure at the cell faces
_____________________________________________________________________*/
//  Copied from ICE::scheduleComputePressFC and modified for SPICE.  Keep it diffable against the
//  ICE original so ICE fixes can be ported.
void SPICE::scheduleComputePressFC( SchedulerP          & sched,
                                  const PatchSet      * patches,
                                  const MaterialSubset* press_matl,
                                  const MaterialSet   * matls)
{
  printSchedule( patches, m_spice_tasks, " SPICE::scheduleComputePressFC" );

  Task* task = scinew Task("SPICE::computePressFC",
                     this, &SPICE::computePressFC);

  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.

  // KT face pressure: the own/nei stencil reads press_CC two cells out,
  // so use 2 ghost cells when it is on.  Must match the get() in
  // SPICE::computePressFC.
  int pressCCGhostCells = ( d_useCentralUpwind && d_ktFacePress ) ? 2 : 1;
  task->requiresVar( Task::NewDW,lb->press_CCLabel,   press_matl,oims, m_gac,pressCCGhostCells );
  task->requiresVar( Task::NewDW,lb->sum_rho_CCLabel, press_matl,oims, m_gac,1 );

  if( d_useCentralUpwind && d_ktFacePress ){
    // ktAlphaOwn is an ICE-material variable while press_CC/press_FC use
    // the pressure material, so it needs its own out-of-domain subset.
    task->requiresVar( Task::NewDW, Slb->ktAlphaOwn_FCXLabel, d_ktICE_matl, oims, m_gn,0 );
    task->requiresVar( Task::NewDW, Slb->ktAlphaOwn_FCYLabel, d_ktICE_matl, oims, m_gn,0 );
    task->requiresVar( Task::NewDW, Slb->ktAlphaOwn_FCZLabel, d_ktICE_matl, oims, m_gn,0 );
  }

  task->computesVar( lb->pressX_FCLabel, press_matl, oims);
  task->computesVar( lb->pressY_FCLabel, press_matl, oims);
  task->computesVar( lb->pressZ_FCLabel, press_matl, oims);

  sched->addTask( task, patches, matls);
}

/* _____________________________________________________________________
 Task:     scheduleAccumulateEnergySourceSinks--
 Purpose:  Accumulate all of the energy sources and sinks
           Copied from ICE::scheduleAccumulateEnergySourceSinks: consistent
           pressure-work formulation only.
_____________________________________________________________________*/
void SPICE::scheduleAccumulateEnergySourceSinks(SchedulerP          & sched,
                                              const PatchSet      * patches,
                                              const MaterialSubset* ice_mss,
                                              const MaterialSubset* /*mpm_mss*/,
                                              const MaterialSubset* press_mss,
                                              const MaterialSet   * matls)

{
  printSchedule( patches, m_spice_tasks, " SPICE::scheduleAccumulateEnergySourceSinks" );

  Task* t = scinew Task("SPICE::accumulateEnergySourceSinks",
                  this, &SPICE::accumulateEnergySourceSinks);

  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.

  t->requiresVar( Task::OldDW, lb->simulationTimeLabel );
  t->requiresVar( Task::OldDW, lb->delTLabel, getLevel(patches) );
  t->requiresVar( Task::NewDW, lb->press_CCLabel,     press_mss,oims, m_gac,1 );
  t->requiresVar( Task::NewDW, lb->sum_rho_CCLabel,   press_mss,oims, m_gac,1 );
  t->requiresVar( Task::NewDW, lb->pressX_FCLabel,    press_mss,oims, m_gac,1 );
  t->requiresVar( Task::NewDW, lb->pressY_FCLabel,    press_mss,oims, m_gac,1 );
  t->requiresVar( Task::NewDW, lb->pressZ_FCLabel,    press_mss,oims, m_gac,1 );
  t->requiresVar( Task::OldDW, lb->vel_CCLabel,       ice_mss, m_gac,1 );
  t->requiresVar( Task::NewDW, lb->f_theta_CCLabel,            m_gn );
  t->requiresVar( Task::OldDW, lb->temp_CCLabel,      ice_mss, m_gac,1 );
  t->requiresVar( Task::NewDW, lb->thermalCondLabel,  ice_mss, m_gac,1 );
  t->requiresVar( Task::NewDW, lb->rho_CCLabel,                m_gac,1 );
  t->requiresVar( Task::NewDW, lb->sp_vol_CCLabel,             m_gac,1 );
  t->requiresVar( Task::NewDW, lb->vol_frac_CCLabel,           m_gac,1 );

  t->computesVar( lb->int_eng_source_CCLabel );
  t->computesVar( lb->heatCond_src_CCLabel );
  sched->addTask(t, patches, matls);
}

/* _____________________________________________________________________
Task:     SPICE::schedulePimpleCopyModelSources--
Purpose:  PIMPLE outer iterations k>=1 freeze the chemistry/model sources
          at their k=0 value (reactingPimpleCentralFoam calls
          reaction->correct() once, before pimple.loop()).  Scheduled by
          scheduleFluidChain() in place of scheduleComputeModelSources()
          when inOuterLoop==true: computes, in the sub-graph's own NewDW,
          exactly the label set ICE::zeroModelSources() computes, by
          copying from Task::ParentNewDW (the main graph's k=0 result).
_____________________________________________________________________*/
void SPICE::schedulePimpleCopyModelSources(SchedulerP        & sched,
                                         const LevelP      & level,
                                         const MaterialSet * matls)
{
  if(d_ice->d_models.size() != 0){

    printSchedule( level, m_spice_tasks, " SPICE::schedulePimpleCopyModelSources" );

    Task* t = scinew Task("SPICE::pimpleCopyModelSources",this,
                          &SPICE::pimpleCopyModelSources);

    t->requiresVar( Task::ParentNewDW, lb->modelMass_srcLabel, m_gn,0 );
    t->requiresVar( Task::ParentNewDW, lb->modelMom_srcLabel,  m_gn,0 );
    t->requiresVar( Task::ParentNewDW, lb->modelEng_srcLabel,  m_gn,0 );
    t->requiresVar( Task::ParentNewDW, lb->modelVol_srcLabel,  m_gn,0 );

    t->computesVar( lb->modelMass_srcLabel );
    t->computesVar( lb->modelMom_srcLabel );
    t->computesVar( lb->modelEng_srcLabel );
    t->computesVar( lb->modelVol_srcLabel );

    //__________________________________
    // Model with transported variables.
    for(vector<ModelInterface*>::iterator m_iter  = d_ice->d_models.begin();
                                          m_iter != d_ice->d_models.end(); m_iter++){

      FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( *m_iter );

      if( fb_model && fb_model->d_transVars.size() ){
        for( auto& tvar : fb_model->d_transVars ){
          if(tvar->src){
            t->requiresVar( Task::ParentNewDW, tvar->src, tvar->matls, m_gn,0 );
            t->computesVar( tvar->src, tvar->matls );
          }
        }
      }
    }

    sched->addTask(t, level->eachPatch(), matls);
  }
}

/* _____________________________________________________________________
Task:     SPICE::scheduleCapturePimpleAdv0--
Purpose:  Snapshot the main graph's (k=0) advected conserved state into
          the "*_adv0" labels right after scheduleAdvectAndAdvanceInTime,
          so SPICE::pimpleBlend can read it from ParentNewDW without also
          modifying mass_adv/mom_adv/eng_adv/sp_vol_adv/tvar->var_adv (the
          SAME labels) in the sub-graph's own NewDW.
_____________________________________________________________________*/
void SPICE::scheduleCapturePimpleAdv0(SchedulerP        & sched,
                                    const LevelP      & level,
                                    const MaterialSet * matls)
{
  printSchedule( level, m_spice_tasks, " SPICE::scheduleCapturePimpleAdv0" );

  Task* t = scinew Task("SPICE::capturePimpleAdv0", this, &SPICE::capturePimpleAdv0);

  t->requiresVar( Task::NewDW, lb->mass_advLabel,   m_gn,0 );
  t->requiresVar( Task::NewDW, lb->mom_advLabel,    m_gn,0 );
  t->requiresVar( Task::NewDW, lb->eng_advLabel,    m_gn,0 );
  t->requiresVar( Task::NewDW, lb->sp_vol_advLabel, m_gn,0 );

  t->computesVar( Slb->mass_adv0Label );
  t->computesVar( Slb->mom_adv0Label );
  t->computesVar( Slb->eng_adv0Label );
  t->computesVar( Slb->sp_vol_adv0Label );

  int idx = 0;
  for( auto& model : d_ice->d_models ){
    FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( model );
    if( fb_model ){
      for( auto& tvar : fb_model->d_transVars ){
        const VarLabel* adv0 = pimpleTvarAdv0Label(idx);
        t->requiresVar( Task::NewDW, tvar->var_adv, tvar->matls, m_gn,0 );
        t->computesVar( adv0, tvar->matls );
      }
    }
  }

  sched->addTask(t, level->eachPatch(), matls);
}

/* _____________________________________________________________________
Task:     SPICE::schedulePimpleBlend--
Purpose:  PIMPLE outer iteration k (sub-graph only): combine the k=0
          advected state with this iteration's freshly advected state
          using the Crank-Nicolson theta blend
            X^k = X^n + (1-theta)*(X_adv0 - X^n) + theta*(X_sub - X^{k-1})
          (reactingPimpleCentralFoam, ddt CrankNicolson).  Total energy is
          blended (not int_eng directly), to avoid creating/destroying
          heat through the KE nonlinearity; see SPICE::pimpleBlend.
_____________________________________________________________________*/
void SPICE::schedulePimpleBlend(SchedulerP        & sched,
                              const LevelP      & level,
                              const MaterialSet * ice_matls)
{
  printSchedule( level, m_spice_tasks, " SPICE::schedulePimpleBlend" );

  const MaterialSubset* ice_mss = ice_matls->getUnion();

  Task* t = scinew Task("SPICE::pimpleBlend", this, &SPICE::pimpleBlend);

  // X^n : ParentOldDW primitives
  t->requiresVar( Task::ParentOldDW, lb->rho_CCLabel,     ice_mss, m_gn,0 );
  t->requiresVar( Task::ParentOldDW, lb->vel_CCLabel,     ice_mss, m_gn,0 );
  t->requiresVar( Task::ParentOldDW, lb->temp_CCLabel,    ice_mss, m_gn,0 );
  t->requiresVar( Task::ParentOldDW, lb->sp_vol_CCLabel,  ice_mss, m_gn,0 );

  // X_adv0 : ParentNewDW captured k=0 advected conserved state
  t->requiresVar( Task::ParentNewDW, Slb->mass_adv0Label,     ice_mss, m_gn,0 );
  t->requiresVar( Task::ParentNewDW, Slb->mom_adv0Label,      ice_mss, m_gn,0 );
  t->requiresVar( Task::ParentNewDW, Slb->eng_adv0Label,      ice_mss, m_gn,0 );
  t->requiresVar( Task::ParentNewDW, Slb->sp_vol_adv0Label,   ice_mss, m_gn,0 );
  t->requiresVar( Task::ParentNewDW, lb->specific_heatLabel, ice_mss, m_gn,0 );

  // X^{k-1} : sub OldDW primitives (the previous outer iterate)
  t->requiresVar( Task::OldDW, lb->rho_CCLabel,    ice_mss, m_gn,0 );
  t->requiresVar( Task::OldDW, lb->vel_CCLabel,    ice_mss, m_gn,0 );
  t->requiresVar( Task::OldDW, lb->temp_CCLabel,   ice_mss, m_gn,0 );
  t->requiresVar( Task::OldDW, lb->sp_vol_CCLabel, ice_mss, m_gn,0 );

  // cv for THIS sub-graph iteration, pairs with the sub OldDW's temp_CC
  // exactly as computeLagrangianValues pairs OldDW temp with NewDW cv.
  t->requiresVar( Task::NewDW, lb->specific_heatLabel, ice_mss, m_gn,0 );

  // X_sub : this iteration's raw advected result (computed earlier in
  // this same sub-graph by SPICE::advectAndAdvanceInTime) -- blended in place
  t->modifiesVar( lb->mass_advLabel );
  t->modifiesVar( lb->mom_advLabel );
  t->modifiesVar( lb->eng_advLabel );
  t->modifiesVar( lb->sp_vol_advLabel );

  int idx = 0;
  for( auto& model : d_ice->d_models ){
    FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( model );
    if( fb_model ){
      for( auto& tvar : fb_model->d_transVars ){
        const VarLabel* adv0 = pimpleTvarAdv0Label(idx);

        t->requiresVar( Task::ParentOldDW, tvar->var, tvar->matls, m_gn,0 );
        t->requiresVar( Task::ParentNewDW, adv0,      tvar->matls, m_gn,0 );
        t->requiresVar( Task::OldDW,       tvar->var, tvar->matls, m_gn,0 );
        t->modifiesVar( tvar->var_adv, tvar->matls );
      }
    }
  }

  sched->addTask(t, level->eachPatch(), ice_matls);
}

/* _____________________________________________________________________
Task:     SPICE::schedulePimpleResidual--
Purpose:  PIMPLE outer iteration k (sub-graph only, scheduled after
          conservedtoPrimitive_Vars): residual = max over interior cells
          of max(|rho_new-rho_old|/rho_new, |T_new-T_old|/T_new), used for
          the optional <outerTolerance> early exit.
_____________________________________________________________________*/
void SPICE::schedulePimpleResidual(SchedulerP        & sched,
                                 const LevelP      & level,
                                 const MaterialSet * ice_matls)
{
  printSchedule( level, m_spice_tasks, " SPICE::schedulePimpleResidual" );

  Task* t = scinew Task("SPICE::pimpleResidual", this, &SPICE::pimpleResidual);

  t->requiresVar( Task::OldDW, lb->rho_CCLabel,  m_gn,0 );
  t->requiresVar( Task::OldDW, lb->temp_CCLabel, m_gn,0 );
  t->requiresVar( Task::NewDW, lb->rho_CCLabel,  m_gn,0 );
  t->requiresVar( Task::NewDW, lb->temp_CCLabel, m_gn,0 );

  t->computesVar( Slb->pimpleResidualLabel );

  sched->addTask(t, level->eachPatch(), ice_matls);
}

/* _____________________________________________________________________
Task:     SPICE::schedulePimpleOuterLoop--
Purpose:  For nOuterCorrectors > 1, re-run the fluid chain in a
          sub-scheduler for outer iterations k=1..nOuterCorrectors-1,
          seeded from the main graph's own k=0 primitives, and copy the
          final iterate's primitives back.  Scheduled in
          scheduleTimeAdvance immediately after
          scheduleConservedtoPrimitive_Vars.
_____________________________________________________________________*/
void SPICE::schedulePimpleOuterLoop( SchedulerP & sched, const LevelP & level )
{
  if( d_pimpleOuter < 2 ){
    return;
  }

  // compiling the outer taskgraph means the inner one must be recompiled too
  d_recompilePimpleSched = true;

  printSchedule( level, m_spice_tasks, " SPICE::schedulePimpleOuterLoop" );

  const MaterialSet* ice_matls = m_materialManager->allMaterials( "ICE" );
  const MaterialSet* all_matls = m_materialManager->allMaterials();
  const MaterialSubset* ice_mss = ice_matls->getUnion();

  Task* t = scinew Task("SPICE::pimpleOuterLoop", this, &SPICE::pimpleOuterLoop, level);
  t->hasSubScheduler();

  t->requiresVar( Task::OldDW, lb->timeStepLabel );
  t->requiresVar( Task::OldDW, lb->simulationTimeLabel );
  t->requiresVar( Task::OldDW, lb->delTLabel, level.get_rep() );

  // u^n primitives, read inside the sub-graph via ParentOldDW
  t->requiresVar( Task::OldDW, lb->rho_CCLabel,    ice_mss, m_gn,0 );
  t->requiresVar( Task::OldDW, lb->vel_CCLabel,    ice_mss, m_gn,0 );
  t->requiresVar( Task::OldDW, lb->temp_CCLabel,   ice_mss, m_gn,0 );
  t->requiresVar( Task::OldDW, lb->sp_vol_CCLabel, ice_mss, m_gn,0 );

  // X_adv0 (k=0 captured advected state), read inside the sub-graph via
  // ParentNewDW
  t->requiresVar( Task::NewDW, Slb->mass_adv0Label,   ice_mss, m_gn,0 );
  t->requiresVar( Task::NewDW, Slb->mom_adv0Label,    ice_mss, m_gn,0 );
  t->requiresVar( Task::NewDW, Slb->eng_adv0Label,    ice_mss, m_gn,0 );
  t->requiresVar( Task::NewDW, Slb->sp_vol_adv0Label, ice_mss, m_gn,0 );

  // frozen model sources (copied every outer iteration by
  // SPICE::pimpleCopyModelSources, via ParentNewDW) -- require them here too
  // so the scheduler doesn't scrub them before the sub-graph reads them
  if( d_ice->d_models.size() != 0 ){
    t->requiresVar( Task::NewDW, lb->modelMass_srcLabel, m_gn,0 );
    t->requiresVar( Task::NewDW, lb->modelMom_srcLabel,  m_gn,0 );
    t->requiresVar( Task::NewDW, lb->modelEng_srcLabel,  m_gn,0 );
    t->requiresVar( Task::NewDW, lb->modelVol_srcLabel,  m_gn,0 );
  }

  // k=0 seed material, from ParentNewDW (the main graph's own primitives)
  t->requiresVar( Task::NewDW, lb->specific_heatLabel, ice_mss,      m_gn,0 );
  t->requiresVar( Task::NewDW, lb->speedSound_CCLabel, ice_mss,      m_gn,0 );
  t->requiresVar( Task::NewDW, lb->press_CCLabel,      d_ice->d_press_matl, m_gn,0 );
  t->requiresVar( Task::NewDW, lb->vol_frac_CCLabel,   ice_mss,      m_gn,0 );

  // rho/vel/temp/sp_vol/mach: already computed for k=0 by
  // conservedtoPrimitive_Vars just before this task; overwritten with the
  // outer loop's converged iterate.
  t->modifiesVar( lb->rho_CCLabel );
  t->modifiesVar( lb->vel_CCLabel );
  t->modifiesVar( lb->temp_CCLabel );
  t->modifiesVar( lb->sp_vol_CCLabel );
  t->modifiesVar( lb->machLabel );

  Task::MaterialDomainSpec oims = Task::OutOfDomain;
  t->modifiesVar( lb->press_CCLabel, d_ice->d_press_matl, oims );

  int idx = 0;
  for( auto& model : d_ice->d_models ){
    FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( model );
    if( fb_model ){
      for( auto& tvar : fb_model->d_transVars ){
        const VarLabel* adv0 = pimpleTvarAdv0Label(idx);

        t->requiresVar( Task::OldDW, tvar->var, tvar->matls, m_gn,0 );
        t->requiresVar( Task::NewDW, adv0,      tvar->matls, m_gn,0 );
        if(tvar->src){
          t->requiresVar( Task::NewDW, tvar->src, tvar->matls, m_gn,0 );
        }
        t->modifiesVar( tvar->var, tvar->matls );
      }
    }
  }

  // once per processor (like impICE's implicitPressureSolve): the sub-graph
  // execute() is collective and must see every local patch seeded at once
  const PatchSet * perProcPatches = m_loadBalancer->getPerProcessorPatchSet(level);
  sched->addTask(t, perProcPatches, all_matls);
}

/* _____________________________________________________________________
  Task:     scheduleAdvectAndAdvanceInTime--
  Purpose:  Advect the conserved quantities, mass, momentum, energy
            specific volume and any passive scalar variables.
_____________________________________________________________________*/
//  Copied from ICE::scheduleAdvectAndAdvanceInTime and modified for SPICE.  Keep it diffable against the
//  ICE original so ICE fixes can be ported.
void SPICE::scheduleAdvectAndAdvanceInTime(SchedulerP     & sched,
                                    const PatchSet      * patches,
                                    const MaterialSubset* ice_matlsub,
                                    const MaterialSet   * ice_matls)
{
  printSchedule( patches, m_spice_tasks, " SPICE::scheduleAdvectAndAdvanceInTime" );

  Task* t = scinew Task("SPICE::advectAndAdvanceInTime",this,
                        &SPICE::advectAndAdvanceInTime);


  t->requiresVar( Task::OldDW, lb->delTLabel,getLevel(patches));
  t->requiresVar( Task::NewDW, lb->uvel_FCMELabel,   m_gac,2 );
  t->requiresVar( Task::NewDW, lb->vvel_FCMELabel,   m_gac,2 );
  t->requiresVar( Task::NewDW, lb->wvel_FCMELabel,   m_gac,2 );
  t->requiresVar( Task::NewDW, lb->mom_L_ME_CCLabel, m_gac,2 );
  t->requiresVar( Task::NewDW, lb->mass_L_CCLabel,   m_gac,2 );
  t->requiresVar( Task::NewDW, lb->eng_L_ME_CCLabel, m_gac,2 );
  t->requiresVar( Task::NewDW, lb->sp_vol_L_CCLabel, m_gac,2 );

  if( d_useCentralUpwind ){
    // gac 2, like uvel_FCME: the last interior cell reads its high face
    // (c+1), which belongs to the neighbor patch.
    t->requiresVar( Task::NewDW, Slb->ktPhiOwn_FCXLabel, m_gac,2 );
    t->requiresVar( Task::NewDW, Slb->ktPhiOwn_FCYLabel, m_gac,2 );
    t->requiresVar( Task::NewDW, Slb->ktPhiOwn_FCZLabel, m_gac,2 );
    t->requiresVar( Task::NewDW, Slb->ktPhiNei_FCXLabel, m_gac,2 );
    t->requiresVar( Task::NewDW, Slb->ktPhiNei_FCYLabel, m_gac,2 );
    t->requiresVar( Task::NewDW, Slb->ktPhiNei_FCZLabel, m_gac,2 );
  }

  if(d_ice->d_doRefluxing){
    d_ice->computesRequires_AMR_Refluxing(t, ice_matls);
  }

  t->computesVar( lb->mass_advLabel );
  t->computesVar( lb->mom_advLabel );
  t->computesVar( lb->eng_advLabel );
  t->computesVar( lb->sp_vol_advLabel );
  //__________________________________
  // Model with transported variables.
  if(d_ice->d_models.size()){
    for(vector<ModelInterface*>::iterator m_iter  = d_ice->d_models.begin();
                                          m_iter != d_ice->d_models.end(); m_iter++){

      FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( *m_iter );

      if( fb_model && fb_model->d_transVars.size() ){

        vector<TransportedVariable*>::iterator t_iter;

        for(t_iter  = fb_model->d_transVars.begin();
            t_iter != fb_model->d_transVars.end(); t_iter++){
          TransportedVariable* tvar = *t_iter;

          t->requiresVar( Task::NewDW, tvar->var_Lagrangian, tvar->matls, m_gac, 2);
          t->computesVar( tvar->var_adv, tvar->matls);
        }
      }
    }
  }

  t->computesVar( VarLabel::find(abortTimeStep_name) );
  t->computesVar( VarLabel::find(recomputeTimeStep_name) );

  sched->addTask(t, patches, ice_matls);
}
/* _____________________________________________________________________
 Task:      SPICE::computeCentralWeights
 Purpose:   Per-face Kurganov-Tadmor weights (alphaOwn, aSf) and the
            blending function kappa.  PIMPLE_KT_method.pdf Section 4;
            math in Core/CentralUpwind.h.
 _____________________________________________________________________  */
void SPICE::computeCentralWeights(const ProcessorGroup  *,
                                const PatchSubset     * patches,
                                const MaterialSubset  * /*matls*/,
                                DataWarehouse         * old_dw,
                                DataWarehouse         * new_dw)
{
  using namespace Uintah::centralUpwind;

  for(int p = 0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch, m_spice_tasks, "SPICE::computeCentralWeights" );

    Vector dx = patch->dCell();
    const Level* level = getLevel(patches);
    IntVector lo, hi;
    level->findCellIndexRange(lo, hi);
    IntVector periodic = level->getPeriodicBoundaries();

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, level);

    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);    // X faces
    adj_offset[1] = IntVector(0, -1, 0);    // Y faces
    adj_offset[2] = IntVector(0,  0, -1);   // Z faces

    unsigned int numMatls = m_materialManager->getNumMatls( "ICE" );
    for(unsigned int m = 0; m < numMatls; m++) {
      ICEMaterial* ice_matl = (ICEMaterial*) m_materialManager->getMaterial( "ICE", m );
      int indx = ice_matl->getDWIndex();

      constCCVariable<double> rho_CC;
      constCCVariable<double> speedSound;
      constCCVariable<Vector> vel_CC;
      new_dw->get( rho_CC,     lb->rho_CCLabel,        indx,patch,m_gac,2 );
      new_dw->get( speedSound, lb->speedSound_CCLabel, indx,patch,m_gac,2 );
      old_dw->get( vel_CC,     lb->vel_CCLabel,        indx,patch,m_gac,2 );

      SFCXVariable<double> alphaOwnX, aSfX, kappaX;
      SFCYVariable<double> alphaOwnY, aSfY, kappaY;
      SFCZVariable<double> alphaOwnZ, aSfZ, kappaZ;

      new_dw->allocateAndPut( alphaOwnX, Slb->ktAlphaOwn_FCXLabel, indx,patch );
      new_dw->allocateAndPut( alphaOwnY, Slb->ktAlphaOwn_FCYLabel, indx,patch );
      new_dw->allocateAndPut( alphaOwnZ, Slb->ktAlphaOwn_FCZLabel, indx,patch );
      new_dw->allocateAndPut( aSfX,      Slb->ktaSf_FCXLabel,      indx,patch );
      new_dw->allocateAndPut( aSfY,      Slb->ktaSf_FCYLabel,      indx,patch );
      new_dw->allocateAndPut( aSfZ,      Slb->ktaSf_FCZLabel,      indx,patch );
      new_dw->allocateAndPut( kappaX,    Slb->ktKappa_FCXLabel,    indx,patch );
      new_dw->allocateAndPut( kappaY,    Slb->ktKappa_FCYLabel,    indx,patch );
      new_dw->allocateAndPut( kappaZ,    Slb->ktKappa_FCZLabel,    indx,patch );

      // initialize the whole allocated array so no garbage remains where
      // this patch's own face range (below) doesn't reach
      IntVector lowIndex(patch->getExtraSFCXLowIndex());
      alphaOwnX.initialize( 0.5, lowIndex, patch->getExtraSFCXHighIndex() );
      alphaOwnY.initialize( 0.5, lowIndex, patch->getExtraSFCYHighIndex() );
      alphaOwnZ.initialize( 0.5, lowIndex, patch->getExtraSFCZHighIndex() );
      aSfX.initialize( 0.0, lowIndex, patch->getExtraSFCXHighIndex() );
      aSfY.initialize( 0.0, lowIndex, patch->getExtraSFCYHighIndex() );
      aSfZ.initialize( 0.0, lowIndex, patch->getExtraSFCZHighIndex() );
      kappaX.initialize( 1.0, lowIndex, patch->getExtraSFCXHighIndex() );
      kappaY.initialize( 1.0, lowIndex, patch->getExtraSFCYHighIndex() );
      kappaZ.initialize( 1.0, lowIndex, patch->getExtraSFCZHighIndex() );

      //__________________________________
      //  X,Y,Z faces: identical to computeVelFace's face range/offset
      //  (patch->getSFC{X,Y,Z}Iterator(), adj_offset) so this and
      //  computeVel_FC agree on which cell owns which face.
      for( int dir = 0; dir < 3; dir++ ){
        CellIterator it = (dir==0) ? patch->getSFCXIterator() :
                          (dir==1) ? patch->getSFCYIterator() :
                                     patch->getSFCZIterator();
        for(; !it.done(); it++){
          IntVector N   = *it;                     // face index == "nei" cell
          IntVector P   = N + adj_offset[dir];      // "own" cell
          IntVector Pm1 = P + adj_offset[dir];
          IntVector Np1 = N - adj_offset[dir];

          bool firstOrderOwn, firstOrderNei;
          centralUpwind::ktStencilFlags(Pm1, Np1, dir, lo, hi, periodic,
                        firstOrderOwn, firstOrderNei);

          // When a side is first order, ownNei() ignores qm1/qp2, but the
          // argument is still read before the call.  Pass P/N instead so
          // the read stays inside the array.
          const IntVector & Pm1Safe = firstOrderOwn ? P : Pm1;
          const IntVector & Np1Safe = firstOrderNei ? N : Np1;

          double uOwn, uNei, cOwn, cNei;
          ownNei( vel_CC[Pm1Safe][dir], vel_CC[P][dir], vel_CC[N][dir], vel_CC[Np1Safe][dir],
                 d_ktLimiter, firstOrderOwn, firstOrderNei, uOwn, uNei );
          ownNei( speedSound[Pm1Safe], speedSound[P], speedSound[N], speedSound[Np1Safe],
                 d_ktLimiter, firstOrderOwn, firstOrderNei, cOwn, cNei );

          double alphaOwn, aSf, amax;
          weights( uOwn, uNei, cOwn, cNei, d_ktScheme, alphaOwn, aSf, amax );

          double alphaNei = 1.0 - alphaOwn;
          const double tiny = 1.0e-100;         // matches d_ice->d_SMALL_NUM
          double maF  = std::fabs( alphaOwn*uOwn + alphaNei*uNei ) /
                       std::max( alphaOwn*cOwn + alphaNei*cNei, tiny );
          double coAc = amax*delT/dx[dir];
          double kappaVal = kappa( maF, coAc, d_ktKappaMode );

          if( dir == 0 ){
            alphaOwnX[N] = alphaOwn;  aSfX[N] = aSf;  kappaX[N] = kappaVal;
          } else if( dir == 1 ){
            alphaOwnY[N] = alphaOwn;  aSfY[N] = aSf;  kappaY[N] = kappaVal;
          } else {
            alphaOwnZ[N] = alphaOwn;  aSfZ[N] = aSf;  kappaZ[N] = kappaVal;
          }
        }
      }
    } // matls loop
  }  // patch loop
}

/* _____________________________________________________________________
 Task:      SPICE::computeCentralFluxes
 Purpose:   Per-face KT mass fluxes (phiOwn, phiNei) and the KT face
            velocity ktUvol_FC.  PIMPLE_KT_method.pdf Section 5.
 _____________________________________________________________________  */
void SPICE::computeCentralFluxes(const ProcessorGroup  *,
                               const PatchSubset     * patches,
                               const MaterialSubset  * /*matls*/,
                               DataWarehouse         * /*old_dw*/,
                               DataWarehouse         * new_dw)
{
  using namespace Uintah::centralUpwind;

  for(int p = 0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch, m_spice_tasks, "SPICE::computeCentralFluxes" );

    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);
    adj_offset[1] = IntVector(0, -1, 0);
    adj_offset[2] = IntVector(0,  0, -1);

    unsigned int numMatls = m_materialManager->getNumMatls( "ICE" );
    for(unsigned int m = 0; m < numMatls; m++) {
      ICEMaterial* ice_matl = (ICEMaterial*) m_materialManager->getMaterial( "ICE", m );
      int indx = ice_matl->getDWIndex();

      constSFCXVariable<double> uvel_FC, alphaOwnX, aSfX, kappaX;
      constSFCYVariable<double> vvel_FC, alphaOwnY, aSfY, kappaY;
      constSFCZVariable<double> wvel_FC, alphaOwnZ, aSfZ, kappaZ;
      constCCVariable<double>   rho_CC;

      new_dw->get( uvel_FC,   lb->uvel_FCMELabel,      indx,patch,m_gn,0 );
      new_dw->get( vvel_FC,   lb->vvel_FCMELabel,      indx,patch,m_gn,0 );
      new_dw->get( wvel_FC,   lb->wvel_FCMELabel,      indx,patch,m_gn,0 );
      new_dw->get( alphaOwnX, Slb->ktAlphaOwn_FCXLabel,  indx,patch,m_gn,0 );
      new_dw->get( alphaOwnY, Slb->ktAlphaOwn_FCYLabel,  indx,patch,m_gn,0 );
      new_dw->get( alphaOwnZ, Slb->ktAlphaOwn_FCZLabel,  indx,patch,m_gn,0 );
      new_dw->get( aSfX,      Slb->ktaSf_FCXLabel,       indx,patch,m_gn,0 );
      new_dw->get( aSfY,      Slb->ktaSf_FCYLabel,       indx,patch,m_gn,0 );
      new_dw->get( aSfZ,      Slb->ktaSf_FCZLabel,       indx,patch,m_gn,0 );
      new_dw->get( kappaX,    Slb->ktKappa_FCXLabel,     indx,patch,m_gn,0 );
      new_dw->get( kappaY,    Slb->ktKappa_FCYLabel,     indx,patch,m_gn,0 );
      new_dw->get( kappaZ,    Slb->ktKappa_FCZLabel,     indx,patch,m_gn,0 );
      new_dw->get( rho_CC,    lb->rho_CCLabel,          indx,patch,m_gac,2 );

      SFCXVariable<double> phiOwnX, phiNeiX, uVolX;
      SFCYVariable<double> phiOwnY, phiNeiY, uVolY;
      SFCZVariable<double> phiOwnZ, phiNeiZ, uVolZ;

      new_dw->allocateAndPut( phiOwnX, Slb->ktPhiOwn_FCXLabel, indx,patch );
      new_dw->allocateAndPut( phiOwnY, Slb->ktPhiOwn_FCYLabel, indx,patch );
      new_dw->allocateAndPut( phiOwnZ, Slb->ktPhiOwn_FCZLabel, indx,patch );
      new_dw->allocateAndPut( phiNeiX, Slb->ktPhiNei_FCXLabel, indx,patch );
      new_dw->allocateAndPut( phiNeiY, Slb->ktPhiNei_FCYLabel, indx,patch );
      new_dw->allocateAndPut( phiNeiZ, Slb->ktPhiNei_FCZLabel, indx,patch );
      new_dw->allocateAndPut( uVolX,   Slb->ktUvol_FCXLabel,   indx,patch );
      new_dw->allocateAndPut( uVolY,   Slb->ktUvol_FCYLabel,   indx,patch );
      new_dw->allocateAndPut( uVolZ,   Slb->ktUvol_FCZLabel,   indx,patch );

      IntVector lowIndex(patch->getExtraSFCXLowIndex());
      phiOwnX.initialize( 0.0, lowIndex, patch->getExtraSFCXHighIndex() );
      phiOwnY.initialize( 0.0, lowIndex, patch->getExtraSFCYHighIndex() );
      phiOwnZ.initialize( 0.0, lowIndex, patch->getExtraSFCZHighIndex() );
      phiNeiX.initialize( 0.0, lowIndex, patch->getExtraSFCXHighIndex() );
      phiNeiY.initialize( 0.0, lowIndex, patch->getExtraSFCYHighIndex() );
      phiNeiZ.initialize( 0.0, lowIndex, patch->getExtraSFCZHighIndex() );
      uVolX.initialize( 0.0, lowIndex, patch->getExtraSFCXHighIndex() );
      uVolY.initialize( 0.0, lowIndex, patch->getExtraSFCYHighIndex() );
      uVolZ.initialize( 0.0, lowIndex, patch->getExtraSFCZHighIndex() );

      const Level* level = getLevel(patches);
      IntVector lo, hi;
      level->findCellIndexRange(lo, hi);
      IntVector periodic = level->getPeriodicBoundaries();
      const double tiny = 1.0e-100;             // matches d_ice->d_SMALL_NUM

      for( int dir = 0; dir < 3; dir++ ){
        CellIterator it = (dir==0) ? patch->getSFCXIterator() :
                          (dir==1) ? patch->getSFCYIterator() :
                                     patch->getSFCZIterator();
        for(; !it.done(); it++){
          IntVector N   = *it;
          IntVector P   = N + adj_offset[dir];
          IntVector Pm1 = P + adj_offset[dir];
          IntVector Np1 = N - adj_offset[dir];

          bool firstOrderOwn, firstOrderNei;
          centralUpwind::ktStencilFlags(Pm1, Np1, dir, lo, hi, periodic,
                        firstOrderOwn, firstOrderNei);

          // see the Pm1Safe/Np1Safe comment in SPICE::computeCentralWeights
          const IntVector & Pm1Safe = firstOrderOwn ? P : Pm1;
          const IntVector & Np1Safe = firstOrderNei ? N : Np1;

          double rhoOwn, rhoNei;
          ownNei( rho_CC[Pm1Safe], rho_CC[P], rho_CC[N], rho_CC[Np1Safe],
                 d_ktLimiter, firstOrderOwn, firstOrderNei, rhoOwn, rhoNei );

          double alphaOwn = (dir==0)? alphaOwnX[N] : (dir==1)? alphaOwnY[N] : alphaOwnZ[N];
          double aSf      = (dir==0)? aSfX[N]      : (dir==1)? aSfY[N]      : aSfZ[N];
          double kap      = (dir==0)? kappaX[N]     : (dir==1)? kappaY[N]     : kappaZ[N];
          double u_f       = (dir==0)? uvel_FC[N]    : (dir==1)? vvel_FC[N]    : wvel_FC[N];
          double alphaNei = 1.0 - alphaOwn;
          double aphivOwn = alphaOwn*u_f - aSf;
          double aphivNei = alphaNei*u_f + aSf;

          double rhoUp = upwindValue( u_f, rhoOwn, rhoNei );
          double rhoOwnEff = kap*rhoOwn + (1.0-kap)*rhoUp;
          double rhoNeiEff = kap*rhoNei + (1.0-kap)*rhoUp;

          double phiOwn = aphivOwn*rhoOwnEff;
          double phiNei = aphivNei*rhoNeiEff;

          phiOwn += (1.0-kap)*phiNei;     // updateKappa.H
          phiNei *= kap;

          double rhoF = kap*(alphaOwn*rhoOwn + alphaNei*rhoNei) + (1.0-kap)*rhoUp;
          double uVol = (phiOwn + phiNei)/std::max(rhoF, tiny);

          if( dir == 0 ){
            phiOwnX[N] = phiOwn;  phiNeiX[N] = phiNei;  uVolX[N] = uVol;
          } else if( dir == 1 ){
            phiOwnY[N] = phiOwn;  phiNeiY[N] = phiNei;  uVolY[N] = uVol;
          } else {
            phiOwnZ[N] = phiOwn;  phiNeiZ[N] = phiNei;  uVolZ[N] = uVol;
          }
        }
      }
    } // matls loop
  }  // patch loop
}

/* _____________________________________________________________________
 Function:  SPICE::computeVelFace
 Purpose:   compute the face centered velocities minus the exchange
            contribution.
_____________________________________________________________________*/
//  Copied from ICE::computeVelFace and modified for SPICE.  Keep it diffable against the
//  ICE original so ICE fixes can be ported.
template<class T, class AT> void SPICE::computeVelFace(int dir,
                                           CellIterator it,
                                           IntVector adj_offset,
                                           double dx,
                                           double delT,
                                           double gravity,
                                           constCCVariable<double>& rho_CC,
                                           constCCVariable<double>& sp_vol_CC,
                                           constCCVariable<Vector>& vel_CC,
                                           constCCVariable<double>& press_CC,
                                           T& vel_FC,
                                           T& grad_P_FC,
                                           bool include_acc,
                                           const KTFaceVelInfo<AT>* kt)
{
  double inv_dx = 1.0/dx;

  double one_or_zero=1.;
  if(!include_acc){
    one_or_zero=0.0;
  }

  for(;!it.done(); it++){
    IntVector R = *it;
    IntVector L = R + adj_offset;

    double rho_FC = rho_CC[L] + rho_CC[R];
#if SCI_ASSERTION_LEVEL >=2
    if (rho_FC <= 0.0) {
      cout << d_myworld->myRank() << " rho_fc <= 0: " << rho_FC << " with L= " << L << " ("
           << rho_CC[L] << ") R= " << R << " (" << rho_CC[R]<< ")\n";
    }
#endif
    ASSERT(rho_FC > 0.0);

    //__________________________________
    // interpolation to the face
    double term1;
    if( kt ){
     // KT face velocity (PDF Section 7): limited own/nei velocities
      IntVector Pm1 = L + adj_offset;
      IntVector Np1 = R - adj_offset;

      bool firstOrderOwn, firstOrderNei;
      centralUpwind::ktStencilFlags( Pm1, Np1, dir, kt->lo, kt->hi, kt->periodic,
                      firstOrderOwn, firstOrderNei );

      // see the Pm1Safe/Np1Safe comment in SPICE::computeCentralWeights
      const IntVector & Pm1Safe = firstOrderOwn ? L : Pm1;
      const IntVector & Np1Safe = firstOrderNei ? R : Np1;

      double uOwn, uNei;
      Uintah::centralUpwind::ownNei( vel_CC[Pm1Safe][dir], vel_CC[L][dir],
                                     vel_CC[R][dir], vel_CC[Np1Safe][dir],
                                     d_ktLimiter, firstOrderOwn, firstOrderNei,
                                     uOwn, uNei );

      double alphaOwn = (*kt->alphaOwn)[R];
      term1 = alphaOwn*uOwn + (1.0 - alphaOwn)*uNei;
    }
    else {
      term1 = (rho_CC[L] * vel_CC[L][dir] +
              rho_CC[R] * vel_CC[R][dir])/(rho_FC);
    }
    //__________________________________
    // pressure term
    double sp_vol_brack = 2.*(sp_vol_CC[L] * sp_vol_CC[R])/
                             (sp_vol_CC[L] + sp_vol_CC[R]);

    grad_P_FC[R] = (press_CC[R] - press_CC[L]) * inv_dx;
    double term2 = delT * sp_vol_brack * grad_P_FC[R];

    //__________________________________
    // gravity term
    double term3 =  delT * gravity;

    vel_FC[R] = term1 - one_or_zero*term2 + one_or_zero*term3;
  }
}

/*_____________________________________________________________________
 Task:     SPICE::computeVel_FC
 Purpose:
   This function calculates the face centered velocities on all faces
   in the domain.
 _____________________________________________________________________  */
//  Copied from ICE::computeVel_FC and modified for SPICE.  Keep it diffable against the
//  ICE original so ICE fixes can be ported.
void SPICE::computeVel_FC(const ProcessorGroup  *,
                        const PatchSubset     * patches,
                        const MaterialSubset  * /*matls*/,
                        DataWarehouse         * old_dw,
                        DataWarehouse         * new_dw)
{
  for(int p = 0; p<patches->size(); p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch, m_spice_tasks, "ICE::computeVel_FCVel" );

    Vector dx      = patch->dCell();
    Vector gravity = d_ice->getGravity();

    constCCVariable<double> press_CC;
    new_dw->get( press_CC,lb->press_equil_CCLabel, 0, patch,m_gac,1);

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    // KT (Kurganov-Tadmor) central-upwind face velocity needs the level's
    // cell index range/periodicity for the own/nei stencil validity test
    // (ktStencilFlags); cheap, so just compute it once per patch.
    const Level* level = getLevel(patches);
    IntVector ktLo, ktHi;
    level->findCellIndexRange(ktLo, ktHi);
    IntVector ktPeriodic = level->getPeriodicBoundaries();

    // Compute the face centered velocities
    unsigned int numMatls = m_materialManager->getNumMatls();
    for(unsigned int m = 0; m < numMatls; m++) {

      Material* matl = m_materialManager->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

      // KT face velocity is only implemented for ICE materials (see
      // problemSetup -- CentralUpwind requires a single ICE material).
      // MPM materials keep the original density-weighted average.
      bool useKT = d_useCentralUpwind && d_ktFaceVel && ice_matl;

      constCCVariable<double> rho_CC;
      constCCVariable<double> sp_vol_CC;
      constCCVariable<Vector> vel_CC;

      constSFCXVariable<double> ktAlphaOwnX;
      constSFCYVariable<double> ktAlphaOwnY;
      constSFCZVariable<double> ktAlphaOwnZ;

      if(ice_matl){
        new_dw->get( rho_CC, lb->rho_CCLabel, indx,patch,m_gac,1 );
        // widen to gac 2 only when the KT own/nei stencil (Pm1/Np1) needs
        // the extra layer; keep the baseline gac 1 otherwise (must match
        // the requiresVar ghost count in scheduleComputeVel_FC).
        int velCCGhostCells = useKT ? 2 : 1;
        old_dw->get( vel_CC, lb->vel_CCLabel, indx,patch,m_gac,velCCGhostCells );

        if( useKT ){
          new_dw->get( ktAlphaOwnX, Slb->ktAlphaOwn_FCXLabel, indx,patch,m_gn,0 );
          new_dw->get( ktAlphaOwnY, Slb->ktAlphaOwn_FCYLabel, indx,patch,m_gn,0 );
          new_dw->get( ktAlphaOwnZ, Slb->ktAlphaOwn_FCZLabel, indx,patch,m_gn,0 );
        }
      }
      else {
        new_dw->get( rho_CC, lb->rho_CCLabel, indx,patch,m_gac,1 );
        new_dw->get( vel_CC, lb->vel_CCLabel, indx,patch,m_gac,1 );
      }
      new_dw->get( sp_vol_CC, lb->sp_vol_CCLabel,indx,patch,m_gac,1 );

      SFCXVariable<double> uvel_FC;
      SFCYVariable<double> vvel_FC;
      SFCZVariable<double> wvel_FC;

      SFCXVariable<double> grad_P_XFC;
      SFCYVariable<double> grad_P_YFC;
      SFCZVariable<double> grad_P_ZFC;

      new_dw->allocateAndPut( uvel_FC, lb->uvel_FCLabel, indx,patch );
      new_dw->allocateAndPut( vvel_FC, lb->vvel_FCLabel, indx,patch );
      new_dw->allocateAndPut( wvel_FC, lb->wvel_FCLabel, indx,patch );
      // debugging variables
      new_dw->allocateAndPut( grad_P_XFC, lb->grad_P_XFCLabel, indx,patch );
      new_dw->allocateAndPut( grad_P_YFC, lb->grad_P_YFCLabel, indx,patch );
      new_dw->allocateAndPut( grad_P_ZFC, lb->grad_P_ZFCLabel, indx,patch );

      IntVector lowIndex(patch->getExtraSFCXLowIndex());
      uvel_FC.initialize( 0.0, lowIndex,patch->getExtraSFCXHighIndex() );
      vvel_FC.initialize( 0.0, lowIndex,patch->getExtraSFCYHighIndex() );
      wvel_FC.initialize( 0.0, lowIndex,patch->getExtraSFCZHighIndex() );

      grad_P_XFC.initialize(0.0);
      grad_P_YFC.initialize(0.0);
      grad_P_ZFC.initialize(0.0);

      vector<IntVector> adj_offset(3);
      adj_offset[0] = IntVector(-1, 0, 0);    // X faces
      adj_offset[1] = IntVector(0, -1, 0);    // Y faces
      adj_offset[2] = IntVector(0,  0, -1);   // Z faces

      CellIterator XFC_iterator = patch->getSFCXIterator();
      CellIterator YFC_iterator = patch->getSFCYIterator();
      CellIterator ZFC_iterator = patch->getSFCZIterator();

      bool include_acc = true;
      if(mpm_matl && d_ice->d_with_rigid_mpm){
        include_acc = false;
      }

      //__________________________________
      //  Compute vel_FC for each face
      KTFaceVelInfo<constSFCXVariable<double> > ktInfoX{ &ktAlphaOwnX, ktLo, ktHi, ktPeriodic };
      KTFaceVelInfo<constSFCYVariable<double> > ktInfoY{ &ktAlphaOwnY, ktLo, ktHi, ktPeriodic };
      KTFaceVelInfo<constSFCZVariable<double> > ktInfoZ{ &ktAlphaOwnZ, ktLo, ktHi, ktPeriodic };

      computeVelFace<SFCXVariable<double>, constSFCXVariable<double> >(0, XFC_iterator,
                                       adj_offset[0],dx[0],delT,gravity[0],
                                       rho_CC,sp_vol_CC,vel_CC,press_CC,
                                       uvel_FC, grad_P_XFC, include_acc,
                                       useKT ? &ktInfoX : nullptr);

      computeVelFace<SFCYVariable<double>, constSFCYVariable<double> >(1, YFC_iterator,
                                       adj_offset[1],dx[1],delT,gravity[1],
                                       rho_CC,sp_vol_CC,vel_CC,press_CC,
                                       vvel_FC, grad_P_YFC, include_acc,
                                       useKT ? &ktInfoY : nullptr);

      computeVelFace<SFCZVariable<double>, constSFCZVariable<double> >(2, ZFC_iterator,
                                       adj_offset[2],dx[2],delT,gravity[2],
                                       rho_CC,sp_vol_CC,vel_CC,press_CC,
                                       wvel_FC, grad_P_ZFC, include_acc,
                                       useKT ? &ktInfoZ : nullptr);

      //__________________________________
      // (*)vel_FC BC are updated in
      // ICE::addExchangeContributionToFCVel()

    } // matls loop
  }  // patch loop
}
/*_____________________________________________________________________
 Task:     SPICE::computeDelPressAndUpdatePressCC--
 Purpose:
   This function calculates the change in pressure explicitly.
 Note:  Units of delp_Dilatate and delP_MassX are [Pa]
 Reference:  Multimaterial Formalism eq. 1.5
 _____________________________________________________________________  */
//  Copied from ICE::computeDelPressAndUpdatePressCC and modified for SPICE.  Keep it diffable against the
//  ICE original so ICE fixes can be ported.
void SPICE::computeDelPressAndUpdatePressCC(const ProcessorGroup  *,
                                          const PatchSubset     * patches,
                                          const MaterialSubset  * /*matls*/,
                                          DataWarehouse         * old_dw,
                                          DataWarehouse         * new_dw)
{
  timeStep_vartype timeStep;
  old_dw->get(timeStep, lb->timeStepLabel);
  bool isNotInitialTimeStep = (timeStep > 0);

  for(int p=0;p<patches->size();p++){

    const Patch* patch = patches->get(p);
    printTask(patches, patch, m_spice_tasks, "SPICE::computeDelPressAndUpdatePressCC" );

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches) );

    Vector dx      = patch->dCell();
    double inv_vol = 1.0/(dx.x()*dx.y()*dx.z());

    unsigned int numMatls = m_materialManager->getNumMatls();

    Advector* advector = d_ice->d_advector->clone(new_dw,patch,isRegridTimeStep());

    CCVariable<double> q_advected;
    CCVariable<double> delP_Dilatate;
    CCVariable<double> delP_MassX;
    CCVariable<double> sum_rho_CC;
    CCVariable<double> press_CC;
    CCVariable<double> term1;
    CCVariable<double> term2;
    constCCVariable<double> sumKappa;
    constCCVariable<double> press_equil;

    vector<CCVariable<double> > placeHolder(0);
    vector<constCCVariable<double> > sp_vol_CC(numMatls);

    const IntVector gc(1,1,1);
    new_dw->get( sumKappa,                lb->sumKappaLabel,      0,patch,m_gn,0 );
    new_dw->get( press_equil,             lb->press_equil_CCLabel,0,patch,m_gn,0 );

    new_dw->allocateAndPut( press_CC,     lb->press_CCLabel,      0, patch );
    new_dw->allocateAndPut( delP_Dilatate,lb->delP_DilatateLabel, 0, patch );
    new_dw->allocateAndPut( delP_MassX,   lb->delP_MassXLabel,    0, patch );
    new_dw->allocateAndPut( term2,        lb->term2Label,         0, patch );
    new_dw->allocateAndPut( sum_rho_CC,   lb->sum_rho_CCLabel,    0, patch );

    new_dw->allocateTemporary( q_advected, patch );
    new_dw->allocateTemporary( term1,      patch );

    term1.initialize(0.);
    term2.initialize(0.);
    sum_rho_CC.initialize(0.0);
    delP_Dilatate.initialize(0.0);
    delP_MassX.initialize(0.0);

    for(unsigned int m = 0; m < numMatls; m++) {
      Material* matl = m_materialManager->getMaterial( m );
      int indx = matl->getDWIndex();

      constCCVariable<double> speedSound;
      constCCVariable<double> vol_frac;
      constCCVariable<double> rho_CC;
      constSFCXVariable<double> uvel_FC;
      constSFCYVariable<double> vvel_FC;
      constSFCZVariable<double> wvel_FC;

      if( d_useCentralUpwind && d_ktPressureFlux ){
        // KT acoustic-dissipation term: use the KT volumetric face
        // velocity U_KT = u_f + kappa*aSf*(rhoNei-rhoOwn)/rhoF (see
        // SPICE::computeCentralFluxes) in place of the raw uvel_FCME, so
        // the pEqn convective term picks up the central-upwind
        // dissipation (explicit form).
        new_dw->get( uvel_FC, Slb->ktUvol_FCXLabel, indx,patch,m_gac,2 );
        new_dw->get( vvel_FC, Slb->ktUvol_FCYLabel, indx,patch,m_gac,2 );
        new_dw->get( wvel_FC, Slb->ktUvol_FCZLabel, indx,patch,m_gac,2 );
      } else {
        new_dw->get( uvel_FC, lb->uvel_FCMELabel,  indx,patch,m_gac,2 );
        new_dw->get( vvel_FC, lb->vvel_FCMELabel,  indx,patch,m_gac,2 );
        new_dw->get( wvel_FC, lb->wvel_FCMELabel,  indx,patch,m_gac,2 );
      }
      new_dw->get( vol_frac,    lb->vol_frac_CCLabel,   indx,patch,m_gac,2 );
      new_dw->get( rho_CC,      lb->rho_CCLabel,        indx,patch,m_gn,0 );
      new_dw->get( sp_vol_CC[m],lb->sp_vol_CCLabel,     indx,patch,m_gn,0 );
      new_dw->get( speedSound,  lb->speedSound_CCLabel, indx,patch,m_gn,0 );

      SFCXVariable<double> vol_fracX_FC;
      SFCYVariable<double> vol_fracY_FC;
      SFCZVariable<double> vol_fracZ_FC;

      new_dw->allocateAndPut( vol_fracX_FC, lb->vol_fracX_FCLabel, indx,patch );
      new_dw->allocateAndPut( vol_fracY_FC, lb->vol_fracY_FCLabel, indx,patch );
      new_dw->allocateAndPut( vol_fracZ_FC, lb->vol_fracZ_FCLabel, indx,patch );


      // lowIndex is the same for all vel_FC
      IntVector lowIndex(patch->getExtraSFCXLowIndex());
      double nan= getNan();
      vol_fracX_FC.initialize( nan, lowIndex,patch->getExtraSFCXHighIndex() );
      vol_fracY_FC.initialize( nan, lowIndex,patch->getExtraSFCYHighIndex() );
      vol_fracZ_FC.initialize( nan, lowIndex,patch->getExtraSFCZHighIndex() );

      //__________________________________
      // Advection preprocessing
      // - divide vol_frac_cc/vol
      bool bulletProof_test=true;
      advectVarBasket* varBasket = scinew advectVarBasket();

      advector->inFluxOutFluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch,indx,
                                    bulletProof_test, new_dw, varBasket);
      //__________________________________
      //   advect vol_frac
      varBasket->doRefluxing = false;  // don't need to reflux here
      advector->advectQ(vol_frac, patch, q_advected, varBasket,
                        vol_fracX_FC, vol_fracY_FC,  vol_fracZ_FC, new_dw);

      delete varBasket;

      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        term2[c] -= q_advected[c];
      }

      //__________________________________
      //   term1 contribution from models
      if(d_ice->d_models.size() > 0){
        constCCVariable<double> modelMass_src, modelVol_src;
        new_dw->get(modelMass_src, lb->modelMass_srcLabel, indx, patch, m_gn,0);

        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
         IntVector c = *iter;
         term1[c] += modelMass_src[c] * (sp_vol_CC[m][c]* inv_vol);
        }
      }

      //__________________________________
      //  compute sum_rho_CC used by press_FC
      for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        sum_rho_CC[c] += rho_CC[c];
      }
    }  //matl loop
    delete advector;

    //__________________________________
    //  add delP to press_equil
    //  AMR:  hit the extra cells, BC aren't set an you need a valid pressure there
    // THIS COULD BE TROUBLE
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      press_CC[c] = press_equil[c];
    }

    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
      IntVector c = *iter;
      double inv_sumKappa = 1.0/sumKappa[c];
      delP_MassX[c]    =  term1[c] * inv_sumKappa;
      delP_Dilatate[c] = -term2[c] * inv_sumKappa;
      press_CC[c]      =  press_equil[c] + delP_MassX[c] + delP_Dilatate[c];
      press_CC[c]      = max(1.0e-12, press_CC[c]);  // CLAMP
//      delP_Dilatate[c] = press_CC[c] - delP_MassX[c] - press_equil[c];

    }

    //__________________________________
    //  set boundary conditions
    customBC_localVars* BC_localVars   = scinew customBC_localVars();

    preprocess_CustomBCs("update_press_CC",old_dw, new_dw, lb,  patch, 999,
                          d_ice->d_BC_globalVars, BC_localVars);

    setBC(press_CC, placeHolder, sp_vol_CC, d_ice->d_surroundingMatl_indx,
          "sp_vol", "Pressure", patch ,m_materialManager, 0, new_dw,
          d_ice->d_BC_globalVars, BC_localVars, isNotInitialTimeStep );
    delete_CustomBCs(d_ice->d_BC_globalVars, BC_localVars);

  }  // patch loop

}

/* _____________________________________________________________________
 Function:  SPICE::computePressFace--
 Purpose:   Driver function that is called by task computePressFC
  _____________________________________________________________________  */
//  Copied from ICE::computePressFace and modified for SPICE.  Keep it diffable against the
//  ICE original so ICE fixes can be ported.
template <class T, class AT> void SPICE::computePressFace(CellIterator iter,
                                              int dir,
                                              IntVector adj_offset,
                                              constCCVariable<double>& sum_rho,
                                              constCCVariable<double>& press_CC,
                                              T& press_FC,
                                              const KTFacePressInfo<AT>* kt)
{
  for(;!iter.done(); iter++){
    IntVector R = *iter;
    IntVector L = R + adj_offset;

    if( kt ){
      // KT face pressure (PDF Section 7): alphaOwn*pOwn + alphaNei*pNei,
      // as in reactingPimpleCentralFoam pEqn.H.  This is not a Riemann p*
      // (no impedance jump term); a dissipative p* can collapse a
      // detonation induction zone.  If induction-zone problems appear,
      // try <facePressure>false</facePressure> first.
      IntVector Pm1 = L + adj_offset;
      IntVector Np1 = R - adj_offset;

      bool firstOrderOwn, firstOrderNei;
      centralUpwind::ktStencilFlags( Pm1, Np1, dir, kt->lo, kt->hi, kt->periodic,
                      firstOrderOwn, firstOrderNei );

      // see the Pm1Safe/Np1Safe comment in SPICE::computeCentralWeights
      const IntVector & Pm1Safe = firstOrderOwn ? L : Pm1;
      const IntVector & Np1Safe = firstOrderNei ? R : Np1;

      double pOwn, pNei;
      Uintah::centralUpwind::ownNei( press_CC[Pm1Safe], press_CC[L],
                                     press_CC[R], press_CC[Np1Safe],
                                     d_ktLimiter, firstOrderOwn, firstOrderNei,
                                     pOwn, pNei );

      double alphaOwn = (*kt->alphaOwn)[R];
      press_FC[R] = alphaOwn*pOwn + (1.0 - alphaOwn)*pNei;
    }
    else {
      press_FC[R] = (press_CC[R] * sum_rho[L] + press_CC[L] * sum_rho[R])/
        (sum_rho[R] + sum_rho[L]);
    }
  }
}

/* _____________________________________________________________________
 Task:  SPICE::computePressFC--
 Purpose:
    This function calculates the face centered pressure on each of the
    cell faces for every cell in the computational domain and a single
    layer of ghost cells.
  _____________________________________________________________________  */
//  Copied from ICE::computePressFC and modified for SPICE.  Keep it diffable against the
//  ICE original so ICE fixes can be ported.
void SPICE::computePressFC(const ProcessorGroup *,
                         const PatchSubset    * patches,
                         const MaterialSubset * /*matls*/,
                         DataWarehouse        *,
                         DataWarehouse        * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, m_spice_tasks, "SPICE::computePressFC" );

    constCCVariable<double> press_CC;
    constCCVariable<double> sum_rho_CC;

    // KT (Kurganov-Tadmor) central-upwind face pressure: the own/nei
    // reconstruction stencil reads press_CC two cells out (Pm1/Np1), so
    // widen the ghost count only when the flag is on -- must match the
    // requiresVar ghost count in scheduleComputePressFC.
    bool useKT = d_useCentralUpwind && d_ktFacePress;
    int pressCCGhostCells = useKT ? 2 : 1;

    new_dw->get( press_CC,  lb->press_CCLabel,   0,patch,m_gac,pressCCGhostCells );
    new_dw->get( sum_rho_CC,lb->sum_rho_CCLabel, 0,patch,m_gac,1 );

    constSFCXVariable<double> ktAlphaOwnX;
    constSFCYVariable<double> ktAlphaOwnY;
    constSFCZVariable<double> ktAlphaOwnZ;
    IntVector ktLo, ktHi, ktPeriodic;

    if( useKT ){
      int iceIndx = m_materialManager->getMaterial( "ICE", 0 )->getDWIndex();
      new_dw->get( ktAlphaOwnX, Slb->ktAlphaOwn_FCXLabel, iceIndx,patch,m_gn,0 );
      new_dw->get( ktAlphaOwnY, Slb->ktAlphaOwn_FCYLabel, iceIndx,patch,m_gn,0 );
      new_dw->get( ktAlphaOwnZ, Slb->ktAlphaOwn_FCZLabel, iceIndx,patch,m_gn,0 );

      const Level* level = getLevel(patches);
      level->findCellIndexRange(ktLo, ktHi);
      ktPeriodic = level->getPeriodicBoundaries();
    }

    SFCXVariable<double> pressX_FC;
    SFCYVariable<double> pressY_FC;
    SFCZVariable<double> pressZ_FC;

    new_dw->allocateAndPut( pressX_FC, lb->pressX_FCLabel, 0, patch );
    new_dw->allocateAndPut( pressY_FC, lb->pressY_FCLabel, 0, patch );
    new_dw->allocateAndPut( pressZ_FC, lb->pressZ_FCLabel, 0, patch );

    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);    // X faces
    adj_offset[1] = IntVector(0, -1, 0);    // Y faces
    adj_offset[2] = IntVector(0,  0, -1);   // Z faces

    KTFacePressInfo<constSFCXVariable<double> > ktInfoX{ &ktAlphaOwnX, ktLo, ktHi, ktPeriodic };
    KTFacePressInfo<constSFCYVariable<double> > ktInfoY{ &ktAlphaOwnY, ktLo, ktHi, ktPeriodic };
    KTFacePressInfo<constSFCZVariable<double> > ktInfoZ{ &ktAlphaOwnZ, ktLo, ktHi, ktPeriodic };

    //__________________________________
    //  For each face compute the pressure
    computePressFace<SFCXVariable<double>, constSFCXVariable<double> >(patch->getSFCXIterator(), 0,
                                       adj_offset[0], sum_rho_CC, press_CC,
                                       pressX_FC, useKT ? &ktInfoX : nullptr);

    computePressFace<SFCYVariable<double>, constSFCYVariable<double> >(patch->getSFCYIterator(), 1,
                                       adj_offset[1], sum_rho_CC, press_CC,
                                       pressY_FC, useKT ? &ktInfoY : nullptr);

    computePressFace<SFCZVariable<double>, constSFCZVariable<double> >(patch->getSFCZIterator(), 2,
                                       adj_offset[2], sum_rho_CC, press_CC,
                                       pressZ_FC, useKT ? &ktInfoZ : nullptr);
  }  // patch loop
}

/* _____________________________________________________________________
 Task:      SPICE::accumulateEnergySourceSinks--
 Purpose:   This function accumulates all of the sources/sinks of energy
 Currently the kinetic energy isn't included.
 _____________________________________________________________________  */
//  Copied from ICE::accumulateEnergySourceSinks and modified for SPICE.  Keep it diffable against the
//  ICE original so ICE fixes can be ported.
void SPICE::accumulateEnergySourceSinks(const ProcessorGroup  *,
                                      const PatchSubset     * patches,
                                      const MaterialSubset  * /*matls*/,
                                      DataWarehouse         * old_dw,
                                      DataWarehouse         * new_dw)
{
  simTime_vartype simTimeVar;
  old_dw->get(simTimeVar, lb->simulationTimeLabel);
  double simTime = simTimeVar;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch, m_spice_tasks, "SPICE::accumulateEnergySourceSinks" );

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    Vector dx = patch->dCell();
    double vol=dx.x()*dx.y()*dx.z();

    double areaX = dx.y() * dx.z();
    double areaY = dx.x() * dx.z();
    double areaZ = dx.x() * dx.y();

    constCCVariable<double> sp_vol_CC;
    constCCVariable<double> vol_frac;
    constCCVariable<double> f_theta;
    constCCVariable<double> press_CC;
    constCCVariable<double> sum_rho_CC;
    constCCVariable<double> rho_CC;

    constSFCXVariable<double> pressX_FC;
    constSFCYVariable<double> pressY_FC;
    constSFCZVariable<double> pressZ_FC;

    new_dw->get( press_CC,   lb->press_CCLabel,   0,patch,m_gac,1 );
    new_dw->get( sum_rho_CC, lb->sum_rho_CCLabel, 0,patch,m_gac,1 );
    new_dw->get( pressX_FC,  lb->pressX_FCLabel,  0,patch,m_gac,1 );
    new_dw->get( pressY_FC,  lb->pressY_FCLabel,  0,patch,m_gac,1 );
    new_dw->get( pressZ_FC,  lb->pressZ_FCLabel,  0,patch,m_gac,1 );

    unsigned int numMatls = m_materialManager->getNumMatls();

    //______________________________________________________________________
    //  C O N S I S T E N T   P R E S S U R E   W O R K
    //
    //  Build the flow work from the same face pressure operator the momentum
    //  equation uses (press<X,Y,Z>_FC) instead of from delP_Dilatate, so the
    //  internal energy this task removes is exactly the kinetic energy
    //  accumulateMomentumSourceSinks adds.  Without that pairing IE + KE is not
    //  conserved discretely and the post shock temperature is underpredicted.
    //
    //      int_eng_source_m = f_theta_m * ( -delT*( TE_flux - KE_work ) )
    //
    //  KE_work is the exact discrete KE production of the pressure force summed
    //  over the ICE matls; it matches mom_source term for term.  TE_flux is the flux
    //  of press*(mixture volumetric flux W), cross weighted by sum_rho_CC
    //  exactly the way press_FC itself is (computePressFace).  That weighting is
    //  what makes the sum_rho*W*press cross term in
    //  upFlux[i+1/2] - W[i]*press_FC[i+1/2] cancel identically, leaving a clean
    //  sum_rho*press*(dW) residual -- the discrete -press*div(W).  upFlux at
    //  cell i for its right face is algebraically the same expression as at cell
    //  i+1 for its left face, so the flux telescopes and the formulation is
    //  globally conservative.
    //
    //  The mixture compresses as one at a single equilibrated pressure, so the
    //  work is computed once for the mixture and split by
    //  f_theta = vol_frac*kappa/sumKappa -- the same compressibility weighting
    //  the delP_Dilatate form carried, since
    //     TMV*vol_frac*kappa*press*delP_Dilatate
    //       == f_theta * ( press * total mixture volume change ).
    //  Keeping that weighting matters: in a stiff/compliant pair (water/air) the
    //  compliant material must absorb nearly all of the compression work.
    //
    //  Reduces to the single material form identically: vol_frac and f_theta are
    //  literally 1.0 there and sum_rho_CC is a one term sum.
    //
    //  KNOWN LIMITATION: a phase that is present only in trace amounts and is
    //  far more compressible than its host (air suspended in water at
    //  vol_frac ~ 1e-12) gets an f_theta share in which its own vol_frac
    //  cancels, so kappa_m/sumKappa (~1e4 there) multiplies whatever residual
    //  this operator leaves.  Because W is built from cell centred velocities
    //  while ICE enforces its incompressibility constraint on the FACE
    //  velocities, that residual does not vanish for incompressible flow and
    //  the trace phase picks up a spurious temperature spike.  See the
    //  waterAirOscillator case.
    CCVariable<Vector> W;              // sum over ICE matls of vol_frac * vel_CC
    CCVariable<double> KE_work;        // KE production by the pressure force
    CCVariable<double> netPressWork;   // mixture pressure work, before the f_theta split

    new_dw->allocateTemporary( W,            patch, m_gac, 1 );
    new_dw->allocateTemporary( KE_work,      patch );
    new_dw->allocateTemporary( netPressWork, patch );

    W.initialize( Vector(0.0,0.0,0.0) );
    KE_work.initialize( 0.0 );
    netPressWork.initialize( 0.0 );

    for(unsigned int m = 0; m < numMatls; m++) {
      Material* matl = m_materialManager->getMaterial( m );
      int indx = matl->getDWIndex();

      constCCVariable<double> volFrac;
      constCCVariable<double> rho;
      constCCVariable<Vector> vel_CC;

      new_dw->get( volFrac, lb->vol_frac_CCLabel, indx,patch,m_gac,1 );
      new_dw->get( rho,     lb->rho_CCLabel,      indx,patch,m_gac,1 );
      old_dw->get( vel_CC,  lb->vel_CCLabel,      indx,patch,m_gac,1 );

      //  a gac,1 fill needs the matching ghost count on the iterator, otherwise
      //  the layer at inter-patch faces is skipped and W is left zero there
      for(CellIterator iter = patch->getExtraCellIterator(1); !iter.done(); iter++){
        IntVector c = *iter;
        W[c] += volFrac[c] * vel_CC[c];
      }

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c     = *iter;
        IntVector right = c + IntVector(1,0,0);
        IntVector top   = c + IntVector(0,1,0);
        IntVector front = c + IntVector(0,0,1);

        //  the pressure part of mom_source, matching accumulateMomentumSourceSinks
        //  term for term.  Gravity and the viscous source are deliberately left
        //  out: they do real work on KE and must not be drained from IE.
        double dp_X = ( pressX_FC[right] - pressX_FC[c] ) * areaX * volFrac[c];
        double dp_Y = ( pressY_FC[top]   - pressY_FC[c] ) * areaY * volFrac[c];
        double dp_Z = ( pressZ_FC[front] - pressZ_FC[c] ) * areaZ * volFrac[c];

        //  The exact discrete KE change from the pressure force is
        //  mom_source . u^{n+1/2}, NOT mom_source . u^n, because
        //  mom_L = mass*u^n + mom_source.  Using u^n leaves a spurious
        //  +|mom_source|^2/(2 mass) of heating every step: always positive,
        //  growing as |grad p|^2, so it over-predicts the post-shock
        //  temperature at strong shocks (and vanishes as delT -> 0).
        double halfInvMass = 1.0/( 2.0 * rho[c] * vol );

        double uHalf = vel_CC[c].x() - dp_X * delT * halfInvMass;
        double vHalf = vel_CC[c].y() - dp_Y * delT * halfInvMass;
        double wHalf = vel_CC[c].z() - dp_Z * delT * halfInvMass;

        KE_work[c] += uHalf * dp_X + vHalf * dp_Y + wHalf * dp_Z;
      }
    }

    //__________________________________
    //  mixture pressure work:  -delT*( div(W press) - W.grad(press_FC) )
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      IntVector c      = *iter;
      IntVector right  = c + IntVector(1,0,0);
      IntVector top    = c + IntVector(0,1,0);
      IntVector front  = c + IntVector(0,0,1);
      IntVector left   = c - IntVector(1,0,0);
      IntVector bottom = c - IntVector(0,1,0);
      IntVector back   = c - IntVector(0,0,1);

      double upFlux_R = ( sum_rho_CC[right]*W[c].x()    *press_CC[c]
                         + sum_rho_CC[c]    *W[right].x()*press_CC[right] )
                       / ( sum_rho_CC[c] + sum_rho_CC[right] );
      double upFlux_L = ( sum_rho_CC[c]   *W[left].x()*press_CC[left]
                         + sum_rho_CC[left]*W[c].x()   *press_CC[c] )
                       / ( sum_rho_CC[left] + sum_rho_CC[c] );

      double upFlux_T = ( sum_rho_CC[top]*W[c].y()  *press_CC[c]
                         + sum_rho_CC[c]  *W[top].y()*press_CC[top] )
                       / ( sum_rho_CC[c] + sum_rho_CC[top] );
      double upFlux_B = ( sum_rho_CC[c]     *W[bottom].y()*press_CC[bottom]
                         + sum_rho_CC[bottom]*W[c].y()     *press_CC[c] )
                       / ( sum_rho_CC[bottom] + sum_rho_CC[c] );

      double upFlux_F  = ( sum_rho_CC[front]*W[c].z()    *press_CC[c]
                          + sum_rho_CC[c]    *W[front].z()*press_CC[front] )
                        / ( sum_rho_CC[c] + sum_rho_CC[front] );
      double upFlux_Bk = ( sum_rho_CC[c]   *W[back].z()*press_CC[back]
                          + sum_rho_CC[back]*W[c].z()   *press_CC[c] )
                        / ( sum_rho_CC[back] + sum_rho_CC[c] );

      double TE_flux =
          ( upFlux_R - upFlux_L  ) * areaX
        + ( upFlux_T - upFlux_B  ) * areaY
        + ( upFlux_F - upFlux_Bk ) * areaZ;

      netPressWork[c] = -delT * (TE_flux - KE_work[c]);
    }

    for(unsigned int m = 0; m < numMatls; m++) {
      Material* matl = m_materialManager->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);

      int indx = matl->getDWIndex();
      CCVariable<double> int_eng_source;
      CCVariable<double> heatCond_src;

      new_dw->get( sp_vol_CC,  lb->sp_vol_CCLabel,      indx,patch,m_gac,1 );
      new_dw->get( rho_CC,     lb->rho_CCLabel,         indx,patch,m_gac,1 );
      new_dw->get( vol_frac,   lb->vol_frac_CCLabel,    indx,patch,m_gac,1 );
      new_dw->get( f_theta,    lb->f_theta_CCLabel,     indx,patch,m_gn,0 );

      new_dw->allocateAndPut( int_eng_source,
                               lb->int_eng_source_CCLabel,indx,patch);
      new_dw->allocateAndPut( heatCond_src,
                               lb->heatCond_src_CCLabel,  indx,patch);
      int_eng_source.initialize(0.0);
      heatCond_src.initialize(0.0);

      //__________________________________
      //  Source due to conduction ICE only
      if(ice_matl){
        double thermalCond_test = ice_matl->getThermalConductivity();
        if(thermalCond_test != 0.0 ){
          constCCVariable<double> Temp_CC;
          constCCVariable<double> thermalCond;

          new_dw->get( thermalCond, lb->thermalCondLabel, indx,patch,m_gac,1 );
          old_dw->get( Temp_CC,     lb->temp_CCLabel,     indx,patch,m_gac,1 );

          bool use_vol_frac = true; // include vol_frac in diffusion calc.
          scalarDiffusionOperator(new_dw, patch, use_vol_frac, Temp_CC,
                                  vol_frac, heatCond_src, thermalCond, delT);
        }

        //__________________________________
        //   Flow work
        //  netPressWork is the mixture pressure work computed above; f_theta is
        //  this material's compressibility weighted share of it.
        if( ice_matl->getIncludeFlowWork() ){
          for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
            IntVector c = *iter;
            int_eng_source[c] += f_theta[c] * netPressWork[c] + heatCond_src[c];
          }
        }
      }

      //__________________________________
      //  User specified source/sink
      if (  d_ice->d_add_heat &&
            simTime >= d_ice->d_add_heat_t_start &&
            simTime <= d_ice->d_add_heat_t_final ) {

        for (int i = 0; i<(int) d_ice->d_add_heat_matls.size(); i++) {
          if((int)m == d_ice->d_add_heat_matls[i] ){
            for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
              IntVector c = *iter;
              if ( vol_frac[c] > 0.001) {
                int_eng_source[c] += d_ice->d_add_heat_coeff[i] * delT * rho_CC[c] * vol;
              }
            }  // iter loop
          }  // if right matl
        }
      }  // if add heat

    }  // matl loop
  }  // patch loop
}

/* _____________________________________________________________________
 Task:      advectAndAdvanceInTime--
 Purpose:
   This task calculates the The cell-centered, time n+1, mass, momentum
   internal energy, sp_vol
 _____________________________________________________________________  */
//  Copied from ICE::advectAndAdvanceInTime and modified for SPICE.  Keep it diffable against the
//  ICE original so ICE fixes can be ported.
void SPICE::advectAndAdvanceInTime(const ProcessorGroup * /*pg*/,
                                 const PatchSubset    * patches,
                                 const MaterialSubset * /*matls*/,
                                 DataWarehouse        * old_dw,
                                 DataWarehouse        * new_dw)
{
  const Level* level = getLevel(patches);

  // the advection calculations care about the position of the old dw subcycle
  double AMR_subCycleProgressVar = getSubCycleProgress(old_dw);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    ostringstream msg;
    msg << "SPICE::advectAndAdvanceInTime: progressVar " << AMR_subCycleProgressVar;
    printTask(patches, patch, m_spice_tasks, msg.str() );

    delt_vartype delT;
    old_dw->get( delT, lb->delTLabel, level );

    Advector* advector = d_ice->d_advector->clone(new_dw,patch,isRegridTimeStep());

    CCVariable<double>  q_advected;
    CCVariable<Vector>  qV_advected;
    new_dw->allocateTemporary( q_advected,   patch );
    new_dw->allocateTemporary( qV_advected,  patch );

    unsigned int numMatls = m_materialManager->getNumMatls( "ICE" );

    Vector dx  = patch->dCell();
    double vol = dx.x() * dx.y() * dx.z();

    for (unsigned int m = 0; m < numMatls; m++ ) {
      ICEMaterial* ice_matl = (ICEMaterial*) m_materialManager->getMaterial( "ICE",  m );
      int indx = ice_matl->getDWIndex();

      CCVariable<double> mass_adv;
      CCVariable<double> int_eng_adv;
      CCVariable<double> sp_vol_adv;
      CCVariable<Vector> mom_adv;

      constCCVariable<double> int_eng_L_ME;
      constCCVariable<double> mass_L,sp_vol_L;
      constCCVariable<Vector> mom_L_ME;
      constSFCXVariable<double > uvel_FC;
      constSFCYVariable<double > vvel_FC;
      constSFCZVariable<double > wvel_FC;

      new_dw->get( uvel_FC,     lb->uvel_FCMELabel,   indx,patch,m_gac,2 );
      new_dw->get( vvel_FC,     lb->vvel_FCMELabel,   indx,patch,m_gac,2 );
      new_dw->get( wvel_FC,     lb->wvel_FCMELabel,   indx,patch,m_gac,2 );

      new_dw->get( mass_L,      lb->mass_L_CCLabel,   indx,patch,m_gac,2 );
      new_dw->get( mom_L_ME,    lb->mom_L_ME_CCLabel, indx,patch,m_gac,2 );
      new_dw->get( sp_vol_L,    lb->sp_vol_L_CCLabel, indx,patch,m_gac,2 );
      new_dw->get( int_eng_L_ME,lb->eng_L_ME_CCLabel, indx,patch,m_gac,2 );

      constSFCXVariable<double> ktPhiOwnX, ktPhiNeiX;
      constSFCYVariable<double> ktPhiOwnY, ktPhiNeiY;
      constSFCZVariable<double> ktPhiOwnZ, ktPhiNeiZ;
      if( d_useCentralUpwind ){
        new_dw->get( ktPhiOwnX, Slb->ktPhiOwn_FCXLabel, indx,patch,m_gac,2 );
        new_dw->get( ktPhiOwnY, Slb->ktPhiOwn_FCYLabel, indx,patch,m_gac,2 );
        new_dw->get( ktPhiOwnZ, Slb->ktPhiOwn_FCZLabel, indx,patch,m_gac,2 );
        new_dw->get( ktPhiNeiX, Slb->ktPhiNei_FCXLabel, indx,patch,m_gac,2 );
        new_dw->get( ktPhiNeiY, Slb->ktPhiNei_FCYLabel, indx,patch,m_gac,2 );
        new_dw->get( ktPhiNeiZ, Slb->ktPhiNei_FCZLabel, indx,patch,m_gac,2 );
      }

      new_dw->allocateAndPut( mass_adv,    lb->mass_advLabel,   indx,patch );
      new_dw->allocateAndPut( mom_adv,     lb->mom_advLabel,    indx,patch );
      new_dw->allocateAndPut( int_eng_adv, lb->eng_advLabel,    indx,patch );
      new_dw->allocateAndPut( sp_vol_adv,  lb->sp_vol_advLabel, indx,patch );

      mass_adv.initialize(0.0);
      mom_adv.initialize(Vector(0.0,0.0,0.0));
      int_eng_adv.initialize(0.0);
      sp_vol_adv.initialize(0.0);
      q_advected.initialize(0.0);
      qV_advected.initialize(Vector(0.0,0.0,0.0));

      //__________________________________
      // common variables that get passed into the advection operators
      advectVarBasket* varBasket = scinew advectVarBasket();
      varBasket->new_dw = new_dw;
      varBasket->old_dw = old_dw;
      varBasket->indx   = indx;
      varBasket->patch  = patch;
      varBasket->level  = level;
      varBasket->lb     = lb;
      varBasket->doRefluxing = d_ice->d_doRefluxing;
      varBasket->useCompatibleFluxes = d_ice->d_useCompatibleFluxes;
      varBasket->AMR_subCycleProgressVar = AMR_subCycleProgressVar;

      //  KT fluxes: set before any advect call so mass/mom/int_eng/
      //  sp_vol/transported-var advection all pick up the central-upwind
      //  fluxes (CentralUpwindAdvector::advectMass/advectQ throw without
      //  them).  The ICE original carried them in advectVarBasket.
      if( d_useCentralUpwind ){
        CentralUpwindAdvector* ktAdvector = dynamic_cast<CentralUpwindAdvector*>( advector );
        if( !ktAdvector ){
          throw InternalError( "SPICE::advectAndAdvanceInTime: CentralUpwind is on but "
                               "the advector is not a CentralUpwindAdvector", __FILE__, __LINE__ );
        }

        const SFCXVariable<double>& phiOwnXRef = ktPhiOwnX;
        const SFCYVariable<double>& phiOwnYRef = ktPhiOwnY;
        const SFCZVariable<double>& phiOwnZRef = ktPhiOwnZ;
        const SFCXVariable<double>& phiNeiXRef = ktPhiNeiX;
        const SFCYVariable<double>& phiNeiYRef = ktPhiNeiY;
        const SFCZVariable<double>& phiNeiZRef = ktPhiNeiZ;

        ktAdvector->setCentralFluxes( &phiOwnXRef, &phiOwnYRef, &phiOwnZRef,
                                      &phiNeiXRef, &phiNeiYRef, &phiNeiZRef );
      }

      //__________________________________
      //   Advection preprocessing
      bool bulletProof_test=true;
      advector->inFluxOutFluxVolume(uvel_FC, vvel_FC, wvel_FC, delT, patch,indx,
                                    bulletProof_test, new_dw, varBasket);

      //__________________________________
      // mass
      advector->advectMass(mass_L, q_advected,  varBasket);

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        mass_adv[c]  = (mass_L[c] + q_advected[c]);
      }

      //__________________________________
      // momentum
      varBasket->is_Q_massSpecific   = true;
      varBasket->desc = "mom";
      advector->advectQ(mom_L_ME,mass_L,qV_advected, varBasket);

      for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
        IntVector c = *iter;
        mom_adv[c] = (mom_L_ME[c] + qV_advected[c]) ;
      }

      //__________________________________
      // internal energy
      //  Advect total energy, E = int_eng + KE, and recover the internal
      //  energy from the advected momentum:
      //      int_eng_adv = E_adv - |mom_adv|^2/(2 mass_adv)
      //  Advective KE dissipation -> heat (DeBar-style fix): upwind advection
      //  of momentum is diffusive, so the KE of the advected momentum is less
      //  than the KE the same operator transports conservatively.  Physically
      //  that deficit is the irreversible shock heating (entropy production);
      //  without it the post-shock temperature is underpredicted, and the
      //  error grows with shock strength.  Advecting E conservatively hands the
      //  deficit to int_eng, so IE + KE is conserved jointly across the
      //  advection step.  Identical to advecting int_eng and KE separately and
      //  adding the KE deficit, up to the limiter of a limited (second order,
      //  central upwind) reconstruction, which now acts on E; one advected
      //  quantity instead of two.  AMR refluxing corrects E through the
      //  int_eng flux labels.  Companion to the consistent pressure-work
      //  formulation in accumulateEnergySourceSinks.
      {
        CCVariable<double> E_L;
        new_dw->allocateTemporary( E_L, patch, m_gac, 2 );

        //  a material can be all but absent from a multi-material cell, so floor
        //  the mass the same way computeLagrangianValues does
        double minMass = ice_matl->getTinyRho() * vol;

        int NGC = 2;  // number of ghostCells
        for(CellIterator iter = patch->getExtraCellIterator(NGC); !iter.done(); iter++){
          IntVector c = *iter;
          E_L[c] = int_eng_L_ME[c] + 0.5 * mom_L_ME[c].length2()/std::max( mass_L[c], minMass );
        }

        varBasket->is_Q_massSpecific = true;
        varBasket->desc = "int_eng";
        advector->advectQ( E_L, mass_L, q_advected, varBasket );

        for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
          IntVector c = *iter;
          double KE_fromMom = 0.5 * mom_adv[c].length2()/std::max( mass_adv[c], minMass );
          int_eng_adv[c] = E_L[c] + q_advected[c] - KE_fromMom;
        }
      }

      //__________________________________
      // sp_vol[m] * mass
      varBasket->is_Q_massSpecific = true;
      varBasket->desc = "sp_vol";
      advector->advectQ(sp_vol_L,mass_L, q_advected, varBasket);

      for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
        IntVector c = *iter;
        sp_vol_adv[c] = (sp_vol_L[c] + q_advected[c]) ;
      }


      //__________________________________
      // Model with transported variables.
      if(d_ice->d_models.size()){
        for(vector<ModelInterface*>::iterator m_iter  = d_ice->d_models.begin();
                                              m_iter != d_ice->d_models.end(); m_iter++){
          FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( *m_iter );

          if( fb_model && fb_model->d_transVars.size() ){
            vector<TransportedVariable*>::iterator t_iter;

            for(t_iter  = fb_model->d_transVars.begin();
                t_iter != fb_model->d_transVars.end(); t_iter++){

              TransportedVariable* tvar = *t_iter;

              if(tvar->matls->contains(indx)){
                string Labelname = tvar->var->getName();

                CCVariable<double>      q_adv;
                constCCVariable<double> q_L_CC;

                new_dw->allocateAndPut(q_adv, tvar->var_adv,        indx, patch);
                new_dw->get(           q_L_CC,tvar->var_Lagrangian, indx, patch, m_gac, 2);

                q_adv.initialize(d_ice->d_EVIL_NUM);

                varBasket->desc = Labelname;
                varBasket->is_Q_massSpecific = true;

                advector->advectQ( q_L_CC, mass_L, q_advected, varBasket );

                for(CellIterator iter = patch->getCellIterator(); !iter.done();  iter++){
                  IntVector c = *iter;
                  q_adv[c] = (q_L_CC[c] + q_advected[c]) ;
                }
              }
            }
          }
        }
      }

      delete varBasket;
    }  // ice_matls loop
    delete advector;
  }  // patch loop
}
/* _____________________________________________________________________
 Task:      SPICE::pimpleCopyModelSources
 Purpose:   PIMPLE sub-graph: copy the (frozen) model sources computed by
            the main graph's ICE::zeroModelSources + the models'
            scheduleComputeModelSources out of Task::ParentNewDW, exactly
            mirroring the label set ICE::zeroModelSources computes.
 _____________________________________________________________________  */
void SPICE::pimpleCopyModelSources(const ProcessorGroup  *,
                                 const PatchSubset     * patches,
                                 const MaterialSubset  * matls,
                                 DataWarehouse         * /*old_dw*/,
                                 DataWarehouse         * new_dw)
{
  DataWarehouse* pNewDW = new_dw->getOtherDataWarehouse(Task::ParentNewDW);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, m_spice_tasks, "SPICE::pimpleCopyModelSources" );

    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);

      constCCVariable<double> mass_src, energy_src, vol_src;
      constCCVariable<Vector> mom_src;
      pNewDW->get( mass_src,   lb->modelMass_srcLabel, matl, patch, m_gn,0 );
      pNewDW->get( energy_src, lb->modelEng_srcLabel,  matl, patch, m_gn,0 );
      pNewDW->get( mom_src,    lb->modelMom_srcLabel,  matl, patch, m_gn,0 );
      pNewDW->get( vol_src,    lb->modelVol_srcLabel,  matl, patch, m_gn,0 );

      CCVariable<double> mass_src_new, energy_src_new, vol_src_new;
      CCVariable<Vector> mom_src_new;
      new_dw->allocateAndPut( mass_src_new,   lb->modelMass_srcLabel, matl, patch );
      new_dw->allocateAndPut( energy_src_new, lb->modelEng_srcLabel,  matl, patch );
      new_dw->allocateAndPut( mom_src_new,    lb->modelMom_srcLabel,  matl, patch );
      new_dw->allocateAndPut( vol_src_new,    lb->modelVol_srcLabel,  matl, patch );

      mass_src_new.copyData(mass_src);
      energy_src_new.copyData(energy_src);
      mom_src_new.copyData(mom_src);
      vol_src_new.copyData(vol_src);
    }

    //__________________________________
    // Model with transported variables.
    if(d_ice->d_models.size()){
      for( auto& model : d_ice->d_models ){
        FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( model );

        if( fb_model && fb_model->d_transVars.size() ){
          for( auto& tvar : fb_model->d_transVars ){

            if(!tvar->src){
              continue;
            }

            for(int m=0;m<tvar->matls->size();m++){
              int matl = tvar->matls->get(m);

              constCCVariable<double> src_old;
              pNewDW->get( src_old, tvar->src, matl, patch, m_gn,0 );

              CCVariable<double> src_new;
              new_dw->allocateAndPut( src_new, tvar->src, matl, patch );
              src_new.copyData(src_old);
            }
          }
        }
      }
    }
  }  // patch loop
}

/* _____________________________________________________________________
 Task:      SPICE::capturePimpleAdv0
 Purpose:   Snapshot the main graph's (k=0) mass_adv/mom_adv/eng_adv/
            sp_vol_adv/tvar->var_adv into the "*_adv0" labels.
 _____________________________________________________________________  */
void SPICE::capturePimpleAdv0(const ProcessorGroup  *,
                            const PatchSubset     * patches,
                            const MaterialSubset  * matls,
                            DataWarehouse         * /*old_dw*/,
                            DataWarehouse         * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, m_spice_tasks, "SPICE::capturePimpleAdv0" );

    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);

      constCCVariable<double> mass_adv, eng_adv, sp_vol_adv;
      constCCVariable<Vector> mom_adv;
      new_dw->get( mass_adv,   lb->mass_advLabel,   matl, patch, m_gn,0 );
      new_dw->get( mom_adv,    lb->mom_advLabel,    matl, patch, m_gn,0 );
      new_dw->get( eng_adv,    lb->eng_advLabel,    matl, patch, m_gn,0 );
      new_dw->get( sp_vol_adv, lb->sp_vol_advLabel, matl, patch, m_gn,0 );

      CCVariable<double> mass_adv0, eng_adv0, sp_vol_adv0;
      CCVariable<Vector> mom_adv0;
      new_dw->allocateAndPut( mass_adv0,   Slb->mass_adv0Label,   matl, patch );
      new_dw->allocateAndPut( mom_adv0,    Slb->mom_adv0Label,    matl, patch );
      new_dw->allocateAndPut( eng_adv0,    Slb->eng_adv0Label,    matl, patch );
      new_dw->allocateAndPut( sp_vol_adv0, Slb->sp_vol_adv0Label, matl, patch );

      mass_adv0.copyData(mass_adv);
      mom_adv0.copyData(mom_adv);
      eng_adv0.copyData(eng_adv);
      sp_vol_adv0.copyData(sp_vol_adv);
    }

    int idx = 0;
    for( auto& model : d_ice->d_models ){
      FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( model );
      if( fb_model ){
        for( auto& tvar : fb_model->d_transVars ){
          const VarLabel* adv0 = pimpleTvarAdv0Label(idx);

          for(int m=0;m<tvar->matls->size();m++){
            int matl = tvar->matls->get(m);

            constCCVariable<double> var_adv;
            new_dw->get( var_adv, tvar->var_adv, matl, patch, m_gn,0 );

            CCVariable<double> var_adv0;
            new_dw->allocateAndPut( var_adv0, adv0, matl, patch );
            var_adv0.copyData(var_adv);
          }
        }
      }
    }
  }  // patch loop
}

/* _____________________________________________________________________
 Task:      SPICE::pimpleBlend
 Purpose:   PIMPLE sub-graph, outer iteration k: Crank-Nicolson theta
            blend of the conserved state,
              X^k = X^n + (1-theta)*(X_adv0 - X^n) + theta*(X_sub - X^{k-1})
            with energy blended as TOTAL energy (int_eng + kinetic) so the
            blend doesn't create/destroy heat through the KE nonlinearity
            (reactingPimpleCentralFoam, ddt CrankNicolson correspondence:
            theta=0.5, nOuter=2 => Heun/SSP-RK2; theta=1 => Euler-implicit
            Picard).
 _____________________________________________________________________  */
void SPICE::pimpleBlend(const ProcessorGroup  *,
                      const PatchSubset     * patches,
                      const MaterialSubset  * matls,
                      DataWarehouse         * old_dw,
                      DataWarehouse         * new_dw)
{
  DataWarehouse* pOldDW = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  DataWarehouse* pNewDW = new_dw->getOtherDataWarehouse(Task::ParentNewDW);

  const double th = d_pimpleTheta;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, m_spice_tasks, "SPICE::pimpleBlend" );

    Vector dx  = patch->dCell();
    double vol = dx.x()*dx.y()*dx.z();

    for(int m=0;m<matls->size();m++){
      int indx = matls->get(m);

      constCCVariable<double> rho_n, temp_n, sp_vol_n;
      constCCVariable<Vector> vel_n;
      pOldDW->get( rho_n,    lb->rho_CCLabel,    indx, patch, m_gn,0 );
      pOldDW->get( vel_n,    lb->vel_CCLabel,    indx, patch, m_gn,0 );
      pOldDW->get( temp_n,   lb->temp_CCLabel,   indx, patch, m_gn,0 );
      pOldDW->get( sp_vol_n, lb->sp_vol_CCLabel, indx, patch, m_gn,0 );

      constCCVariable<double> mass_adv0, eng_adv0, sp_vol_adv0, cv_pNew;
      constCCVariable<Vector> mom_adv0;
      pNewDW->get( mass_adv0,   Slb->mass_adv0Label,     indx, patch, m_gn,0 );
      pNewDW->get( mom_adv0,    Slb->mom_adv0Label,      indx, patch, m_gn,0 );
      pNewDW->get( eng_adv0,    Slb->eng_adv0Label,      indx, patch, m_gn,0 );
      pNewDW->get( sp_vol_adv0, Slb->sp_vol_adv0Label,   indx, patch, m_gn,0 );
      pNewDW->get( cv_pNew,     lb->specific_heatLabel, indx, patch, m_gn,0 );

      constCCVariable<double> rho_km1, temp_km1, sp_vol_km1, cv_new;
      constCCVariable<Vector> vel_km1;
      old_dw->get( rho_km1,    lb->rho_CCLabel,    indx, patch, m_gn,0 );
      old_dw->get( vel_km1,    lb->vel_CCLabel,    indx, patch, m_gn,0 );
      old_dw->get( temp_km1,   lb->temp_CCLabel,   indx, patch, m_gn,0 );
      old_dw->get( sp_vol_km1, lb->sp_vol_CCLabel, indx, patch, m_gn,0 );
      new_dw->get( cv_new,     lb->specific_heatLabel, indx, patch, m_gn,0 );

      CCVariable<double> mass_sub, eng_sub, sp_vol_sub;
      CCVariable<Vector> mom_sub;
      new_dw->getModifiable( mass_sub,   lb->mass_advLabel,   indx, patch );
      new_dw->getModifiable( mom_sub,    lb->mom_advLabel,    indx, patch );
      new_dw->getModifiable( eng_sub,    lb->eng_advLabel,    indx, patch );
      new_dw->getModifiable( sp_vol_sub, lb->sp_vol_advLabel, indx, patch );

      //__________________________________
      //  Does a model own this material's caloric EOS?  If so the
      //  internal-energy carrier is the true sensible energy e_s(T,Y), not
      //  cv*T (mirrors ICE::computeLagrangianValues's eosOwner lookup).
      FluidsBasedModel* eosOwner = nullptr;
      for( auto& model : d_ice->d_models ){
        FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( model );
        if( fb_model && fb_model->ownsCaloricEOS( indx ) ){
          eosOwner = fb_model;
        }
      }

      CCVariable<double> es_n, es_km1;
      if( eosOwner ){
        new_dw->allocateTemporary(es_n,   patch);
        new_dw->allocateTemporary(es_km1, patch);
        // Y^n lives in ParentOldDW, Y^{k-1} lives in the sub old_dw -- both
        // as tvar->var (mass-specific primitives), i.e. YForm::OldPrimitive.
        eosOwner->computeSensibleEnergy( es_n, temp_n,
                              patch->getCellIterator(), patch, pOldDW,
                              FluidsBasedModel::YForm::OldPrimitive, indx );
        eosOwner->computeSensibleEnergy( es_km1, temp_km1,
                              patch->getCellIterator(), patch, old_dw,
                              FluidsBasedModel::YForm::OldPrimitive, indx );
      }

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;

        double mass_n   = rho_n[c]   * vol;
        double mass_km1 = rho_km1[c] * vol;
        double mass_a0  = mass_adv0[c];
        double mass_s   = mass_sub[c];

        Vector mom_n   = vel_n[c]   * mass_n;
        Vector mom_km1 = vel_km1[c] * mass_km1;
        Vector mom_a0  = mom_adv0[c];
        Vector mom_s   = mom_sub[c];

        double spv_n   = sp_vol_n[c]   * mass_n;
        double spv_km1 = sp_vol_km1[c] * mass_km1;
        double spv_a0  = sp_vol_adv0[c];
        double spv_s   = sp_vol_sub[c];

        double ie_n   = eosOwner ? mass_n   * es_n[c]
                                 : mass_n   * cv_pNew[c] * temp_n[c];
        double ie_km1 = eosOwner ? mass_km1 * es_km1[c]
                                 : mass_km1 * cv_new[c]  * temp_km1[c];
        double ie_a0  = eng_adv0[c];
        double ie_s   = eng_sub[c];

        double E_n   = ie_n   + 0.5*mom_n.length2()   / mass_n;
        double E_km1 = ie_km1 + 0.5*mom_km1.length2() / mass_km1;
        double E_a0  = ie_a0  + 0.5*mom_a0.length2()  / mass_a0;
        double E_s   = ie_s   + 0.5*mom_s.length2()   / mass_s;

        double mass_k = mass_n + (1.0-th)*(mass_a0 - mass_n) + th*(mass_s - mass_km1);
        Vector mom_k  = mom_n  + (mom_a0 - mom_n)*(1.0-th)    + (mom_s - mom_km1)*th;
        double spv_k  = spv_n  + (1.0-th)*(spv_a0 - spv_n)    + th*(spv_s - spv_km1);
        double E_k    = E_n    + (1.0-th)*(E_a0 - E_n)        + th*(E_s - E_km1);

        mass_sub[c]   = mass_k;
        mom_sub[c]    = mom_k;
        sp_vol_sub[c] = spv_k;
        // back out int_eng from the blended TOTAL energy and the blended
        // momentum/mass (avoids creating/destroying heat through the KE
        // nonlinearity -- see Docs/PIMPLE_KT_method.pdf, Section 8).
        eng_sub[c]    = E_k - 0.5*mom_k.length2()/mass_k;
      }
    }  // matl loop

    //__________________________________
    //  Species (transported variables): same theta blend, in the
    //  extensive (mass-weighted) form mass*Y.
    int idx = 0;
    for( auto& model : d_ice->d_models ){
      FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( model );
      if( fb_model ){
        for( auto& tvar : fb_model->d_transVars ){
          const VarLabel* adv0 = pimpleTvarAdv0Label(idx);

          for(int m=0;m<tvar->matls->size();m++){
            int indx = tvar->matls->get(m);

            constCCVariable<double> Y_n, Y_km1, var_adv0, rho_n2, rho_km1_2;
            pOldDW->get( Y_n,       tvar->var,       indx, patch, m_gn,0 );
            old_dw->get( Y_km1,     tvar->var,       indx, patch, m_gn,0 );
            pNewDW->get( var_adv0,  adv0,            indx, patch, m_gn,0 );
            pOldDW->get( rho_n2,    lb->rho_CCLabel, indx, patch, m_gn,0 );
            old_dw->get( rho_km1_2, lb->rho_CCLabel, indx, patch, m_gn,0 );

            CCVariable<double> var_sub;
            new_dw->getModifiable( var_sub, tvar->var_adv, indx, patch );

            for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
              IntVector c = *iter;

              double mass_n   = rho_n2[c]    * vol;
              double mass_km1 = rho_km1_2[c] * vol;

              double sp_n   = mass_n   * Y_n[c];
              double sp_km1 = mass_km1 * Y_km1[c];
              double sp_a0  = var_adv0[c];
              double sp_s   = var_sub[c];

              var_sub[c] = sp_n + (1.0-th)*(sp_a0 - sp_n) + th*(sp_s - sp_km1);
            }
          }
        }
      }
    }
  }  // patch loop
}

/* _____________________________________________________________________
 Task:      SPICE::pimpleResidual
 Purpose:   PIMPLE sub-graph, after conservedtoPrimitive_Vars: residual =
            max over interior cells of
            max(|rho_new-rho_old|/rho_new, |T_new-T_old|/T_new).
 _____________________________________________________________________  */
void SPICE::pimpleResidual(const ProcessorGroup  *,
                         const PatchSubset     * patches,
                         const MaterialSubset  * matls,
                         DataWarehouse         * old_dw,
                         DataWarehouse         * new_dw)
{
  double residual = 0.0;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, m_spice_tasks, "SPICE::pimpleResidual" );

    for(int m=0;m<matls->size();m++){
      int indx = matls->get(m);

      constCCVariable<double> rho_old, temp_old, rho_new, temp_new;
      old_dw->get( rho_old,  lb->rho_CCLabel,  indx, patch, m_gn,0 );
      old_dw->get( temp_old, lb->temp_CCLabel, indx, patch, m_gn,0 );
      new_dw->get( rho_new,  lb->rho_CCLabel,  indx, patch, m_gn,0 );
      new_dw->get( temp_new, lb->temp_CCLabel, indx, patch, m_gn,0 );

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;

        double rRho  = std::fabs(rho_new[c]  - rho_old[c] ) / rho_new[c];
        double rTemp = std::fabs(temp_new[c] - temp_old[c]) / temp_new[c];

        residual = std::max( residual, std::max(rRho, rTemp) );
      }
    }
  }

  new_dw->put( max_vartype(residual), Slb->pimpleResidualLabel );
}

/* _____________________________________________________________________
 Task:      SPICE::pimpleOuterLoop
 Purpose:   Drive d_pimpleSched through outer iterations k=1..nOuter-1,
            seeded from the main graph's own k=0 primitives, and copy the
            final iterate's primitives back.  Follows the structure of
            Uintah_TS's ICE::sdcSweep (DW slot mapping, recompile, seeding
            slot 3 via setInitTimestep(true)+advanceDataWarehouse+
            transferFrom, per-iteration advanceDataWarehouse, re-putting
            timeStep/simTime/delT bracketed by setInitTimestep, ScrubNone,
            recomputeTimeStep/abortTimeStep forwarding, copy-back).
 _____________________________________________________________________  */
void
SPICE::pimpleOuterLoop( const ProcessorGroup * pg,
                      const PatchSubset    * patches,
                      const MaterialSubset * /*matls*/,
                      DataWarehouse        * ParentOldDW,
                      DataWarehouse        * ParentNewDW,
                      LevelP                 level )
{
  if( d_pimpleOuter < 2 ){
    return;
  }

  const GridP grid = level->getGrid();

  DataWarehouse::ScrubMode ParentOldDW_scrubmode =
                           ParentOldDW->setScrubbing(DataWarehouse::ScrubNone);
  DataWarehouse::ScrubMode ParentNewDW_scrubmode =
                           ParentNewDW->setScrubbing(DataWarehouse::ScrubNone);

  const PatchSubset* patch_sub = patches;
  const MaterialSet* ice_matls = m_materialManager->allMaterials( "ICE" );
  const MaterialSet* all_matls = m_materialManager->allMaterials();
  const MaterialSubset* ice_mss = ice_matls->getUnion();

  d_pimpleSched->setParentDWs(ParentOldDW, ParentNewDW);
  d_pimpleSched->advanceDataWarehouse(grid);

  if( d_recompilePimpleSched ){
    d_pimpleSched->initialize(3, 1);
    // matches scheduleTimeAdvance's own ordering (maxMach BEFORE the
    // chain): computesRequires_CustomBCs wires maxMach_<face> into several
    // of scheduleFluidChain's own tasks when usingLodi, so it must be
    // computed in THIS (sub-graph) NewDW every recompile, not reused from
    // the main graph's k=0 value.
    d_ice->scheduleMaxMach_on_Lodi_BC_Faces(  d_pimpleSched, level,  ice_matls );
    scheduleFluidChain(                       d_pimpleSched, level,  true );
    schedulePimpleBlend(                      d_pimpleSched, level,  ice_matls );
    // "afterAdvection" (not a literal "pimpleOuterLoop" string):
    // conservedtoPrimitive_Vars gates its computesVar(temp_CC/vel_CC/
    // mach/species) on this exact string.
    d_ice->scheduleConservedtoPrimitive_Vars( d_pimpleSched, level->eachPatch(),
                                              ice_mss, all_matls, "afterAdvection" );
    schedulePimpleResidual(             d_pimpleSched, level,  ice_matls );
    d_pimpleSched->compile();
    d_recompilePimpleSched = false;
  }

  // prime slot 3 and seed it with u^0 == u^n primitives (k=0 IS the main
  // graph's own, already-computed result)
  d_pimpleSched->setInitTimestep(true);
  d_pimpleSched->advanceDataWarehouse(grid);

  DataWarehouse* seedDW = d_pimpleSched->get_dw(3);

  seedDW->transferFrom( ParentNewDW, lb->rho_CCLabel,        patch_sub, ice_mss );
  seedDW->transferFrom( ParentNewDW, lb->vel_CCLabel,        patch_sub, ice_mss );
  seedDW->transferFrom( ParentNewDW, lb->temp_CCLabel,       patch_sub, ice_mss );
  seedDW->transferFrom( ParentNewDW, lb->sp_vol_CCLabel,     patch_sub, ice_mss );
  seedDW->transferFrom( ParentNewDW, lb->press_CCLabel,      patch_sub, d_ice->d_press_matl );
  seedDW->transferFrom( ParentNewDW, lb->speedSound_CCLabel, patch_sub, ice_mss );
  seedDW->transferFrom( ParentNewDW, lb->specific_heatLabel, patch_sub, ice_mss );
  // the LODI trace-cell guard (and gasCombustion's, multi-matl only) reads
  // vol_frac_CC from OldDW; k=0's value is the previous-pass estimate, as
  // the main graph's OldDW vol_frac is for the k=0 pass
  seedDW->transferFrom( ParentNewDW, lb->vol_frac_CCLabel,   patch_sub, ice_mss );

  for( auto& model : d_ice->d_models ){
    FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( model );
    if( fb_model ){
      for( auto& tvar : fb_model->d_transVars ){
        seedDW->transferFrom( ParentNewDW, tvar->var, patch_sub, tvar->matls );
      }
    }
  }

  d_pimpleSched->setInitTimestep(false);

  double residual = 0.0;

  for (int k = 1; k < d_pimpleOuter; k++) {

    d_pimpleSched->advanceDataWarehouse(grid);

    DataWarehouse* subOldDW = d_pimpleSched->get_dw(2);
    DataWarehouse* subNewDW = d_pimpleSched->get_dw(3);

    // re-put every outer iteration (does not survive rotation);
    // DWDatabase::put needs the init flag to register a not-yet-known key,
    // so bracket it as below.
    timeStep_vartype timeStepVar;
    simTime_vartype  simTimeVar;
    delt_vartype     delTVar;
    ParentOldDW->get( timeStepVar, lb->timeStepLabel );
    ParentOldDW->get( simTimeVar,  lb->simulationTimeLabel );
    ParentOldDW->get( delTVar,     lb->delTLabel, level.get_rep() );
    // for k >= 2, subOldDW is the previous iteration's NewDW, finalized by execute()
    const bool subOldDW_wasFinalized = subOldDW->isFinalized();
    if( subOldDW_wasFinalized ){
      subOldDW->unfinalize();
    }

    d_pimpleSched->setInitTimestep(true);
    subOldDW->put( timeStepVar, lb->timeStepLabel );
    subOldDW->put( simTimeVar,  lb->simulationTimeLabel );
    subOldDW->put( delTVar,     lb->delTLabel, level.get_rep() );
    d_pimpleSched->setInitTimestep(false);

    if( subOldDW_wasFinalized ){
      subOldDW->refinalize();
    }

    subOldDW->setScrubbing(DataWarehouse::ScrubNone);
    subNewDW->setScrubbing(DataWarehouse::ScrubNone);

    d_pimpleSched->execute();

    max_vartype residualVar;
    subNewDW->get( residualVar, Slb->pimpleResidualLabel );
    residual = residualVar;

    if( pg->myRank() == 0 ){
      DOUT( m_spice_pimple, "  PIMPLE outer iteration " << k << " of "
            << (d_pimpleOuter - 1) << "   timestep " << timeStepVar
            << "   residual " << residual );
    }

    // forward a timestep-recompute request out of the sub-graph
    bool_or_vartype recomputeVar(false);
    if( subNewDW->exists( VarLabel::find(recomputeTimeStep_name) ) ){
      subNewDW->get( recomputeVar, VarLabel::find(recomputeTimeStep_name) );
      if( recomputeVar ){
        ParentNewDW->put( bool_or_vartype(true),
                          VarLabel::find(recomputeTimeStep_name) );
        ParentOldDW->setScrubbing(ParentOldDW_scrubmode);
        ParentNewDW->setScrubbing(ParentNewDW_scrubmode);
        return;
      }
    }

    // forward an abort request out of the sub-graph
    bool_or_vartype abortVar(false);
    if( subNewDW->exists( VarLabel::find(abortTimeStep_name) ) ){
      subNewDW->get( abortVar, VarLabel::find(abortTimeStep_name) );
      if( abortVar ){
        ParentNewDW->put( bool_or_vartype(true),
                          VarLabel::find(abortTimeStep_name) );
      }
    }

    if( d_pimpleTol > 0.0 && residual < d_pimpleTol ){
      break;
    }
  }

  // copy the final outer iterate's primitives back to the main graph
  DataWarehouse* subNewDW = d_pimpleSched->get_dw(3);

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    for (int m = 0; m < ice_mss->size(); m++) {
      int indx = ice_mss->get(m);

      constCCVariable<double> rho, temp, sp_vol, mach;
      constCCVariable<Vector> vel;
      subNewDW->get( rho,    lb->rho_CCLabel,    indx, patch, m_gn, 0 );
      subNewDW->get( vel,    lb->vel_CCLabel,    indx, patch, m_gn, 0 );
      subNewDW->get( temp,   lb->temp_CCLabel,   indx, patch, m_gn, 0 );
      subNewDW->get( sp_vol, lb->sp_vol_CCLabel, indx, patch, m_gn, 0 );
      subNewDW->get( mach,   lb->machLabel,      indx, patch, m_gn, 0 );

      CCVariable<double> rho_out, temp_out, sp_vol_out, mach_out;
      CCVariable<Vector> vel_out;
      ParentNewDW->getModifiable( rho_out,    lb->rho_CCLabel,    indx, patch );
      ParentNewDW->getModifiable( vel_out,    lb->vel_CCLabel,    indx, patch );
      ParentNewDW->getModifiable( temp_out,   lb->temp_CCLabel,   indx, patch );
      ParentNewDW->getModifiable( sp_vol_out, lb->sp_vol_CCLabel, indx, patch );
      ParentNewDW->getModifiable( mach_out,   lb->machLabel,      indx, patch );

      for (CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++) {
        IntVector c = *iter;
        rho_out[c]    = rho[c];
        vel_out[c]    = vel[c];
        temp_out[c]   = temp[c];
        sp_vol_out[c] = sp_vol[c];
        mach_out[c]   = mach[c];
      }
    }
  }

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    for (int m = 0; m < d_ice->d_press_matl->size(); m++) {
      int indx = d_ice->d_press_matl->get(m);

      constCCVariable<double> press;
      subNewDW->get( press, lb->press_CCLabel, indx, patch, m_gn, 0 );

      CCVariable<double> press_out;
      ParentNewDW->getModifiable( press_out, lb->press_CCLabel, indx, patch );

      for (CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++) {
        IntVector c = *iter;
        press_out[c] = press[c];
      }
    }
  }

  for( auto& model : d_ice->d_models ){
    FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( model );
    if( fb_model ){
      for( auto& tvar : fb_model->d_transVars ){

        for (int p = 0; p < patches->size(); p++) {
          const Patch* patch = patches->get(p);

          for (int mm = 0; mm < tvar->matls->size(); mm++) {
            int tindx = tvar->matls->get(mm);

            constCCVariable<double> var_sub;
            subNewDW->get( var_sub, tvar->var, tindx, patch, m_gn, 0 );

            CCVariable<double> var_out;
            ParentNewDW->getModifiable( var_out, tvar->var, tindx, patch );

            for (CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++) {
              IntVector c = *iter;
              var_out[c] = var_sub[c];
            }
          }
        }
      }
    }
  }

  ParentOldDW->setScrubbing(ParentOldDW_scrubmode);
  ParentNewDW->setScrubbing(ParentNewDW_scrubmode);
}

