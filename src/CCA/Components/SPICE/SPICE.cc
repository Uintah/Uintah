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
#include <CCA/Components/SPICE/Advection/InterfaceSecondOrderAdvector.h>
//  ICE/Advection/SecondOrderBase.h #defines d_SMALL_NUM, which would clobber
//  the d_SMALL_NUM members declared in ICE.h and ExchangeModel.h below
#undef d_SMALL_NUM
#include <CCA/Components/SPICE/Core/SPICELabel.h>
#include <sci_defs/uintah_defs.h>
#ifndef NO_MPM
#  include <CCA/Components/SPICE/Exchange/InterfaceScalarExch.h>
#endif

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
  Dout m_spice_pitch( "SPICE_PITCH", "SPICE", "Print PITCH outer-loop iteration residual", false );
  Dout m_spice_pressWork( "SPICE_pressWork", "SPICE", "Check the pressure work's face volume flux against the pressure equation (term2)", false );

  //  FSPICE no-penetration face partition: hand the per-face override to
  //  whichever advector carries this material's vol_frac (the KT advector
  //  for the gas, the SecondOrder copy for MPM materials).  nullptr clears it.
  void setFaceFractions( Advector                   * adv,
                         const SFCXVariable<double> * faceFracX,
                         const SFCYVariable<double> * faceFracY,
                         const SFCZVariable<double> * faceFracZ )
  {
    if( auto ktAdv = dynamic_cast<CentralUpwindAdvector*>( adv ) ){
      ktAdv->setFaceFractions( faceFracX, faceFracY, faceFracZ );
    }
    else if( auto soAdv = dynamic_cast<InterfaceSecondOrderAdvector*>( adv ) ){
      soAdv->setFaceFractions( faceFracX, faceFracY, faceFracZ );
    }
    else {
      throw InternalError( "SPICE: <interfaceTransport> donorAcceptor needs the CentralUpwind "
                           "advector for the gas and InterfaceSecondOrderAdvector for MPM materials",
                           __FILE__, __LINE__ );
    }
  }
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
  //  the PITCH sub-graph holds tasks bound to both SPICE and d_ice
  d_pitchSched = nullptr;

  if( d_ktICE_matl && d_ktICE_matl->removeReference() ){
    delete d_ktICE_matl;
  }

  for( auto label : d_pitchTvarAdv0Labels ){
    VarLabel::destroy( label );
  }

  delete d_mpmAdvector;

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

    //__________________________________
    //  Gas-solid interface options (MPM materials, FSPICE).  With MPM
    //  materials the defaults are donorAcceptor + conduction + model; without
    //  them upwind + stock + stock (and donorAcceptor is an error).
    //  interfaceTransport: how gas crosses faces that touch a solid cell
    //   (ktOpen < 1).
    //    upwind        : stock, the donor cell's average composition crosses
    //    donorAcceptor : no-penetration (Hirt-Nichols donor-acceptor), see
    //                    SPICE::computeCentralFluxes
    //  interfaceExchange: where gas and solid exchange heat, and how the gas
    //   next to a solid books its pressure work.
    //    stock     : heat exchange in every cell with both (incl. the node-
    //                averaged smear of the solid's mass into the gas-side cell)
    //    particles : heat exchange only in cells with particles, no face
    //                momentum exchange between two cells without particles
    //                (InterfaceScalarExch), and no pressure work for
    //                placeholder trace gas (accumulateEnergySourceSinks)
    //    conduction: particles, but no gas-solid heat exchange in any cell;
    //                heat is conducted across the interface instead
    //                (SPICE::interfaceConduction), from the gas conductivity
    //                (thermalCond_CC) and the solid's thermal_conductivity
    //  interfaceShear: the gas's tangential momentum at a gas/solid interface.
    //    stock : the cell in front of a solid (no particles, only the node-
    //            averaged smear of its mass) is held at the solid's velocity
    //            in every direction by the 1e15 momentum exchange: a no-slip
    //            wall half a cell off the surface, at any viscosity
    //    slip  : that cell keeps the exchange only along the surface normal
    //            (InterfaceScalarExch); the gas slides freely
    //    model : slip, plus the wall shear from the near-wall velocity
    //            profile (SPICE::interfaceShear, the momentum form of the
    //            conduction layer model), from the gas viscosity
    const bool withMPM = ( m_materialManager->getNumMatls( "MPM" ) > 0 );

    string interfaceTransport = withMPM ? "donorAcceptor" : "upwind";
    string interfaceExchange  = withMPM ? "conduction"    : "stock";
    string interfaceShear     = withMPM ? "model"         : "stock";

    if( cu_ps ){
      cu_ps->getWithDefault( "solidCellFrac",  d_ktSolidCellFrac, 1.0e-3 );
      if( !( d_ktSolidCellFrac > 0.0 && d_ktSolidCellFrac < 0.1 ) ){
        throw ProblemSetupException(
          "\n ERROR: <CentralUpwind><solidCellFrac> must be > 0 and < 0.1\n",
          __FILE__, __LINE__);
      }
      cu_ps->get( "interfaceTransport", interfaceTransport );
      cu_ps->get( "interfaceExchange",  interfaceExchange );
      cu_ps->get( "interfaceShear",     interfaceShear );
    }

    if( d_useCentralUpwind ){
      if( interfaceTransport == "donorAcceptor" ){
        d_ktInterfaceDA = true;
      }
      else if( interfaceTransport != "upwind" ){
        throw ProblemSetupException(
          "\n ERROR: <CentralUpwind><interfaceTransport> must be 'upwind' or 'donorAcceptor'\n",
          __FILE__, __LINE__);
      }

      if( interfaceExchange == "particles" ){
        d_ktInterfaceExch = true;
      }
      else if( interfaceExchange == "conduction" ){
        d_ktInterfaceExch = true;
        d_ktInterfaceCond = true;
      }
      else if( interfaceExchange != "stock" ){
        throw ProblemSetupException(
          "\n ERROR: <CentralUpwind><interfaceExchange> must be 'stock', 'particles' or 'conduction'\n",
          __FILE__, __LINE__);
      }
      if( d_ktInterfaceExch && !d_ktInterfaceDA ){
        throw ProblemSetupException(
          "\n ERROR: <CentralUpwind><interfaceExchange> particles/conduction needs <interfaceTransport> donorAcceptor\n",
          __FILE__, __LINE__);
      }

      if( interfaceShear == "slip" ){
        d_ktInterfaceSlip = true;
      }
      else if( interfaceShear == "model" ){
        d_ktInterfaceSlip  = true;
        d_ktInterfaceShear = true;
      }
      else if( interfaceShear != "stock" ){
        throw ProblemSetupException(
          "\n ERROR: <CentralUpwind><interfaceShear> must be 'stock', 'slip' or 'model'\n",
          __FILE__, __LINE__);
      }
      if( d_ktInterfaceSlip && !d_ktInterfaceDA ){
        throw ProblemSetupException(
          "\n ERROR: <CentralUpwind><interfaceShear> slip/model needs <interfaceTransport> donorAcceptor\n",
          __FILE__, __LINE__);
      }

      if( d_ktInterfaceDA && !withMPM ){
        throw ProblemSetupException(
          "\n ERROR: <CentralUpwind><interfaceTransport> donorAcceptor needs MPM materials (FSPICE)\n",
          __FILE__, __LINE__);
      }
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
  //  Optional <PITCH> block
  ProblemSpecP pitch_ps = cfd_spice_ps->findBlock("PITCH");
  if( pitch_ps ){
    pitch_ps->getWithDefault( "nOuterCorrectors", d_pitchOuter, 1 );
    if( d_pitchOuter < 1 ){
      throw ProblemSetupException(
        "\n ERROR: <PITCH><nOuterCorrectors> must be >= 1\n", __FILE__, __LINE__);
    }

    pitch_ps->getWithDefault( "nCorrectors", d_pitchNCorr, 1 );
    if( d_pitchNCorr < 1 ){
      throw ProblemSetupException(
        "\n ERROR: <PITCH><nCorrectors> must be >= 1\n", __FILE__, __LINE__);
    }

    pitch_ps->getWithDefault( "theta", d_pitchTheta, 0.5 );
    if( d_pitchTheta <= 0.0 || d_pitchTheta > 1.0 ){
      throw ProblemSetupException(
        "\n ERROR: <PITCH><theta> must satisfy 0 < theta <= 1\n", __FILE__, __LINE__);
    }

    pitch_ps->getWithDefault( "outerTolerance", d_pitchTol, 0.0 );
    if( d_pitchTol < 0.0 ){
      throw ProblemSetupException(
        "\n ERROR: <PITCH><outerTolerance> must be >= 0\n", __FILE__, __LINE__);
    }
  }

  //__________________________________
  //  Bulletproofing that needs ICE's setup (impICE, materials, models)
  if( m_materialManager->getNumMatls( "MPM" ) > 0 && !d_hostedByFSPICE ){
    throw ProblemSetupException(
      "\n ERROR: SPICE does not support MPM materials (use FSPICE)\n", __FILE__, __LINE__);
  }

  //  SPICE is a single-fluid code: one ICE material, with any species carried
  //  as transported scalars (e.g. gasCombustion), as in reactingPimpleCentralFoam.
  //  The KT face reconstruction and the PITCH blend are single-material, and
  //  the multi-material f_theta split of the pressure work is untested here.
  //  Use ICE for multi-material problems.
  if( m_materialManager->getNumMatls( "ICE" ) != 1 ){
    ostringstream msg;
    msg << "\n ERROR: SPICE requires exactly one ICE material (found "
        << m_materialManager->getNumMatls( "ICE" ) << ").  Use ICE for multi-material problems.\n";
    throw ProblemSetupException( msg.str(), __FILE__, __LINE__ );
  }

  if( d_pitchOuter > 1 ){
    if( d_ice->d_impICE ){
      throw ProblemSetupException(
        "\n ERROR: PITCH outer loop with the implicit pressure solve is not supported yet\n",
        __FILE__, __LINE__);
    }

    // The PITCH sub-graph re-runs scheduleFluidChain() and recomputes the
    // LODI maxMach_<face> values, but not the turbulence variance, which
    // scheduleTimeAdvance computes outside the fluid chain.
    if( d_ice->d_turbulence ){
      throw ProblemSetupException(
        "\n ERROR: PITCH outer loop with a turbulence model is not "
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

    //  MPM materials' vol_frac in the pressure equation keeps a SecondOrder
    //  advector (KT applies to the gas only).  With the KT advector, gas
    //  stayed trapped inside a moving piston.
    if( m_materialManager->getNumMatls( "MPM" ) > 0 ){
      d_mpmAdvector = scinew InterfaceSecondOrderAdvector();
    }
  }

  //__________________________________
  //  <interfaceExchange> particles/conduction and <interfaceShear> slip/model:
  //  replace ICE's ScalarExch by the subclass that carries them
  if( d_ktInterfaceExch || d_ktInterfaceSlip ){
#ifndef NO_MPM
    if( dynamic_cast<ExchangeModels::ScalarExch*>( d_ice->d_exchModel ) == nullptr ){
      throw ProblemSetupException(
        "\n ERROR: <CentralUpwind><interfaceExchange> particles/conduction and "
        "<interfaceShear> slip/model need the scalar exchange model\n", __FILE__, __LINE__);
    }

    //  the <MaterialProperties> block ICE built its exchange model from
    ProblemSpecP matProp_ps = prob_spec->findBlockWithOutAttribute("MaterialProperties");
    if( !matProp_ps && restart_prob_spec ){
      matProp_ps = restart_prob_spec->findBlockWithOutAttribute("MaterialProperties");
    }

    ExchangeModels::InterfaceScalarExch* exch =
      scinew ExchangeModels::InterfaceScalarExch( matProp_ps, m_materialManager, d_ice->d_with_mpm );
    exch->problemSetup( matProp_ps );
    exch->setSolidLabels( Slb->solidVolFracP_CCLabel, Slb->solidInteriorP_CCLabel );

    if( d_ktInterfaceExch ){
      exch->setParticleAware( 1.0 - d_ktCoveredSolid, d_ktInterfaceCond );
      if( d_ktInterfaceCond ){
        proc0cout << " interfaceExchange conduction: the gas-solid <heat> exchange coefficient is"
                  << " not used; heat crosses the gas/solid interface by conduction\n";
      }
    }
    if( d_ktInterfaceSlip ){
      exch->setTangentialSlip( 1.0 - d_ktCoveredSolid );
    }

    //  the same fluids-based models ICE handed its exchange model
    std::vector<FluidsBasedModel*> fbModels;
    for( auto& model : d_ice->d_models ){
      FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( model );
      if( fb_model ){
        fbModels.push_back( fb_model );
      }
    }
    if( !fbModels.empty() ){
      exch->setFluidsBasedModels( fbModels );
    }

    delete d_ice->d_exchModel;
    d_ice->d_exchModel = exch;
#else
    throw ProblemSetupException(
      "\n ERROR: the gas-solid interface options need a build with MPM\n", __FILE__, __LINE__);
#endif
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
              << " ktPressureFlux: " << d_ktPressureFlux;
    if( m_materialManager->getNumMatls( "MPM" ) > 0 ){
      proc0cout << " solidCellFrac: "   << d_ktSolidCellFrac
                << " interfaceTransport: " << ( d_ktInterfaceDA ? "donorAcceptor" : "upwind" )
                << " interfaceExchange: "  << ( d_ktInterfaceCond ? "conduction" :
                                                d_ktInterfaceExch ? "particles" : "stock" )
                << " interfaceShear: "     << ( d_ktInterfaceShear ? "model" :
                                                d_ktInterfaceSlip ? "slip" : "stock" );
    }
    proc0cout << "\n";
  }
  if( d_pitchOuter > 1 ){
    proc0cout << " PITCH outer loop: "
              << " nOuterCorrectors: " << d_pitchOuter
              << " nCorrectors: "      << d_pitchNCorr
              << " theta: "            << d_pitchTheta
              << " outerTolerance: "   << d_pitchTol << "\n";
  }
  proc0cout << "__________________________________ \n";

  //__________________________________
  //  PITCH outer-loop sub-scheduler, created exactly like ICE's implicit
  //  pressure sub-scheduler.
  if( d_pitchOuter > 1 ){
    d_pitchSched = m_scheduler->createSubScheduler();
    d_pitchSched->initialize(3,1);
    d_pitchSched->clearMappings();
    d_pitchSched->mapDataWarehouse( Task::ParentOldDW, 0 );
    d_pitchSched->mapDataWarehouse( Task::ParentNewDW, 1 );
    d_pitchSched->mapDataWarehouse( Task::OldDW,       2 );
    d_pitchSched->mapDataWarehouse( Task::NewDW,       3 );
    // single-level convention (matches SchedulerCommon's own default
    // mapping): Coarse == Old/New so getSubCycleProgress() degenerates
    // safely.
    d_pitchSched->mapDataWarehouse( Task::CoarseOldDW, 2 );
    d_pitchSched->mapDataWarehouse( Task::CoarseNewDW, 3 );

    d_recompilePitchSched = true;

    //  One "*_adv0" capture label per transported variable, in the nested
    //  (model, transported variable) order every loop over d_models uses.
    for( auto& model : d_ice->d_models ){
      FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( model );
      if( fb_model ){
        for( auto& tvar : fb_model->d_transVars ){
          d_pitchTvarAdv0Labels.push_back(
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

  if( d_pitchOuter > 1 ){
    // snapshot the k=0 (main graph) advected conserved state so the outer
    // loop's sub-graph can read it from ParentNewDW without also modifying
    // the same label (mass_adv/mom_adv/eng_adv/sp_vol_adv/tvar->var_adv)
    // in its own NewDW
    scheduleCapturePitchAdv0(             sched, level,   ice_matls);
  }

  d_ice->scheduleConservedtoPrimitive_Vars(sched, patches, ice_mss,
                                                           all_matls,
                                                           "afterAdvection");

  if( d_pitchOuter > 1 ){
    schedulePitchOuterLoop(               sched, level );
  }
}
/* _____________________________________________________________________
 Task:  SPICE::scheduleFluidChain--
 Purpose:  Schedule the fluid-dynamics portion of a single timestep, from
           the thermo/transport properties through the advection step,
           so it can be scheduled again for the PITCH outer loop.

           ICE tasks are scheduled through d_ice.  The tasks SPICE owns are
           scheduled here; computeVel_FC, computePressFC and
           computeDelPressAndUpdatePressCC schedule the SPICE copy only when
           the Kurganov-Tadmor option that changes them is on, and ICE's
           otherwise (see SPICE::scheduleComputeVel_FC).
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
    // PITCH outer loop (reactingPimpleCentralFoam: reaction->correct()
    // precedes pimple.loop())
    d_ice->scheduleComputeModelSources(          sched, level,   all_matls);
  }
  else {
    // PITCH outer iterations k>=1: model sources are FROZEN at the k=0
    // value -- copy them from ParentNewDW instead of recomputing them.
    schedulePitchCopyModelSources(              sched, level,   all_matls);
  }

  d_ice->scheduleUpdateVolumeFraction(           sched, level,   press_matl,
                                                                 all_matls);

  scheduleComputeVel_FC(                         sched, patches, ice_mss,
                                                                 mpm_mss,
                                                                 press_matl,
                                                                 all_matls);

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

  if(d_ice->d_impICE) {        //  I M P L I C I T  (KT and PITCH refuse impICE)

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
    scheduleComputeDelPressAndUpdatePressCC(     sched, patches, press_matl,
                                                                 ice_mss,
                                                                 mpm_mss,
                                                                 all_matls);
  }

  scheduleComputePressFC(                        sched, patches, press_matl,
                                                                 all_matls);

  d_ice->scheduleVelTau_CC(                      sched, patches, ice_matls);

  d_ice->scheduleViscousShearStress(             sched, patches, ice_matls);

  scheduleAccumulateMomentumSourceSinks(         sched, patches, press_matl,
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

  scheduleComputeLagrangianSpecificVolume(       sched, patches, ice_mss,
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
  //  ICE's task unless the KT face velocity is on.  The flags are fixed at
  //  problemSetup, so the main graph and the PITCH sub-graph (and FSPICE)
  //  always choose the same way.
  if( !( d_useCentralUpwind && d_ktFaceVel ) ){
    d_ice->scheduleComputeVel_FC( sched, patches, ice_mss, mpm_mss, press_mss, all_matls );
    return;
  }

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
    if( d_ice->d_with_mpm ){
      t->requiresVar( Task::NewDW, Slb->ktOpen_FCXLabel, ice_mss, m_gn,0 );
      t->requiresVar( Task::NewDW, Slb->ktOpen_FCYLabel, ice_mss, m_gn,0 );
      t->requiresVar( Task::NewDW, Slb->ktOpen_FCZLabel, ice_mss, m_gn,0 );
    }
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

  if( d_ice->d_with_mpm ){
    // FSPICE: open-face weight from the ICE material's volume fraction
    t->requiresVar( Task::NewDW, lb->vol_frac_CCLabel, m_gac,1 );
    t->computesVar( Slb->ktOpen_FCXLabel );
    t->computesVar( Slb->ktOpen_FCYLabel );
    t->computesVar( Slb->ktOpen_FCZLabel );
  }

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

  // donorAcceptor caps a donor's gas outflow by its total outflux volume,
  // so it needs the face velocities around a ghost donor cell too
  if( d_ktInterfaceDA ){
    t->requiresVar( Task::OldDW, lb->delTLabel, getLevel(patches) );
    t->requiresVar( Task::NewDW, lb->uvel_FCMELabel,     m_gac,2 );
    t->requiresVar( Task::NewDW, lb->vvel_FCMELabel,     m_gac,2 );
    t->requiresVar( Task::NewDW, lb->wvel_FCMELabel,     m_gac,2 );
    t->requiresVar( Task::NewDW, Slb->ktOpen_FCXLabel,    m_gn,0 );
    t->requiresVar( Task::NewDW, Slb->ktOpen_FCYLabel,    m_gn,0 );
    t->requiresVar( Task::NewDW, Slb->ktOpen_FCZLabel,    m_gn,0 );
    t->requiresVar( Task::NewDW, lb->vol_frac_CCLabel,   m_gac,2 );
    t->requiresVar( Task::NewDW, Slb->solidVolFracP_CCLabel,  m_gac,2 );
    t->requiresVar( Task::NewDW, Slb->solidInteriorP_CCLabel, m_gac,2 );
  } else {
    t->requiresVar( Task::NewDW, lb->uvel_FCMELabel,     m_gn,0 );
    t->requiresVar( Task::NewDW, lb->vvel_FCMELabel,     m_gn,0 );
    t->requiresVar( Task::NewDW, lb->wvel_FCMELabel,     m_gn,0 );
  }
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
  if( d_ktInterfaceDA ){
    t->computesVar( Slb->gasFaceFrac_FCXLabel );
    t->computesVar( Slb->gasFaceFrac_FCYLabel );
    t->computesVar( Slb->gasFaceFrac_FCZLabel );
  }

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
                                                  const MaterialSubset  * mpm_matls,
                                                  const MaterialSet     * matls)
{
  //  ICE's task unless KT is on (see scheduleComputeVel_FC)
  if( !d_useCentralUpwind ){
    d_ice->scheduleComputeDelPressAndUpdatePressCC( sched, patches, press_matl,
                                                    ice_matls, mpm_matls, matls );
    return;
  }

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
    task->requiresVar( Task::NewDW, Slb->ktUvol_FCXLabel, ice_matls, m_gac,2 );
    task->requiresVar( Task::NewDW, Slb->ktUvol_FCYLabel, ice_matls, m_gac,2 );
    task->requiresVar( Task::NewDW, Slb->ktUvol_FCZLabel, ice_matls, m_gac,2 );
  }

  if( d_ktInterfaceDA ){
    // no-penetration gas share of each interface face flux; gac 2 (as for
    // uvel_FCME) so the high face of the patch's last cell is fetched
    task->requiresVar( Task::NewDW, Slb->gasFaceFrac_FCXLabel, ice_matls, m_gac,2 );
    task->requiresVar( Task::NewDW, Slb->gasFaceFrac_FCYLabel, ice_matls, m_gac,2 );
    task->requiresVar( Task::NewDW, Slb->gasFaceFrac_FCZLabel, ice_matls, m_gac,2 );
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

  //  the pressure equation's face volume flux, for the pressure work
  //  (accumulateEnergySourceSinks, PressureWorkFix.md)
  task->computesVar( Slb->volFlux_FCXLabel,    press_matl, oims );
  task->computesVar( Slb->volFlux_FCYLabel,    press_matl, oims );
  task->computesVar( Slb->volFlux_FCZLabel,    press_matl, oims );

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
  //  ICE's task unless the KT face pressure is on (see scheduleComputeVel_FC)
  if( !( d_useCentralUpwind && d_ktFacePress ) ){
    d_ice->scheduleComputePressFC( sched, patches, press_matl, matls );
    return;
  }

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
    if( d_ice->d_with_mpm ){
      task->requiresVar( Task::NewDW, Slb->ktOpen_FCXLabel, d_ktICE_matl, oims, m_gn,0 );
      task->requiresVar( Task::NewDW, Slb->ktOpen_FCYLabel, d_ktICE_matl, oims, m_gn,0 );
      task->requiresVar( Task::NewDW, Slb->ktOpen_FCZLabel, d_ktICE_matl, oims, m_gn,0 );
    }
  }

  task->computesVar( lb->pressX_FCLabel, press_matl, oims);
  task->computesVar( lb->pressY_FCLabel, press_matl, oims);
  task->computesVar( lb->pressZ_FCLabel, press_matl, oims);

  sched->addTask( task, patches, matls);
}

/* _____________________________________________________________________
Task:     SPICE::scheduleAccumulateMomentumSourceSinks--
Purpose:  Sum up all of the momentum sources or sinks
_____________________________________________________________________*/
//  Copied from ICE::scheduleAccumulateMomentumSourceSinks and modified for SPICE.  Keep it diffable against the
//  ICE original so ICE fixes can be ported.
void
SPICE::scheduleAccumulateMomentumSourceSinks(SchedulerP           & sched,
                                           const PatchSet       * patches,
                                           const MaterialSubset * press_mss,
                                           const MaterialSubset * ice_mss,
                                           const MaterialSubset * mpm_mss,
                                           const MaterialSet    * matls)
{
  printSchedule( patches, m_spice_tasks, " SPICE::scheduleAccumulateMomentumSourceSinks" );

  Task* t = scinew Task("SPICE::accumulateMomentumSourceSinks",
            this, &SPICE::accumulateMomentumSourceSinks);

  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.

  t->requiresVar( Task::OldDW, lb->delTLabel,getLevel(patches));
  t->requiresVar( Task::NewDW, lb->pressX_FCLabel,     press_mss, oims, m_gac,1);
  t->requiresVar( Task::NewDW, lb->pressY_FCLabel,     press_mss, oims, m_gac,1);
  t->requiresVar( Task::NewDW, lb->pressZ_FCLabel,     press_mss, oims, m_gac,1);
  t->requiresVar( Task::NewDW, lb->viscous_src_CCLabel, ice_mss,  m_gn,0);

  //  <interfaceShear> model (SPICE::interfaceShear): interface owners up to 2
  //  cells outside the patch, as interfaceConduction
  const bool shear = d_ice->d_with_mpm && d_ktInterfaceShear;
  const Ghost::GhostType gShear = shear ? m_gac : m_gn;
  t->requiresVar( Task::NewDW, lb->rho_CCLabel,         gShear, shear ? 2 : 0 );
  t->requiresVar( Task::NewDW, lb->vol_frac_CCLabel,    gShear, shear ? 3 : 0 );
  if( shear ){
    t->requiresVar( Task::NewDW, Slb->solidVolFracP_CCLabel, ice_mss, m_gac, 3 );
    t->requiresVar( Task::OldDW, lb->vel_CCLabel,           ice_mss, m_gac, 3 );
    t->requiresVar( Task::OldDW, lb->rho_CCLabel,           ice_mss, m_gac, 2 );
    t->requiresVar( Task::NewDW, lb->viscosityLabel,        ice_mss, m_gac, 2 );
    t->requiresVar( Task::NewDW, lb->vel_CCLabel,           mpm_mss, m_gac, 3 );
  }

  t->computesVar( lb->mom_source_CCLabel );
  sched->addTask(t, patches, matls);
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
                                              const MaterialSubset* mpm_mss,
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

  //  the pressure equation's face volume flux (PressureWorkFix.md)
  if( d_useCentralUpwind && !d_ice->d_impICE ){
    //  SPICE's own pressure task stored it (computeDelPressAndUpdatePressCC)
    t->requiresVar( Task::NewDW, Slb->volFlux_FCXLabel, press_mss,oims, m_gac,1 );
    t->requiresVar( Task::NewDW, Slb->volFlux_FCYLabel, press_mss,oims, m_gac,1 );
    t->requiresVar( Task::NewDW, Slb->volFlux_FCZLabel, press_mss,oims, m_gac,1 );
  }
  else{
    //  ICE's pressure task (explicit without KT) or the implicit solve, which
    //  leaves the converged values in these: rebuilt in the task
    t->requiresVar( Task::NewDW, lb->uvel_FCMELabel,    m_gac,1 );
    t->requiresVar( Task::NewDW, lb->vvel_FCMELabel,    m_gac,1 );
    t->requiresVar( Task::NewDW, lb->wvel_FCMELabel,    m_gac,1 );
    t->requiresVar( Task::NewDW, lb->vol_fracX_FCLabel, m_gac,1 );
    t->requiresVar( Task::NewDW, lb->vol_fracY_FCLabel, m_gac,1 );
    t->requiresVar( Task::NewDW, lb->vol_fracZ_FCLabel, m_gac,1 );
  }
  //  SCI_DEBUG SPICE_pressWork: the flux against the pressure equation's
  //  dilatation.  Explicit only: impICE keeps the mass exchange term in term2.
  if( m_spice_pressWork.active() && !d_ice->d_impICE ){
    t->requiresVar( Task::NewDW, lb->term2Label,        press_mss,oims, m_gn,0 );
  }
  //  <interfaceShear> model: the shear's heat (SPICE::interfaceShear)
  const bool shear = d_ice->d_with_mpm && d_ktInterfaceShear;
  const int  gVel  = shear ? 3 : 1;
  t->requiresVar( Task::OldDW, lb->vel_CCLabel,       ice_mss, m_gac,gVel );
  t->requiresVar( Task::NewDW, lb->f_theta_CCLabel,            m_gn );
  if(d_ice->d_with_mpm){
    //  MPM matls are part of the mixture W; their time n CC velocity is
    //  computed in the new DW by FSPICE::interpolateNCToCC_0
    t->requiresVar( Task::NewDW, lb->vel_CCLabel,     mpm_mss, m_gac,gVel );

    //  <interfaceExchange> particles/conduction: placeholder trace gas books
    //  no pressure work
    if( d_ktInterfaceExch ){
      t->requiresVar( Task::NewDW, Slb->solidInteriorP_CCLabel, ice_mss, m_gn );
    }
  }
  //  <interfaceExchange> conduction (SPICE::interfaceConduction): interface
  //  owners up to 2 cells outside the patch, each reading the solid fraction
  //  one cell further; pair conductances on nodes up to 2 cells out
  const bool cond = d_ice->d_with_mpm && d_ktInterfaceCond;
  const int  gWide = ( cond || shear ) ? 2 : 1;
  const int  gFar  = ( cond || shear ) ? 3 : 1;
  t->requiresVar( Task::OldDW, lb->temp_CCLabel,      ice_mss, m_gac,gFar );
  t->requiresVar( Task::NewDW, lb->thermalCondLabel,  ice_mss, m_gac,gWide );
  t->requiresVar( Task::NewDW, lb->rho_CCLabel,                m_gac,gWide );
  t->requiresVar( Task::NewDW, lb->sp_vol_CCLabel,             m_gac,1 );
  t->requiresVar( Task::NewDW, lb->vol_frac_CCLabel,           m_gac,gFar );
  if( cond || shear ){
    t->requiresVar( Task::NewDW, Slb->solidVolFracP_CCLabel, ice_mss, m_gac, 3 );
    t->requiresVar( Task::OldDW, lb->rho_CCLabel,           ice_mss, m_gac, 2 );
  }
  if( cond ){
    t->requiresVar( Task::NewDW, lb->specific_heatLabel,    ice_mss, m_gac, 2 );
    t->requiresVar( Task::NewDW, lb->temp_CCLabel,          mpm_mss, m_gac, 3 );
  }
  if( shear ){
    t->requiresVar( Task::NewDW, lb->viscosityLabel,        ice_mss, m_gac, 2 );
  }

  t->computesVar( lb->int_eng_source_CCLabel );
  t->computesVar( lb->heatCond_src_CCLabel );
  sched->addTask(t, patches, matls);
}

/* _____________________________________________________________________
 Task:     scheduleComputeLagrangianSpecificVolume--
 Purpose:
_____________________________________________________________________*/
void SPICE::scheduleComputeLagrangianSpecificVolume(SchedulerP            & sched,
                                                  const PatchSet        * patches,
                                                  const MaterialSubset  * ice_mss,
                                                  const MaterialSubset  * mpm_mss,
                                                  const MaterialSubset  * press_mss,
                                                  const MaterialSet     * matls)
{
  printSchedule( patches, m_spice_tasks, " SPICE::scheduleComputeLagrangianSpecificVolume" );

  Task* t = scinew Task("SPICE::computeLagrangianSpecificVolume",this,
                        &SPICE::computeLagrangianSpecificVolume);

  Task::MaterialDomainSpec oims = Task::OutOfDomain;  //outside of ice matlSet.

  t->requiresVar( Task::OldDW, lb->delTLabel,getLevel(patches));
  t->requiresVar( Task::NewDW, lb->rho_CCLabel,         m_gn );
  t->requiresVar( Task::NewDW, lb->sp_vol_CCLabel,      m_gn );
  t->requiresVar( Task::NewDW, lb->Tdot_CCLabel,        m_gn );
  t->requiresVar( Task::NewDW, lb->f_theta_CCLabel,     m_gn );
  t->requiresVar( Task::NewDW, lb->compressibilityLabel,m_gn );
  t->requiresVar( Task::NewDW, lb->vol_frac_CCLabel,    m_gac,1 );

  t->requiresVar( Task::OldDW, lb->temp_CCLabel,        ice_mss, m_gn );
  t->requiresVar( Task::NewDW, lb->specific_heatLabel,  ice_mss, m_gn );

  if( mpm_mss ){
    t->requiresVar( Task::NewDW, lb->temp_CCLabel,      mpm_mss, m_gn );
  }

  t->requiresVar( Task::NewDW, lb->delP_DilatateLabel,  press_mss,oims,m_gn );
  t->requiresVar( Task::NewDW, lb->press_CCLabel,       press_mss,oims,m_gn );
  if(d_ice->d_with_mpm){
   t->requiresVar( Task::NewDW,lb->TMV_CCLabel,       press_mss,oims, m_gn );
  }

  if(d_ice->d_models.size() > 0){
    t->requiresVar( Task::NewDW, lb->modelVol_srcLabel,    m_gn);
  }

  t->computesVar( lb->sp_vol_L_CCLabel );
  t->computesVar( lb->sp_vol_src_CCLabel );

  t->computesVar( VarLabel::find(abortTimeStep_name) );
  t->computesVar( VarLabel::find(recomputeTimeStep_name) );

  sched->addTask(t, patches, matls);
}

/* _____________________________________________________________________
Task:     SPICE::schedulePitchCopyModelSources--
Purpose:  PITCH outer iterations k>=1 freeze the chemistry/model sources
          at their k=0 value (reactingPimpleCentralFoam calls
          reaction->correct() once, before pimple.loop()).  Scheduled by
          scheduleFluidChain() in place of scheduleComputeModelSources()
          when inOuterLoop==true: computes, in the sub-graph's own NewDW,
          exactly the label set ICE::zeroModelSources() computes, by
          copying from Task::ParentNewDW (the main graph's k=0 result).
_____________________________________________________________________*/
void SPICE::schedulePitchCopyModelSources(SchedulerP        & sched,
                                         const LevelP      & level,
                                         const MaterialSet * matls)
{
  if(d_ice->d_models.size() != 0){

    printSchedule( level, m_spice_tasks, " SPICE::schedulePitchCopyModelSources" );

    Task* t = scinew Task("SPICE::pitchCopyModelSources",this,
                          &SPICE::pitchCopyModelSources);

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
Task:     SPICE::scheduleCapturePitchAdv0--
Purpose:  Snapshot the main graph's (k=0) advected conserved state into
          the "*_adv0" labels right after scheduleAdvectAndAdvanceInTime,
          so SPICE::pitchBlend can read it from ParentNewDW without also
          modifying mass_adv/mom_adv/eng_adv/sp_vol_adv/tvar->var_adv (the
          SAME labels) in the sub-graph's own NewDW.
_____________________________________________________________________*/
void SPICE::scheduleCapturePitchAdv0(SchedulerP        & sched,
                                    const LevelP      & level,
                                    const MaterialSet * matls)
{
  printSchedule( level, m_spice_tasks, " SPICE::scheduleCapturePitchAdv0" );

  Task* t = scinew Task("SPICE::capturePitchAdv0", this, &SPICE::capturePitchAdv0);

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
        const VarLabel* adv0 = pitchTvarAdv0Label(idx);
        t->requiresVar( Task::NewDW, tvar->var_adv, tvar->matls, m_gn,0 );
        t->computesVar( adv0, tvar->matls );
      }
    }
  }

  sched->addTask(t, level->eachPatch(), matls);
}

/* _____________________________________________________________________
Task:     SPICE::schedulePitchBlend--
Purpose:  PITCH outer iteration k (sub-graph only): combine the k=0
          advected state with this iteration's freshly advected state
          using the Crank-Nicolson theta blend
            X^k = X^n + (1-theta)*(X_adv0 - X^n) + theta*(X_sub - X^{k-1})
          (reactingPimpleCentralFoam, ddt CrankNicolson).  Total energy is
          blended (not int_eng directly), to avoid creating/destroying
          heat through the KE nonlinearity; see SPICE::pitchBlend.
_____________________________________________________________________*/
void SPICE::schedulePitchBlend(SchedulerP        & sched,
                              const LevelP      & level,
                              const MaterialSet * ice_matls)
{
  printSchedule( level, m_spice_tasks, " SPICE::schedulePitchBlend" );

  const MaterialSubset* ice_mss = ice_matls->getUnion();

  Task* t = scinew Task("SPICE::pitchBlend", this, &SPICE::pitchBlend);

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
        const VarLabel* adv0 = pitchTvarAdv0Label(idx);

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
Task:     SPICE::schedulePitchResidual--
Purpose:  PITCH outer iteration k (sub-graph only, scheduled after
          conservedtoPrimitive_Vars): residual = max over interior cells
          of max(|rho_new-rho_old|/rho_new, |T_new-T_old|/T_new), used for
          the optional <outerTolerance> early exit.
_____________________________________________________________________*/
void SPICE::schedulePitchResidual(SchedulerP        & sched,
                                 const LevelP      & level,
                                 const MaterialSet * ice_matls)
{
  printSchedule( level, m_spice_tasks, " SPICE::schedulePitchResidual" );

  Task* t = scinew Task("SPICE::pitchResidual", this, &SPICE::pitchResidual);

  t->requiresVar( Task::OldDW, lb->rho_CCLabel,  m_gn,0 );
  t->requiresVar( Task::OldDW, lb->temp_CCLabel, m_gn,0 );
  t->requiresVar( Task::NewDW, lb->rho_CCLabel,  m_gn,0 );
  t->requiresVar( Task::NewDW, lb->temp_CCLabel, m_gn,0 );

  if( d_ice->d_with_mpm ){
    // solid cells are skipped in the residual
    t->requiresVar( Task::ParentNewDW, lb->vol_frac_CCLabel, m_gn,0 );
  }

  t->computesVar( Slb->pitchResidualLabel );

  sched->addTask(t, level->eachPatch(), ice_matls);
}

/* _____________________________________________________________________
Task:     SPICE::schedulePitchOuterLoop--
Purpose:  For nOuterCorrectors > 1, re-run the fluid chain in a
          sub-scheduler for outer iterations k=1..nOuterCorrectors-1,
          seeded from the main graph's own k=0 primitives, and copy the
          final iterate's primitives back.  Scheduled in
          scheduleTimeAdvance immediately after
          scheduleConservedtoPrimitive_Vars.
_____________________________________________________________________*/
void SPICE::schedulePitchOuterLoop( SchedulerP & sched, const LevelP & level )
{
  if( d_pitchOuter < 2 ){
    return;
  }

  // compiling the outer taskgraph means the inner one must be recompiled too
  d_recompilePitchSched = true;

  printSchedule( level, m_spice_tasks, " SPICE::schedulePitchOuterLoop" );

  const MaterialSet* ice_matls = m_materialManager->allMaterials( "ICE" );
  const MaterialSet* all_matls = m_materialManager->allMaterials();
  const MaterialSubset* ice_mss = ice_matls->getUnion();

  Task* t = scinew Task("SPICE::pitchOuterLoop", this, &SPICE::pitchOuterLoop, level);
  t->hasSubScheduler();

  t->requiresVar( Task::OldDW, lb->timeStepLabel );
  t->requiresVar( Task::OldDW, lb->simulationTimeLabel );
  t->requiresVar( Task::OldDW, lb->delTLabel, level.get_rep() );

  // u^n primitives, read inside the sub-graph via ParentOldDW
  t->requiresVar( Task::OldDW, lb->rho_CCLabel,    ice_mss, m_gn,0 );
  t->requiresVar( Task::OldDW, lb->vel_CCLabel,    ice_mss, m_gn,0 );
  t->requiresVar( Task::OldDW, lb->temp_CCLabel,   ice_mss, m_gn,0 );
  t->requiresVar( Task::OldDW, lb->sp_vol_CCLabel, ice_mss, m_gn,0 );
  if( d_ice->d_with_mpm ){
    // time-n pressure for the solid-cell restart (pitchRestartSolidCells)
    t->requiresVar( Task::OldDW, lb->press_CCLabel, d_ice->d_press_matl, m_gn,0 );
  }

  // X_adv0 (k=0 captured advected state), read inside the sub-graph via
  // ParentNewDW
  t->requiresVar( Task::NewDW, Slb->mass_adv0Label,   ice_mss, m_gn,0 );
  t->requiresVar( Task::NewDW, Slb->mom_adv0Label,    ice_mss, m_gn,0 );
  t->requiresVar( Task::NewDW, Slb->eng_adv0Label,    ice_mss, m_gn,0 );
  t->requiresVar( Task::NewDW, Slb->sp_vol_adv0Label, ice_mss, m_gn,0 );

  // frozen model sources (copied every outer iteration by
  // SPICE::pitchCopyModelSources, via ParentNewDW) -- require them here too
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
  t->modifiesVar( lb->rho_CCLabel,    ice_mss );
  t->modifiesVar( lb->vel_CCLabel,    ice_mss );
  t->modifiesVar( lb->temp_CCLabel,   ice_mss );
  t->modifiesVar( lb->sp_vol_CCLabel, ice_mss );
  t->modifiesVar( lb->machLabel,      ice_mss );

  Task::MaterialDomainSpec oims = Task::OutOfDomain;
  t->modifiesVar( lb->press_CCLabel, d_ice->d_press_matl, oims );

  int idx = 0;
  for( auto& model : d_ice->d_models ){
    FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( model );
    if( fb_model ){
      for( auto& tvar : fb_model->d_transVars ){
        const VarLabel* adv0 = pitchTvarAdv0Label(idx);

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

  // FSPICE: the frozen-solid fields the sub-graph reads via ParentNewDW
  if( d_pitchRequiresHook ){
    d_pitchRequiresHook( t );
  }

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
    if( d_ktInterfaceDA ){
      t->requiresVar( Task::NewDW, Slb->gasFaceFrac_FCXLabel, m_gac,2 );
      t->requiresVar( Task::NewDW, Slb->gasFaceFrac_FCYLabel, m_gac,2 );
      t->requiresVar( Task::NewDW, Slb->gasFaceFrac_FCZLabel, m_gac,2 );
    }
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

      // FSPICE only: gas volume fraction in, open-face weights out
      constCCVariable<double> vol_frac;
      SFCXVariable<double>    openX;
      SFCYVariable<double>    openY;
      SFCZVariable<double>    openZ;
      if( d_ice->d_with_mpm ){
        new_dw->get( vol_frac, lb->vol_frac_CCLabel, indx,patch,m_gac,1 );
        new_dw->allocateAndPut( openX, Slb->ktOpen_FCXLabel, indx,patch );
        new_dw->allocateAndPut( openY, Slb->ktOpen_FCYLabel, indx,patch );
        new_dw->allocateAndPut( openZ, Slb->ktOpen_FCZLabel, indx,patch );
      }

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
      if( d_ice->d_with_mpm ){
        openX.initialize( 1.0, lowIndex, patch->getExtraSFCXHighIndex() );
        openY.initialize( 1.0, lowIndex, patch->getExtraSFCYHighIndex() );
        openZ.initialize( 1.0, lowIndex, patch->getExtraSFCZHighIndex() );
      }

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

          if( d_ice->d_with_mpm ){
            // FSPICE: KT acts only between open-gas cells.  w = 1 when the smaller
            // gas vol_frac of the two cells is >= 1-s, 0 when <= 1-10s, linear in
            // between (s = solidCellFrac).  w = 0 turns this face into the stock ICE
            // upwind flux (kappa = 0 => phiOwn = u_f*rhoUp, phiNei = 0, ktUvol = u_f);
            // without it KT's aSf term moves trace gas in and out of the solid.
            const double s    = d_ktSolidCellFrac;
            const double fMin = std::min( vol_frac[P], vol_frac[N] );
            double w = ( fMin - (1.0 - 10.0*s) )/( 9.0*s );
            w = std::min( 1.0, std::max( 0.0, w ) );
            kappaVal *= w;
            if( dir == 0 ){ openX[N] = w; } else if( dir == 1 ){ openY[N] = w; } else { openZ[N] = w; }
          }

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
                               DataWarehouse         * old_dw,
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

      // must match the ghost counts in scheduleComputeCentralFluxes (gac 2,
      // not 1: an SFC fetch with gac 1 can miss the neighbor patch's face)
      const Ghost::GhostType gFC = d_ktInterfaceDA ? m_gac : m_gn;
      const int              nFC = d_ktInterfaceDA ? 2 : 0;
      new_dw->get( uvel_FC,   lb->uvel_FCMELabel,      indx,patch,gFC,nFC );
      new_dw->get( vvel_FC,   lb->vvel_FCMELabel,      indx,patch,gFC,nFC );
      new_dw->get( wvel_FC,   lb->wvel_FCMELabel,      indx,patch,gFC,nFC );
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

      //__________________________________
      //  FSPICE <interfaceTransport> donorAcceptor inputs/outputs
      constSFCXVariable<double> openX;
      constSFCYVariable<double> openY;
      constSFCZVariable<double> openZ;
      constCCVariable<double>   volFrac;        // gas (the single ICE matl), from EqPress
      constCCVariable<double>   solidP;         // solid fraction from the particles (sharp)
      constCCVariable<double>   interiorP;      // 1 inside the solid (all 8 nodes carry MPM mass)
      SFCXVariable<double>      gasFracX;
      SFCYVariable<double>      gasFracY;
      SFCZVariable<double>      gasFracZ;
      double delT = 0.0;
      Vector dx   = patch->dCell();
      const double cellVol = dx.x()*dx.y()*dx.z();
      const double area[3] = { dx.y()*dx.z(), dx.x()*dx.z(), dx.x()*dx.y() };

      if( d_ktInterfaceDA ){
        delt_vartype delT_var;
        old_dw->get( delT_var, lb->delTLabel, level );
        delT = delT_var;
        new_dw->get( openX,   Slb->ktOpen_FCXLabel,  indx,patch,m_gn,0 );
        new_dw->get( openY,   Slb->ktOpen_FCYLabel,  indx,patch,m_gn,0 );
        new_dw->get( openZ,   Slb->ktOpen_FCZLabel,  indx,patch,m_gn,0 );
        new_dw->get( volFrac, lb->vol_frac_CCLabel, indx,patch,m_gac,2 );
        new_dw->get( solidP,    Slb->solidVolFracP_CCLabel,  indx,patch,m_gac,2 );
        new_dw->get( interiorP, Slb->solidInteriorP_CCLabel, indx,patch,m_gac,2 );
        new_dw->allocateAndPut( gasFracX, Slb->gasFaceFrac_FCXLabel, indx,patch );
        new_dw->allocateAndPut( gasFracY, Slb->gasFaceFrac_FCYLabel, indx,patch );
        new_dw->allocateAndPut( gasFracZ, Slb->gasFaceFrac_FCZLabel, indx,patch );
        gasFracX.initialize( -1.0, lowIndex, patch->getExtraSFCXHighIndex() );
        gasFracY.initialize( -1.0, lowIndex, patch->getExtraSFCYHighIndex() );
        gasFracZ.initialize( -1.0, lowIndex, patch->getExtraSFCZHighIndex() );
      }

      const IntVector unit[3] = { IntVector(1,0,0), IntVector(0,1,0), IntVector(0,0,1) };

      //  face velocity, 0 outside the fetched window (a donor in the
      //  extra-cell layer of a non-periodic domain face has no outer face)
      auto faceVel = [&]( int dir, const IntVector & f ) -> double {
        auto inside = [&]( const IntVector & l, const IntVector & h ){
          return f.x() >= l.x() && f.y() >= l.y() && f.z() >= l.z() &&
                 f.x() <  h.x() && f.y() <  h.y() && f.z() <  h.z();
        };
        if( dir == 0 ){ return inside( uvel_FC.getLowIndex(), uvel_FC.getHighIndex() ) ? uvel_FC[f] : 0.0; }
        if( dir == 1 ){ return inside( vvel_FC.getLowIndex(), vvel_FC.getHighIndex() ) ? vvel_FC[f] : 0.0; }
        return inside( wvel_FC.getLowIndex(), wvel_FC.getHighIndex() ) ? wvel_FC[f] : 0.0;
      };

      //  gas outflux volume of cell c over the step (all 6 faces)
      auto outfluxVol = [&]( const IntVector & c ) -> double {
        double v = 0.0;
        for( int k = 0; k < 3; k++ ){
          v += area[k]*( std::max( 0.0, -faceVel( k, c ) ) + std::max( 0.0, faceVel( k, c + unit[k] ) ) );
        }
        return delT*v;
      };

      //  gas inflow volume of cell c over the step (all 6 faces)
      auto influxVol = [&]( const IntVector & c ) -> double {
        double v = 0.0;
        for( int k = 0; k < 3; k++ ){
          v += area[k]*( std::max( 0.0, faceVel( k, c ) ) + std::max( 0.0, -faceVel( k, c + unit[k] ) ) );
        }
        return delT*v;
      };

      //  gradient of the particle solid fraction at c (points into the solid);
      //  one sided / zero where a neighbor is outside a non-periodic domain
      auto gradSolid = [&]( const IntVector & c ) -> Vector {
        Vector g( 0.0, 0.0, 0.0 );
        for( int k = 0; k < 3; k++ ){
          IntVector pl = c + unit[k];
          IntVector mi = c - unit[k];
          if( periodic[k] == 0 ){
            if( pl[k] >= hi[k] ){ pl = c; }
            if( mi[k] <  lo[k] ){ mi = c; }
          }
          const int steps = pl[k] - mi[k];
          if( steps > 0 ){
            g[k] = ( solidP[pl] - solidP[mi] )/( steps*dx[k] );
          }
        }
        return g;
      };

      //__________________________________
      //  No-penetration transport at a gas/solid face.  A material interface
      //  moves with the material, so no gas crosses it: the volume swept
      //  through a face in delT is filled by the material lying next to that
      //  face, not by the donor's average composition.  Where the solid lies
      //  comes from solidP (FSPICE::computeSolidVolFracP: the part of the cell
      //  where the nodal solid volume fraction exceeds 1/2), not from EqPress's
      //  vol_frac, which inherits the node-averaged cell mass and smears a
      //  sharp surface over about two cells.
      //   1) Geometry, Hirt & Nichols (1981) donor-acceptor with the particle
      //      solid fraction s:  X = acceptor when the interface lies across
      //      the face (dominant component of grad s at the donor), else donor;
      //        V_g = (1-s_X) V_f + max( s_X V_f - s_D V, 0 )
      //      Gas enters a solid acceptor only once the donor's solid is used
      //      up; trace gas in a solid donor leaves toward the gas first; two
      //      gas cells reduce exactly to upwind.
      //   2) Acceptor: a cell its particles cover (s_A >= 1 - tilingTol)
      //      takes no gas, so a wall that does not move stays sealed whatever
      //      the face velocity (a pressure jump at the wall face otherwise
      //      seeps gas into its first cell).  A solid-filled cell the surface
      //      is receding from (EqPress gas vol_frac below traceVolFrac) takes
      //      gas only from a donor without particles (s_D < 1 - covered) and
      //      only into the volume its particles have left, (1 - s_A - f_g,A) V,
      //      shared among its inflow faces, so gas follows the surface as fast
      //      as the particles leave.  The donor test keeps any spurious room
      //      inside the solid from passing gas from cell to cell into the
      //      wall.  A cell with a real gas pocket stays compressible (no bound).
      //      tilingTol = 1 - d_ktCoveredSolid absorbs noise in s (found with
      //      the first, particle-box version of s, which opened 3-40% gaps
      //      inside a strained wall).  Cost: gas follows a receding surface
      //      once it has moved tilingTol of a cell.
      //      Each test is a linear ramp, not an on/off switch: covered over
      //      s_A = covered -+ rampHalf, particle-free donor over s_D =
      //      tilingTol +- rampHalf, trace acceptor over f_g,A = traceVolFrac
      //      .. 10*traceVolFrac, and the across/along choice in 1) over
      //      gDir/(gDir+gOth) = 0.4 .. 0.6.  With switches, a round-off
      //      difference flipped a cell the receding surface uncovers one step
      //      early: a 3D periodic run left its 1D twin by 4e-2 in gas vol_frac
      //      (verification C2, 2026-10-28).
      //   0) Trace gas does not move between two trace-gas cells inside the
      //      solid (all 8 nodes carry MPM mass): it is a placeholder.
      //      Gas left in a cell the solid moves back over still leaves toward
      //      a neighbor that holds real gas.
      //   3) Donor: at most half its gas leaves per step (all faces), and
      //      never below the material's tiny_rho: shuttled back and forth, a
      //      trace donor was emptied to 1e-115 kg/m^3 (T2t), and a round-off
      //      negative sp_vol_L then stalled the run in recomputes.
      //  The gas mass moved is that share of the donor's gas:
      //    phi = u_f * (rho_CC/f_g)_D * V_g/V_f.   Returns V_g/V_f.
      const double traceVolFrac = 1.0e-3;     // as in LODI2.cc, gasCombustion, FSPICE EqPress
      const double drainMax     = 0.5;
      const double tinyRho      = ice_matl->getTinyRho();
      const double rampHalf     = 0.05;       // half width of the solid-fraction ramps

      //  0 at x <= x0, 1 at x >= x1 (x1 < x0 for a falling ramp), linear between
      auto ramp = []( double x, double x0, double x1 ) -> double {
        return std::min( 1.0, std::max( 0.0, ( x - x0 )/( x1 - x0 ) ) );
      };

      auto donorAcceptor = [&]( const IntVector & P, const IntVector & N, int dir,
                                double u_f, double & phi ) -> double {
        const IntVector D = ( u_f >= 0.0 ) ? P : N;
        const IntVector A = ( u_f >= 0.0 ) ? N : P;
        const double fgD = std::min( 1.0, std::max( 0.0, volFrac[D] ) );
        const double fgA = std::min( 1.0, std::max( 0.0, volFrac[A] ) );
        const double sD  = solidP[D];
        const double sA  = solidP[A];
        const double Vf  = std::fabs(u_f)*delT*area[dir];

        if( Vf <= 0.0 ){
          phi = 0.0;
          return fgD;
        }

        // 0) placeholder gas inside the solid (weights as in FSPICE EqPress)
        auto placeW = [&]( double fg, const IntVector & c ) -> double {
          return ( interiorP[c] > 0.5 ) ? ramp( fg, 10.0*traceVolFrac, traceVolFrac ) : 0.0;
        };
        const double wStay = placeW( fgD, D )*placeW( fgA, A );
        if( wStay >= 1.0 ){
          phi = 0.0;
          return 0.0;
        }

        // 1) geometry
        const Vector g    = gradSolid( D );
        const double gDir = std::fabs( g[dir] );
        const double gOth = std::max( std::fabs( g[(dir+1)%3] ), std::fabs( g[(dir+2)%3] ) );
        const double wAcross = ( gDir > 0.0 ) ? ramp( gDir/( gDir + gOth ), 0.4, 0.6 ) : 0.0;
        const double sX   = wAcross*sA + ( 1.0 - wAcross )*sD;

        double Vg = (1.0 - sX)*Vf + std::max( sX*Vf - sD*cellVol, 0.0 );

        // 2) acceptor: covered, or solid-filled with room, or open
        const double tilingTol = 1.0 - d_ktCoveredSolid;
        const double wOpen  = ramp( sA, d_ktCoveredSolid + rampHalf, d_ktCoveredSolid - rampHalf );
        const double wGasD  = ramp( sD, tilingTol + rampHalf, tilingTol - rampHalf );
        const double wReal  = ramp( fgA, traceVolFrac, 10.0*traceVolFrac );

        const double room    = wGasD*std::max( 0.0, (1.0 - sA - fgA)*cellVol );
        const double inShare = Vf/std::max( influxVol( A ), Vf );
        const double VgRoom  = std::min( Vg, room*inShare );
        Vg = wOpen*( wReal*Vg + ( 1.0 - wReal )*VgRoom );

        // 3) the donor's gas
        const double outShare = Vf/std::max( outfluxVol( D ), Vf );
        const double aboveFloor = std::max( 0.0, 1.0 - tinyRho/std::max( rho_CC[D], tiny ) );
        Vg = std::min( Vg, drainMax*aboveFloor*fgD*cellVol*outShare );
        Vg = std::max( 0.0, std::min( Vg, Vf ) )*( 1.0 - wStay );

        const double gasFrac = Vg/Vf;
        phi = u_f*( rho_CC[D]/std::max( fgD, tiny ) )*gasFrac;
        return gasFrac;
      };

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

          //  Names in reactingPimpleCentralFoam:
          //    aphivOwn/aphivNei   <-> aphiv_own/aphiv_nei
          //    rhoOwnEff/rhoNeiEff <-> kappa blend of own/nei and upwind density
          //    phiOwn/phiNei shift <-> updateKappa.H
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

          //  FSPICE no-penetration transport on faces that touch a solid cell
          //  (ktOpen < 1, where kappa is already blended toward 0).  The gas
          //  volume flux the pressure equation sees stays u_f; only the gas's
          //  share of it (gasFaceFrac) changes.
          if( d_ktInterfaceDA ){
            const double open = (dir==0)? openX[N] : (dir==1)? openY[N] : openZ[N];
            if( open < 1.0 ){
              double phiDA = 0.0;
              const double gasFrac = donorAcceptor( P, N, dir, u_f, phiDA );
              phiOwn = phiDA;
              phiNei = 0.0;
              uVol   = u_f;
              if( dir == 0 ){ gasFracX[N] = gasFrac; } else if( dir == 1 ){ gasFracY[N] = gasFrac; } else { gasFracZ[N] = gasFrac; }
            }
          }

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
      // weighted by alphaOwn, as in reactingPimpleCentralFoam
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

      if( kt->open ){
        // FSPICE: blend toward the stock density-weighted average next to solid cells
        const double w = (*kt->open)[R];
        if( w < 1.0 ){
          double term1_ICE = ( rho_CC[L] * vel_CC[L][dir] +
                               rho_CC[R] * vel_CC[R][dir] )/rho_FC;
          term1 = w*term1 + (1.0 - w)*term1_ICE;
        }
      }
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
      constSFCXVariable<double> ktOpenX;
      constSFCYVariable<double> ktOpenY;
      constSFCZVariable<double> ktOpenZ;

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
          if( d_ice->d_with_mpm ){
            new_dw->get( ktOpenX, Slb->ktOpen_FCXLabel, indx,patch,m_gn,0 );
            new_dw->get( ktOpenY, Slb->ktOpen_FCYLabel, indx,patch,m_gn,0 );
            new_dw->get( ktOpenZ, Slb->ktOpen_FCZLabel, indx,patch,m_gn,0 );
          }
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
      KTFaceVelInfo<constSFCXVariable<double> > ktInfoX{ &ktAlphaOwnX, ktLo, ktHi, ktPeriodic,
                                                         d_ice->d_with_mpm ? &ktOpenX : nullptr };
      KTFaceVelInfo<constSFCYVariable<double> > ktInfoY{ &ktAlphaOwnY, ktLo, ktHi, ktPeriodic,
                                                         d_ice->d_with_mpm ? &ktOpenY : nullptr };
      KTFaceVelInfo<constSFCZVariable<double> > ktInfoZ{ &ktAlphaOwnZ, ktLo, ktHi, ktPeriodic,
                                                         d_ice->d_with_mpm ? &ktOpenZ : nullptr };

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

    // FSPICE + CentralUpwind: MPM materials' vol_frac keeps the stock
    // SecondOrder advector (KT applies to the ICE material only)
    Advector* mpmAdvector = nullptr;
    if( d_ice->d_with_mpm && d_mpmAdvector ){
      mpmAdvector = d_mpmAdvector->clone(new_dw,patch,isRegridTimeStep());
    }

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

    //  the mixture face volume flux this task's dilatation (term2) is built
    //  from, for the pressure work (PressureWorkFix.md)
    SFCXVariable<double> volFluxX;
    SFCYVariable<double> volFluxY;
    SFCZVariable<double> volFluxZ;
    new_dw->allocateAndPut( volFluxX, Slb->volFlux_FCXLabel, 0, patch );
    new_dw->allocateAndPut( volFluxY, Slb->volFlux_FCYLabel, 0, patch );
    new_dw->allocateAndPut( volFluxZ, Slb->volFlux_FCZLabel, 0, patch );
    volFluxX.initialize( 0.0 );
    volFluxY.initialize( 0.0 );
    volFluxZ.initialize( 0.0 );

    new_dw->allocateTemporary( q_advected, patch );
    new_dw->allocateTemporary( term1,      patch );

    term1.initialize(0.);
    term2.initialize(0.);
    sum_rho_CC.initialize(0.0);
    delP_Dilatate.initialize(0.0);
    delP_MassX.initialize(0.0);

    //__________________________________
    //  FSPICE <interfaceTransport> donorAcceptor: at gas/solid faces the gas
    //  carries gasFaceFrac of the face volume flux and the MPM matls carry the
    //  rest (split by their donor-cell fractions), so the face partition is
    //  complementary and the same one the gas mass flux used
    //  (SPICE::computeCentralFluxes).
    constSFCXVariable<double> gasFracX;
    constSFCYVariable<double> gasFracY;
    constSFCZVariable<double> gasFracZ;
    constCCVariable<double>   iceVolFrac;
    if( d_ktInterfaceDA ){
      const int iceIndx = m_materialManager->getMaterial( "ICE", 0 )->getDWIndex();
      new_dw->get( gasFracX,   Slb->gasFaceFrac_FCXLabel, iceIndx,patch,m_gac,2 );
      new_dw->get( gasFracY,   Slb->gasFaceFrac_FCYLabel, iceIndx,patch,m_gac,2 );
      new_dw->get( gasFracZ,   Slb->gasFaceFrac_FCZLabel, iceIndx,patch,m_gac,2 );
      new_dw->get( iceVolFrac, lb->vol_frac_CCLabel,     iceIndx,patch,m_gac,2 );
    }

    //  per-material face override: the gas share for the ICE matl, this
    //  matl's part of the rest for an MPM matl, -1 off the interface faces
    auto faceOverride = [&]( auto & ovr, const auto & gasFrac, const auto & vel_FC,
                             const constCCVariable<double> & matlVolFrac,
                             bool isICE, const IntVector & offset ){
      const IntVector l = Max( ovr.getLowIndex(),  gasFrac.getLowIndex() );
      const IntVector h = Min( ovr.getHighIndex(), gasFrac.getHighIndex() );
      ovr.initialize( -1.0 );
      for( CellIterator it( l, h ); !it.done(); it++ ){
        const IntVector f = *it;
        const double g = gasFrac[f];
        if( g < 0.0 ){
          continue;
        }
        if( isICE ){
          ovr[f] = g;
          continue;
        }
        const IntVector D = ( vel_FC[f] >= 0.0 ) ? f - offset : f;      // this matl's donor
        const double solid = 1.0 - iceVolFrac[D];
        const double part  = ( solid > 1.0e-100 ) ? matlVolFrac[D]/solid : 0.0;
        ovr[f] = std::min( 1.0, std::max( 0.0, (1.0 - g)*part ) );
      }
    };

    for(unsigned int m = 0; m < numMatls; m++) {
      Material* matl = m_materialManager->getMaterial( m );
      int indx = matl->getDWIndex();
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      Advector* adv = ( !ice_matl && mpmAdvector ) ? mpmAdvector : advector;

      constCCVariable<double> speedSound;
      constCCVariable<double> vol_frac;
      constCCVariable<double> rho_CC;
      constSFCXVariable<double> uvel_FC;
      constSFCYVariable<double> vvel_FC;
      constSFCZVariable<double> wvel_FC;

      // MPM materials: no KT fluxes, keep uvel_FCME
      if( d_useCentralUpwind && d_ktPressureFlux && ice_matl ){
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

      adv->inFluxOutFluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch,indx,
                               bulletProof_test, new_dw, varBasket);

      SFCXVariable<double> ovrX;
      SFCYVariable<double> ovrY;
      SFCZVariable<double> ovrZ;
      if( d_ktInterfaceDA ){
        new_dw->allocateTemporary( ovrX, patch, m_gac, 2 );
        new_dw->allocateTemporary( ovrY, patch, m_gac, 2 );
        new_dw->allocateTemporary( ovrZ, patch, m_gac, 2 );
        faceOverride( ovrX, gasFracX, uvel_FC, vol_frac, ice_matl != nullptr, IntVector(1,0,0) );
        faceOverride( ovrY, gasFracY, vvel_FC, vol_frac, ice_matl != nullptr, IntVector(0,1,0) );
        faceOverride( ovrZ, gasFracZ, wvel_FC, vol_frac, ice_matl != nullptr, IntVector(0,0,1) );
        setFaceFractions( adv, &ovrX, &ovrY, &ovrZ );
      }
      //__________________________________
      //   advect vol_frac
      varBasket->doRefluxing = false;  // don't need to reflux here
      adv->advectQ(vol_frac, patch, q_advected, varBasket,
                   vol_fracX_FC, vol_fracY_FC,  vol_fracZ_FC, new_dw);

      if( d_ktInterfaceDA ){
        setFaceFractions( adv, nullptr, nullptr, nullptr );   // ovr* go out of scope
      }

      //__________________________________
      //  This matl's face volume flux, exactly what advectQ just put in
      //  q_advected (PressureWorkFix.md 4.8): the override where one was set,
      //  0 where the donor cell holds none of this matl (save_q_FC stores q_CC
      //  there although the flux was 0), else the face value
      auto addVolFlux = [&]( auto & volFlux, const auto & velFC, const auto & volFracFC,
                             const auto & ovr, bool useOvr, CellIterator it,
                             const IntVector & offset ){
        for( ; !it.done(); it++ ){
          const IntVector f     = *it;
          const IntVector donor = ( velFC[f] >= 0.0 ) ? f - offset : f;
          double frac = volFracFC[f];
          if( useOvr && ovr[f] >= 0.0 ){
            frac = ovr[f];
          }
          else if( vol_frac[donor] == 0.0 ){
            frac = 0.0;
          }
          volFlux[f] += frac * velFC[f];
        }
      };
      //  faces this patch owns, as CentralUpwindAdvector saves qFace on them
      //  (includes the domain-boundary + face, Patch::getSFCXHighIndex)
      addVolFlux( volFluxX, uvel_FC, vol_fracX_FC, ovrX, d_ktInterfaceDA,
                  patch->getSFCXIterator(), IntVector(1,0,0) );
      addVolFlux( volFluxY, vvel_FC, vol_fracY_FC, ovrY, d_ktInterfaceDA,
                  patch->getSFCYIterator(), IntVector(0,1,0) );
      addVolFlux( volFluxZ, wvel_FC, vol_fracZ_FC, ovrZ, d_ktInterfaceDA,
                  patch->getSFCZIterator(), IntVector(0,0,1) );

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
    delete mpmAdvector;

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

      if( kt->open ){
        // FSPICE: blend toward the stock sum_rho-weighted face pressure next to solid cells
        const double w = (*kt->open)[R];
        if( w < 1.0 ){
          double pICE = ( press_CC[R] * sum_rho[L] + press_CC[L] * sum_rho[R] )/
                        ( sum_rho[R] + sum_rho[L] );
          press_FC[R] = w*press_FC[R] + (1.0 - w)*pICE;
        }
      }
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
    constSFCXVariable<double> ktOpenX;
    constSFCYVariable<double> ktOpenY;
    constSFCZVariable<double> ktOpenZ;
    IntVector ktLo, ktHi, ktPeriodic;

    if( useKT ){
      int iceIndx = m_materialManager->getMaterial( "ICE", 0 )->getDWIndex();
      new_dw->get( ktAlphaOwnX, Slb->ktAlphaOwn_FCXLabel, iceIndx,patch,m_gn,0 );
      new_dw->get( ktAlphaOwnY, Slb->ktAlphaOwn_FCYLabel, iceIndx,patch,m_gn,0 );
      new_dw->get( ktAlphaOwnZ, Slb->ktAlphaOwn_FCZLabel, iceIndx,patch,m_gn,0 );
      if( d_ice->d_with_mpm ){
        new_dw->get( ktOpenX, Slb->ktOpen_FCXLabel, iceIndx,patch,m_gn,0 );
        new_dw->get( ktOpenY, Slb->ktOpen_FCYLabel, iceIndx,patch,m_gn,0 );
        new_dw->get( ktOpenZ, Slb->ktOpen_FCZLabel, iceIndx,patch,m_gn,0 );
      }

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

    KTFacePressInfo<constSFCXVariable<double> > ktInfoX{ &ktAlphaOwnX, ktLo, ktHi, ktPeriodic,
                                                         d_ice->d_with_mpm ? &ktOpenX : nullptr };
    KTFacePressInfo<constSFCYVariable<double> > ktInfoY{ &ktAlphaOwnY, ktLo, ktHi, ktPeriodic,
                                                         d_ice->d_with_mpm ? &ktOpenY : nullptr };
    KTFacePressInfo<constSFCZVariable<double> > ktInfoZ{ &ktAlphaOwnZ, ktLo, ktHi, ktPeriodic,
                                                         d_ice->d_with_mpm ? &ktOpenZ : nullptr };

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
 Task:      SPICE::accumulateMomentumSourceSinks
 Purpose:   accumulate all of the sources/sinks of momentum
 _____________________________________________________________________  */
void SPICE::accumulateMomentumSourceSinks(const ProcessorGroup  *,
                                        const PatchSubset     * patches,
                                        const MaterialSubset  * /*matls*/,
                                        DataWarehouse         * old_dw,
                                        DataWarehouse         * new_dw)
{
  const Level* level = getLevel(patches);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, m_spice_tasks, "SPICE::accumulateMomentumSourceSinks" );

    IntVector right, left, top, bottom, front, back;

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, level);

    Vector dx      = patch->dCell();
    double vol     = dx.x() * dx.y() * dx.z();
    Vector gravity = d_ice->getGravity();

    double areaX = dx.y() * dx.z();
    double areaY = dx.x() * dx.z();
    double areaZ = dx.x() * dx.y();

    constSFCXVariable<double> pressX_FC;
    constSFCYVariable<double> pressY_FC;
    constSFCZVariable<double> pressZ_FC;


    new_dw->get( pressX_FC,lb->pressX_FCLabel, 0, patch,m_gac,1 );
    new_dw->get( pressY_FC,lb->pressY_FCLabel, 0, patch,m_gac,1 );
    new_dw->get( pressZ_FC,lb->pressZ_FCLabel, 0, patch,m_gac,1 );

    unsigned int numMatls  = m_materialManager->getNumMatls();

    //__________________________________
    //  <interfaceShear> model: the wall shear between the gas and the solids
    CCVariable<Vector> shearGas;
    std::vector< CCVariable<Vector> > shearSolid;
    std::vector<int> mpmDWIndex;
    const bool shear = d_ice->d_with_mpm && d_ktInterfaceShear;
    const int  gasIndx = m_materialManager->getMaterial( "ICE", 0 )->getDWIndex();
    if( shear ){
      const unsigned int numMPM = m_materialManager->getNumMatls( "MPM" );
      shearSolid.resize( numMPM );
      for(unsigned int i = 0; i < numMPM; i++){
        mpmDWIndex.push_back( m_materialManager->getMaterial( "MPM", i )->getDWIndex() );
        new_dw->allocateTemporary( shearSolid[i], patch );
        shearSolid[i].initialize( Vector( 0.0, 0.0, 0.0 ) );
      }
      new_dw->allocateTemporary( shearGas, patch );
      shearGas.initialize( Vector( 0.0, 0.0, 0.0 ) );
      interfaceShear( patch, old_dw, new_dw, delT, shearGas, shearSolid, nullptr );
    }

    //__________________________________
    //  Matl loop
    for(unsigned int m = 0; m < numMatls; m++) {
      Material* matl        = m_materialManager->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      int indx = matl->getDWIndex();

      constCCVariable<double>  vol_frac;
      constCCVariable<double>  rho_CC;
      CCVariable<Vector>   mom_source;

      new_dw->get( vol_frac,  lb->vol_frac_CCLabel, indx,patch,m_gn,0 );
      new_dw->get( rho_CC,    lb->rho_CCLabel,      indx,patch,m_gn,0 );

      new_dw->allocateAndPut(mom_source,  lb->mom_source_CCLabel,  indx, patch);
      mom_source.initialize( Vector(0.,0.,0.) );

      //__________________________________
      //  accumulate sources MPM and ICE matls
      for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;

        right    = c + IntVector(1,0,0);    left     = c;
        top      = c + IntVector(0,1,0);    bottom   = c;
        front    = c + IntVector(0,0,1);    back     = c;

        double press_src_X = ( pressX_FC[right] - pressX_FC[left] )   * vol_frac[c];
        double press_src_Y = ( pressY_FC[top]   - pressY_FC[bottom] ) * vol_frac[c];
        double press_src_Z = ( pressZ_FC[front] - pressZ_FC[back] )   * vol_frac[c];

        mom_source[c].x( -press_src_X * areaX );
        mom_source[c].y( -press_src_Y * areaY );
        mom_source[c].z( -press_src_Z * areaZ );
      }

      //__________________________________
      //  Add user defined pressure gradient
      Vector evilNumV( d_ice->d_EVIL_NUM );

      if( d_ice->d_fixedPressGrad.length() != evilNumV.length() ) {

        Vector oneZero;
        oneZero.x(  ( d_ice->d_fixedPressGrad.x() == d_ice->d_EVIL_NUM) ? 1.0 : 0.0 );
        oneZero.y(  ( d_ice->d_fixedPressGrad.y() == d_ice->d_EVIL_NUM) ? 1.0 : 0.0 );
        oneZero.z(  ( d_ice->d_fixedPressGrad.z() == d_ice->d_EVIL_NUM) ? 1.0 : 0.0 );

        double src_X    = d_ice->d_fixedPressGrad.x() * dx.x() * areaX ;
        double src_Y    = d_ice->d_fixedPressGrad.y() * dx.y() * areaY ;
        double src_Z    = d_ice->d_fixedPressGrad.z() * dx.z() * areaZ ;
        Vector pressSrc = Vector( src_X, src_Y, src_Z );
        Vector one( 1.0, 1.0, 1.0 );

        for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
          IntVector c = *iter;

          Vector fixedPressSrc = pressSrc * vol_frac[c];
          mom_source[c] = mom_source[c] + (one - oneZero) * fixedPressSrc;
        }
      }

      //__________________________________
      //  ICE _matls:
      if(ice_matl){
        constCCVariable<Vector> viscous_src;
        new_dw->get( viscous_src, lb->viscous_src_CCLabel, indx, patch,m_gn,0 );

        for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
          IntVector c = *iter;
          double mass = rho_CC[c] * vol;

          mom_source[c] = (mom_source[c] + viscous_src[c] + mass * gravity );
        }
      }  //ice_matl

      //__________________________________
      //  All Matls
      for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        mom_source[c] *= delT;
      }

      //__________________________________
      //  <interfaceShear> model: the gas loses what the solids gain (already
      //  momentum over the step)
      if( shear ){
        const CCVariable<Vector>* src = ( indx == gasIndx ) ? &shearGas : nullptr;
        for( size_t i = 0; i < mpmDWIndex.size(); i++ ){
          if( mpmDWIndex[i] == indx ){
            src = &shearSolid[i];
          }
        }
        if( src ){
          for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
            const IntVector c = *iter;
            mom_source[c] += (*src)[c];
          }
        }
      }

    }  // matls loop
  }  //patches
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

    //__________________________________
    //  phi: the pressure equation's mixture face volume flux sum_m volFrac_FC*u_f
    //  [m/s] on both faces of every cell of the patch (PressureWorkFix.md 4.2)
    constSFCXVariable<double> phiX;
    constSFCYVariable<double> phiY;
    constSFCZVariable<double> phiZ;

    if( d_useCentralUpwind && !d_ice->d_impICE ){
      new_dw->get( phiX, Slb->volFlux_FCXLabel, 0, patch, m_gac,1 );
      new_dw->get( phiY, Slb->volFlux_FCYLabel, 0, patch, m_gac,1 );
      new_dw->get( phiZ, Slb->volFlux_FCZLabel, 0, patch, m_gac,1 );
    }
    else{
      //  ICE's pressure task or the implicit solve: no face override exists on
      //  these paths; same empty donor rule as
      //  SPICE::computeDelPressAndUpdatePressCC
      SFCXVariable<double> phiXsum;
      SFCYVariable<double> phiYsum;
      SFCZVariable<double> phiZsum;
      new_dw->allocateTemporary( phiXsum, patch, m_gac, 1 );
      new_dw->allocateTemporary( phiYsum, patch, m_gac, 1 );
      new_dw->allocateTemporary( phiZsum, patch, m_gac, 1 );
      phiXsum.initialize( 0.0 );
      phiYsum.initialize( 0.0 );
      phiZsum.initialize( 0.0 );
      const IntVector lo = patch->getCellLowIndex();
      const IntVector hi = patch->getCellHighIndex();

      for(unsigned int m = 0; m < numMatls; m++) {
        int indx = m_materialManager->getMaterial( m )->getDWIndex();
        constSFCXVariable<double> uFC, fracX;
        constSFCYVariable<double> vFC, fracY;
        constSFCZVariable<double> wFC, fracZ;
        constCCVariable<double>   volFracCC;
        new_dw->get( uFC,       lb->uvel_FCMELabel,    indx, patch, m_gac,1 );
        new_dw->get( vFC,       lb->vvel_FCMELabel,    indx, patch, m_gac,1 );
        new_dw->get( wFC,       lb->wvel_FCMELabel,    indx, patch, m_gac,1 );
        new_dw->get( fracX,     lb->vol_fracX_FCLabel, indx, patch, m_gac,1 );
        new_dw->get( fracY,     lb->vol_fracY_FCLabel, indx, patch, m_gac,1 );
        new_dw->get( fracZ,     lb->vol_fracZ_FCLabel, indx, patch, m_gac,1 );
        new_dw->get( volFracCC, lb->vol_frac_CCLabel,  indx, patch, m_gac,1 );

        auto addPhi = [&]( auto & phi, const auto & velFC, const auto & frac,
                           const IntVector & offset ){
          //  both faces of every cell: cell low .. cell high + offset
          for( CellIterator it( lo, hi + offset ); !it.done(); it++ ){
            const IntVector f     = *it;
            const IntVector donor = ( velFC[f] >= 0.0 ) ? f - offset : f;
            const double    theta = ( volFracCC[donor] == 0.0 ) ? 0.0 : frac[f];
            phi[f] += theta * velFC[f];
          }
        };
        addPhi( phiXsum, uFC, fracX, IntVector(1,0,0) );
        addPhi( phiYsum, vFC, fracY, IntVector(0,1,0) );
        addPhi( phiZsum, wFC, fracZ, IntVector(0,0,1) );
      }
      phiX = phiXsum;
      phiY = phiYsum;
      phiZ = phiZsum;
    }

    //  SCI_DEBUG SPICE_pressWork: phi must reproduce the dilatation the
    //  pressure equation used, (delT/vol)*sum_d (phi_hi - phi_lo)*area = term2
    if( m_spice_pressWork.active() && !d_ice->d_impICE ){
      constCCVariable<double> term2;
      new_dw->get( term2, lb->term2Label, 0, patch, m_gn,0 );
      double maxMiss  = 0.0;
      double maxTerm2 = 0.0;
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        const double dilatation = ( delT/vol ) *
            ( ( phiX[c + IntVector(1,0,0)] - phiX[c] ) * areaX
            + ( phiY[c + IntVector(0,1,0)] - phiY[c] ) * areaY
            + ( phiZ[c + IntVector(0,0,1)] - phiZ[c] ) * areaZ );
        maxMiss  = std::max( maxMiss,  std::fabs( dilatation - term2[c] ) );
        maxTerm2 = std::max( maxTerm2, std::fabs( term2[c] ) );
      }
      DOUTR( m_spice_pressWork, "SPICE_pressWork patch " << patch->getID()
             << " max|phi dilatation - term2| " << maxMiss << " max|term2| " << maxTerm2 );
    }

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
    //  over all matls; it matches mom_source term for term.  TE_flux is the flux
    //  of press_FC*phi, where phi = sum_m volFrac_FC*u_f is the mixture face
    //  volume flux the pressure equation built its dilatation from (stored by
    //  SPICE::computeDelPressAndUpdatePressCC, rebuilt above from the converged
    //  face fields for ICE's pressure task and the implicit solve).  One value
    //  per face, so the flux telescopes and IE + KE is conserved, and
    //      TE_flux - KE_work = press_CC*(volume change of the pressure equation)
    //                          + sum_faces (press_FC - press_CC)*(phi - u^{n+1/2})*A
    //  the second term a product of two small differences: the internal energy
    //  change implies the pressure the pressure equation solved for.
    //  (Until 2026-11 the flux carried W, a cross weighted average of the old
    //  time CELL CENTRED velocities.  Where div(W) differed from the face
    //  dilatation the fluid was compressed by one volume change and heated for
    //  another: an EOS pressure 3-5% off the solved one per implicit step
    //  (impHotBlob), and gas cooled at constant density wherever a cell
    //  velocity had no face counterpart.  PressureWorkFix.md.)
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
    //  literally 1.0 there.
    //
    //  M P M I C E  (d_ice->d_with_mpm)
    //  The same work face by face, relative to the cell's own mixture velocity
    //  W_c:  nu*press_FC*(phi_f - W_c)*A.  The two cells of a face together book
    //  press_FC*(W_n - W_c)*A whatever phi is; phi sets the split.
    //  Each of the points below is needed because a trace ICE matl inside a solid cell
    //  (vol_frac ~1e-12, inflated to ~1e-7 in a partly filled boundary cell)
    //  takes an f_theta share of 1e-3..1 of whatever work that cell is
    //  given, while holding ~1e-13 of the energy that work is measured
    //  against.  Any residual the operator leaves in a solid cell drives it
    //  negative (pistonValidation: Negative int_eng_L on the piston face).
    //
    //   1) W runs over ALL matls and is normalized by sum(vol_frac).  A solid
    //      occupies volume and moves, so it is part of the mixture volume
    //      flux; with W over ICE matls only, the ICE volume fraction dropping
    //      from ~1 to 1e-12 across the solid boundary reads as a jump in W.
    //      Normalizing removes the equilibration tolerance
    //      (sum(vol_frac) = 1 +- 1e-8), so a rigidly moving solid has
    //      div(W) = 0 to round off.
    //
    //   2) Each face's work, press_FC*(W_R - W_L)*A, is split between its two
    //      cells by where the face volume flux phi lies between W_L and W_R.
    //      At a gas/solid face whose flux follows the solid (what
    //      donorAcceptor builds) the solid cell books ~0 and the gas cell the
    //      whole face work, i.e. the work is done on the gas the solid pushes.
    //      (Until 2026-11 the split was by ICE mass, with W in place of phi.)
    //
    //   3) The u^{n+1/2} half step is taken on the mixture,
    //      U^{n+1/2}.sum(dp), and ICE is charged its mass share of it.  The
    //      momentum exchange moves the matls as one, so the |dp|^2 dt/(2 mass)
    //      a trace ICE matl would gain before the exchange relocks it to its
    //      host never materializes, and a solid's KE is MPM's business.
    //
    //  The MPM share of netPressWork (f_theta_MPM) is dropped below: MPM keeps
    //  its own energy books.
    //
    //  KNOWN LIMITATION: a phase that is present only in trace amounts and is
    //  far more compressible than its host (air suspended in water at
    //  vol_frac ~ 1e-12) gets an f_theta share in which its own vol_frac
    //  cancels, so kappa_m/sumKappa (~1e4 there) multiplies whatever residual
    //  this operator leaves (waterAirOscillator).  With phi from the pressure
    //  equation the leading residual, div(W) against the face dilatation, is
    //  gone; the (press_FC - press_CC)*(phi - u) term above remains.
    CCVariable<Vector> W;              // sum over matls of vol_frac * vel_CC
    CCVariable<double> KE_work;        // KE production by the pressure force
    CCVariable<double> netPressWork;   // mixture pressure work, before the f_theta split

    new_dw->allocateTemporary( W,            patch, m_gac, 1 );
    new_dw->allocateTemporary( KE_work,      patch );
    new_dw->allocateTemporary( netPressWork, patch );

    W.initialize( Vector(0.0,0.0,0.0) );
    KE_work.initialize( 0.0 );
    netPressWork.initialize( 0.0 );

    //  FSPICE only
    CCVariable<double> sum_volFrac;    // normalizes W                          1)
    CCVariable<double> sum_rho_ICE;    // ICE mass per volume, face split       2)
    CCVariable<Vector> sum_dp;         // mixture pressure force, half step     3)
    CCVariable<double> sum_mass;       // mixture mass,           half step     3)

    if( d_ice->d_with_mpm ){
      new_dw->allocateTemporary( sum_volFrac, patch, m_gac, 1 );
      new_dw->allocateTemporary( sum_rho_ICE, patch, m_gac, 1 );
      new_dw->allocateTemporary( sum_dp,      patch );
      new_dw->allocateTemporary( sum_mass,    patch );

      sum_volFrac.initialize( 0.0 );
      sum_rho_ICE.initialize( 0.0 );
      sum_dp.initialize( Vector(0.0,0.0,0.0) );
      sum_mass.initialize( 0.0 );
    }

    for(unsigned int m = 0; m < numMatls; m++) {
      Material* matl        = m_materialManager->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      int indx = matl->getDWIndex();

      constCCVariable<double> volFrac;
      constCCVariable<double> rho;
      constCCVariable<Vector> vel_CC;

      new_dw->get( volFrac, lb->vol_frac_CCLabel, indx,patch,m_gac,1 );
      new_dw->get( rho,     lb->rho_CCLabel,      indx,patch,m_gac,1 );

      if( ice_matl ){
        old_dw->get( vel_CC,  lb->vel_CCLabel,    indx,patch,m_gac,1 );
      } else {                  // MPM matl: time n CC velocity from interpolateNCToCC_0
        new_dw->get( vel_CC,  lb->vel_CCLabel,    indx,patch,m_gac,1 );
      }

      //  a gac,1 fill needs the matching ghost count on the iterator, otherwise
      //  the layer at inter-patch faces is skipped and W is left zero there
      for(CellIterator iter = patch->getExtraCellIterator(1); !iter.done(); iter++){
        IntVector c = *iter;
        W[c] += volFrac[c] * vel_CC[c];
      }

      if( d_ice->d_with_mpm ){
        for(CellIterator iter = patch->getExtraCellIterator(1); !iter.done(); iter++){
          IntVector c = *iter;
          sum_volFrac[c] += volFrac[c];
          if( ice_matl ){
            sum_rho_ICE[c] += rho[c];
          }
        }
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
        if( d_ice->d_with_mpm ){       // mixture half step, 3)
          sum_dp[c]   += Vector( dp_X, dp_Y, dp_Z );
          sum_mass[c] += rho[c] * vol;
        }
        else{
          double halfInvMass = 1.0/( 2.0 * rho[c] * vol );

          double uHalf = vel_CC[c].x() - dp_X * delT * halfInvMass;
          double vHalf = vel_CC[c].y() - dp_Y * delT * halfInvMass;
          double wHalf = vel_CC[c].z() - dp_Z * delT * halfInvMass;

          KE_work[c] += uHalf * dp_X + vHalf * dp_Y + wHalf * dp_Z;
        }
      }
    }

    if( d_ice->d_with_mpm ){
      //__________________________________
      //  FSPICE: face by face form of -delT*( div(W press) - W.grad(press_FC) ),
      //  see 1) - 3) above
      for(CellIterator iter = patch->getExtraCellIterator(1); !iter.done(); iter++){
        IntVector c = *iter;
        W[c] /= sum_volFrac[c];
      }

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c      = *iter;
        IntVector right  = c + IntVector(1,0,0);
        IntVector top    = c + IntVector(0,1,0);
        IntVector front  = c + IntVector(0,0,1);

        //  cell c's piece of TE_flux - KE_work from each face:
        //    nu * press_FC * ( phi_f - W_c ) * area,   nu = +1 high face, -1 low
        //  phi is the pressure equation's mixture face volume flux; the face
        //  total is press_FC*(W_n - W_c)*area, split by where phi lies between
        //  W_c and W_n (PressureWorkFix.md 4.5, 6)
        double TE_minus_KE =
            pressX_FC[right]*( phiX[right] - W[c].x() )*areaX
          - pressX_FC[c]    *( phiX[c]     - W[c].x() )*areaX
          + pressY_FC[top]  *( phiY[top]   - W[c].y() )*areaY
          - pressY_FC[c]    *( phiY[c]     - W[c].y() )*areaY
          + pressZ_FC[front]*( phiZ[front] - W[c].z() )*areaZ
          - pressZ_FC[c]    *( phiZ[c]     - W[c].z() )*areaZ;

        //  the u^{n+1/2} part of KE_work, taken on the mixture: ICE is charged
        //  its mass share of dt*|sum dp|^2/(2*mixture mass)
        double ICE_massFrac = sum_rho_ICE[c] * vol / sum_mass[c];
        double KE_halfStep  = ICE_massFrac * delT * sum_dp[c].length2()/( 2.0 * sum_mass[c] );

        netPressWork[c] = -delT * ( TE_minus_KE + KE_halfStep );
      }
    }

    else{
      //__________________________________
      //  mixture pressure work:  -delT*( div(press_FC*phi) - KE_work ), phi the
      //  pressure equation's face volume flux, so the internal energy change is
      //  -press * (the volume change the pressure equation used) to leading
      //  order, while the flux still telescopes (PressureWorkFix.md 4.4-4.5)
      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c     = *iter;
        IntVector right = c + IntVector(1,0,0);
        IntVector top   = c + IntVector(0,1,0);
        IntVector front = c + IntVector(0,0,1);

        double TE_flux =
            ( pressX_FC[right]*phiX[right] - pressX_FC[c]*phiX[c] ) * areaX
          + ( pressY_FC[top]  *phiY[top]   - pressY_FC[c]*phiY[c] ) * areaY
          + ( pressZ_FC[front]*phiZ[front] - pressZ_FC[c]*phiZ[c] ) * areaZ;

        netPressWork[c] = -delT * (TE_flux - KE_work[c]);
      }
    }

    //__________________________________
    //  <interfaceExchange> particles/conduction: placeholder trace gas books no
    //  pressure work.  Trace gas inside the solid (vol_frac below 1e-3 where
    //  all 8 nodes carry MPM mass; weighted 1 -> 0 up to 1e-2 as in FSPICE
    //  EqPress) deforms with its host, whose cell pressure (~1 MPa in a shocked
    //  wall) is ~150x its own: host work at that pressure heated it to 4000 K
    //  inside the T5s wall (heat exchange 0).  Its energy has no meaning, and
    //  its density is frozen in EqPress too.
    //  All other gas books its share of the conservative face form above.
    //  (Until 2026-10-28 gas next to a solid booked its 'realized' work
    //  vol*f*kappa*p*delP_Dilatate instead: the face form overheated the gas
    //  cell in front of a wall, whose node-smeared solid locked its gas to the
    //  wall and hid its mass from EqPress.  With the gas pressure from the
    //  gas's mass (FSPICE EqPress) and no face momentum lock between particle-
    //  free cells (ScalarExch::vel_FC_exchange) the face form gives the exact
    //  interface temperature (T2: 492.7 K, exact 492.3 K), and the realized
    //  form, not telescoping, lost energy: T3 drift -2.3% vs -1.0%.)
    constCCVariable<double> interiorP;
    const bool placeholderWork = d_ice->d_with_mpm && d_ktInterfaceExch;
    const double traceVolFrac = 1.0e-3;     // as in FSPICE EqPress, LODI2.cc, gasCombustion
    if( placeholderWork ){
      const int iceIndx = m_materialManager->getMaterial( "ICE", 0 )->getDWIndex();
      new_dw->get( interiorP, Slb->solidInteriorP_CCLabel, iceIndx, patch, m_gn, 0 );
    }

    //__________________________________
    //  <interfaceExchange> conduction: heat across the gas/solid interface
    CCVariable<double> condGas;
    std::vector< CCVariable<double> > condSolid;
    std::vector<int> mpmDWIndex;
    const bool conduction = d_ice->d_with_mpm && d_ktInterfaceCond;
    if( conduction ){
      const unsigned int numMPM = m_materialManager->getNumMatls( "MPM" );
      condSolid.resize( numMPM );
      for(unsigned int i = 0; i < numMPM; i++){
        mpmDWIndex.push_back( m_materialManager->getMaterial( "MPM", i )->getDWIndex() );
        new_dw->allocateTemporary( condSolid[i], patch );
        condSolid[i].initialize( 0.0 );
      }
      new_dw->allocateTemporary( condGas, patch );
      condGas.initialize( 0.0 );
      interfaceConduction( patch, old_dw, new_dw, delT, condGas, condSolid );
    }

    //__________________________________
    //  <interfaceShear> model: the kinetic energy the wall shear dissipates,
    //  heat in the gas cell (the shear's momentum is booked in
    //  accumulateMomentumSourceSinks from the same inputs)
    CCVariable<double> shearHeat;
    const bool shear = d_ice->d_with_mpm && d_ktInterfaceShear;
    const int  shearGasIndx = m_materialManager->getMaterial( "ICE", 0 )->getDWIndex();
    if( shear ){
      const unsigned int numMPM = m_materialManager->getNumMatls( "MPM" );
      CCVariable<Vector> momGas;
      std::vector< CCVariable<Vector> > momSolid( numMPM );
      for(unsigned int i = 0; i < numMPM; i++){
        new_dw->allocateTemporary( momSolid[i], patch );
        momSolid[i].initialize( Vector( 0.0, 0.0, 0.0 ) );
      }
      new_dw->allocateTemporary( momGas, patch );
      momGas.initialize( Vector( 0.0, 0.0, 0.0 ) );
      new_dw->allocateTemporary( shearHeat, patch );
      shearHeat.initialize( 0.0 );
      interfaceShear( patch, old_dw, new_dw, delT, momGas, momSolid, &shearHeat );
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
            if( placeholderWork && interiorP[c] > 0.5 && vol_frac[c] < 10.0*traceVolFrac ){
              //  placeholder weight: 1 at vol_frac <= traceVolFrac, 0 at >= 10*traceVolFrac
              const double place =
                std::min( 1.0, std::max( 0.0, ( 10.0*traceVolFrac - vol_frac[c] )/( 9.0*traceVolFrac ) ) );
              int_eng_source[c] += ( 1.0 - place )*f_theta[c]*netPressWork[c] + heatCond_src[c];
            } else {
              int_eng_source[c] += f_theta[c] * netPressWork[c] + heatCond_src[c];
            }
          }
        }
      }

      //__________________________________
      //  <interfaceExchange> conduction: the gas loses what the solid gains
      //  (also added to heatCond_src, which is output only past this point)
      if( conduction ){
        const CCVariable<double>* cond = nullptr;
        if( ice_matl ){
          cond = &condGas;
        }
        for( size_t i = 0; i < mpmDWIndex.size(); i++ ){
          if( mpmDWIndex[i] == indx ){
            cond = &condSolid[i];
          }
        }
        if( cond ){
          for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
            const IntVector c = *iter;
            int_eng_source[c] += (*cond)[c];
            heatCond_src[c]   += (*cond)[c];
          }
        }
      }

      //__________________________________
      //  <interfaceShear> model: the shear's dissipated kinetic energy
      if( shear && indx == shearGasIndx ){
        for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
          const IntVector c = *iter;
          int_eng_source[c] += shearHeat[c];
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

namespace {
  //______________________________________________________________________
  //  <interfaceExchange> conduction: the heat flux one side of a gas/solid
  //  interface delivers to it (W/m^2, positive when the side is hotter than
  //  the interface temperature Ti), from the side's two nearest temperatures:
  //  T1 at distance x1 (standing for a layer of thickness w1) and T2 at x2.
  //   - resolved layer: the one-sided second-order gradient through
  //     (0, Ti), (x1, T1), (x2, T2);
  //   - thin layer (thinner than the first cell): heat-balance integral.  The
  //     first node's departure from the second is the heat an erfc layer of
  //     thickness delta = sqrt(pi alpha t) holds,
  //         delta = (pi/2) w1 (T2 - T1)/(T2 - Ti),   q = k (T2 - Ti)/delta,
  //     exact for a semi-infinite body after a step, with no stored history.
  //     delta is held at or above one step's diffusion length sqrt(pi alpha dt);
  //   - blended by delta/w1: thin below 1/2, resolved above 1, linear between.
  //  Without a far node (x2 has no material of this side) it is the
  //  two-point form k (T1 - Ti)/x1.
  double layerFlux( double Ti, double T1, double T2, double x1, double w1, double x2,
                    double k, double alpha, bool farOK, double dt )
  {
    x1 = std::max( x1, 1.0e-3*x2 );
    if( !farOK ){
      return k*( T1 - Ti )/x1;
    }
    const double qRes = k*( T1*x2/( x1*( x2 - x1 ) ) - T2*x1/( x2*( x2 - x1 ) )
                           - Ti*( x1 + x2 )/( x1*x2 ) );
    const double dFar  = T2 - Ti;
    const double dCell = T2 - T1;
    if( dFar*dCell <= 0.0 || std::fabs( dCell ) >= std::fabs( dFar ) ){
      return qRes;
    }
    double delta = 0.5*M_PI*w1*dCell/dFar;
    delta = std::max( delta, std::sqrt( M_PI*alpha*dt ) );
    const double qThin = k*dFar/delta;
    const double wRes  = std::min( 1.0, std::max( 0.0, ( delta/w1 - 0.5 )/0.5 ) );
    return ( 1.0 - wRes )*qThin + wRes*qRes;
  }

  //  interface temperature with continuous flux: q_gas(Ti) + q_solid(Ti) = 0
  double interfaceTemp( double Tg1, double Tg2, double xg1, double wg1, double xg2, double kg,
                        double ag, bool gFarOK,
                        double Ts1, double Ts2, double xs1, double ws1, double xs2, double ks,
                        double as, bool sFarOK, double dt )
  {
    auto f = [&]( double Ti ){
      return layerFlux( Ti, Tg1, Tg2, xg1, wg1, xg2, kg, ag, gFarOK, dt )
           + layerFlux( Ti, Ts1, Ts2, xs1, ws1, xs2, ks, as, sFarOK, dt );
    };
    double lo = std::min( std::min( Tg1, Tg2 ), std::min( Ts1, Ts2 ) );
    double hi = std::max( std::max( Tg1, Tg2 ), std::max( Ts1, Ts2 ) );
    double fLo = f( lo );
    for( int it = 0; it < 60; it++ ){
      const double mid  = 0.5*( lo + hi );
      const double fMid = f( mid );
      if( ( fLo > 0.0 ) == ( fMid > 0.0 ) ){
        lo  = mid;
        fLo = fMid;
      } else {
        hi = mid;
      }
    }
    return 0.5*( lo + hi );
  }

  //______________________________________________________________________
  //  The gas node - solid node pairs of interface owner cell b: the geometry
  //  of SPICE::interfaceConduction, shared with SPICE::interfaceShear.  For each
  //  direction d with a nonzero solid-fraction gradient:
  //      fn( g, sc, w, d, n, xg1, wg1, gFar, xg2, xs1, ws1, sFar, xs2 )
  //  g, sc = the gas and solid nodes; w = the pair's weight (direction weight
  //  n_d^2 times the blend that hands the pair over as the surface crosses the
  //  cell); n = grad s/|grad s| (into the solid); per side, in cells from the
  //  interface: x1 = the node's distance, w1 = the thickness of the layer it
  //  stands for, and the far node (one more cell away) at x2.
  template< class InDomain, class Fn >
  void forInterfacePairs( const IntVector & b, const constCCVariable<double> & s,
                          const InDomain & inDomain, Fn && fn )
  {
    const IntVector unit[3] = { IntVector(1,0,0), IntVector(0,1,0), IntVector(0,0,1) };
    const double sb = s[b];
    if( !( sb > 0.0 ) ){
      return;
    }
    double gs[3] = { 0.0, 0.0, 0.0 };
    double gs2   = 0.0;
    for( int d = 0; d < 3; d++ ){
      const IntVector pl = inDomain( b + unit[d] ) ? b + unit[d] : b;
      const IntVector mi = inDomain( b - unit[d] ) ? b - unit[d] : b;
      gs[d] = s[pl] - s[mi];
      gs2  += gs[d]*gs[d];
    }
    if( !( gs2 > 0.0 ) ){
      return;
    }
    const double gLen = std::sqrt( gs2 );
    const Vector n( gs[0]/gLen, gs[1]/gLen, gs[2]/gLen );

    for( int d = 0; d < 3; d++ ){
      if( gs[d] == 0.0 ){
        continue;
      }
      const IntVector a = ( gs[d] > 0.0 ) ? b - unit[d] : b + unit[d];
      const IntVector c = ( gs[d] > 0.0 ) ? b + unit[d] : b - unit[d];
      if( !inDomain( a ) ){
        continue;
      }
      if( sb >= 1.0 && s[a] > 0.0 ){
        continue;            // a full cell owns only a face-aligned interface
      }
      const double wd = gs[d]*gs[d]/gs2;
      const double xi = 1.0 - sb;

      //  far nodes: one more cell away from the interface on each side
      const IntVector a2 = a - ( c - b );
      const IntVector c2 = c + ( c - b );
      if( sb >= 0.5 ){
        const double theta = 2.0*( 1.0 - sb );
        //  gas a (full cell, x1 = xi + 1/2), solid b (partial, centroid at sb/2)
        fn( a, b, wd*( 1.0 - theta ), d, n,
            xi + 0.5, xi + 1.0, a2, xi + 1.5,
            0.5*sb,   sb,       c,  1.5 - xi );
        //  gas b (sliver, centroid at xi/2), solid b
        fn( b, b, wd*theta, d, n,
            0.5*xi,   xi,       a,  xi + 0.5,
            0.5*sb,   sb,       c,  1.5 - xi );
      } else {
        const double phi = 2.0*sb;
        fn( b, b, wd*phi, d, n,
            0.5*xi,   xi,       a,  xi + 0.5,
            0.5*sb,   sb,       c,  1.5 - xi );
        //  gas b, solid c (full cell)
        fn( b, c, wd*( 1.0 - phi ), d, n,
            0.5*xi,   xi,       a,  xi + 0.5,
            1.5 - xi, 2.0 - xi, c2, 2.5 - xi );
      }
    }
  }
}

/* _____________________________________________________________________
 Function:  SPICE::interfaceConduction
 Purpose:   <CentralUpwind><interfaceExchange> conduction (FSPICE).  The heat
            conducted across the gas/solid interface in one time step, as
            energy sources for the gas (gasSrc, J) and for each MPM material
            (solidSrc, J).  The gas loses exactly what the solids gain.

 Where the interface is: solidVolFracP_CC (s), the sharp solid fraction of
 FSPICE::computeSolidVolFracP.  Along each direction d an owner cell b holds
 the interface: 0 < s_b < 1, or s_b = 1 with s = 0 in its gas-side
 neighbor a (a face-aligned surface).  The gas side is the neighbor with the
 smaller s, c the neighbor on the solid side.  In cells from the a|b face:
 interface at xi = 1 - s_b, gas centroid of b at xi/2, solid centroid of b
 at xi + s_b/2, a's center at -1/2, c's at 3/2.

 Energy balance of conduction at the interface between a gas and a solid
 temperature node: each side delivers the flux its near-interface profile
 implies (layerFlux: second-order gradient through its two nearest nodes
 when its thermal layer spans the first cell, the heat-balance-integral
 thin-layer flux when it does not), and the interface temperature makes the
 two equal (interfaceTemp).  G = q A/(T_g - T_s).  With a linear profile
 this is the two-point series form
     q A = G (T_g - T_s),     G = A / ( delta_g/k_g + delta_s/k_s )
 (also the fallback where a side has no second node).  Item 2, 2026-10-28:
 the two-point form alone replaces each side's layer thickness sqrt(pi alpha
 t) by half a cell, so a layer thinner than that (a polymer wall over a
 detonation's 100 us; any wall in the first microseconds) got far too little
 heat: on a coarse 1D prototype 54% low at 3 us and 9% at 150 us for the T5
 wall, against 0.6% at 15 us with this model (verification/wallmodel_proto.py).
 delta = each node's distance to the interface, k_g = thermalCond_CC of the
 gas, k_s = the MPM thermal_conductivity.  Nothing is conducted when the gas
 material's <thermal_conductivity> is 0 (no conduction inside the gas).
 The nodes are the gas and solid centroids nearest the interface, blended
 as it crosses b so that G changes continuously:
     s_b >= 1/2:  (gas a, solid b)(1 - theta) + (gas b, solid b) theta,
                  theta = 2 (1 - s_b)
     s_b <  1/2:  (gas b, solid b) phi + (gas b, solid c)(1 - phi),
                  phi = 2 s_b
 At s_b = 1 this is the face-aligned two-point flux (delta_g = delta_s =
 dx/2); as s_b -> 0 it hands over, unchanged, to cell c as owner.
 Direction weights n_d^2 (n = grad s/|grad s|; 1 in 1D) and the face area
 normal to d.  Multi-D geometry is approximate: no cut-cell
 reconstruction, and a surface oblique to the grid can be counted by two
 owners.  With several MPM materials each takes its share of the MPM
 vol_frac on the solid side.

 Time integration, per pair, point implicit:
     dE = G dt (T_g - T_s) / ( 1 + dt SumG_g/C_g + dt SumG_s/C_s )
 SumG = the conductances of every pair on that node, C = its heat capacity
 rho V cv.  Resolved conduction (dt SumG/C << 1) gets the explicit flux; a
 node with little gas (a pocket the surface just uncovered) relaxes to its
 partners without overshoot.
 Inputs are start-of-step values (gas T and rho from the old DW, solid T
 from interpolateNCToCC_0, s, k), so each PITCH outer iteration books the
 same energy.  Both patches holding a node of a pair evaluate it alike
 (owners scanned in index order), so the result does not depend on the
 patch layout.
 _____________________________________________________________________  */
void SPICE::interfaceConduction( const Patch                       * patch,
                               DataWarehouse                     * old_dw,
                               DataWarehouse                     * new_dw,
                               const double                        delT,
                               CCVariable<double>                & gasSrc,
                               std::vector< CCVariable<double> > & solidSrc )
{
  //  the gas conducts only when its <thermal_conductivity> is nonzero: the same
  //  test accumulateEnergySourceSinks applies to the gas's own conduction
  //  (with gasCombustion, thermalCond_CC then holds the mixture conductivity).
  //  Conducting across the interface without conducting inside the gas
  //  drains only the gas cell at the wall.
  ICEMaterial* gasMatl = (ICEMaterial*) m_materialManager->getMaterial( "ICE", 0 );
  if( gasMatl->getThermalConductivity() == 0.0 ){
    return;
  }
  const int gasIndx = gasMatl->getDWIndex();
  const unsigned int numMPM = m_materialManager->getNumMatls( "MPM" );

  const Vector dx   = patch->dCell();
  const double vol  = dx.x()*dx.y()*dx.z();
  const double area[3] = { dx.y()*dx.z(), dx.x()*dx.z(), dx.x()*dx.y() };
  const double tiny = 1.0e-100;

  //  ghosts: owners up to 2 cells outside the patch, their nodes up to 3, and
  //  each node's far neighbour one more (temperatures and fractions)
  constCCVariable<double> s, Tg, kg, rhoG, cvG, vfG;
  new_dw->get( s,    Slb->solidVolFracP_CCLabel, gasIndx, patch, m_gac, 3 );
  old_dw->get( Tg,   lb->temp_CCLabel,          gasIndx, patch, m_gac, 3 );
  new_dw->get( kg,   lb->thermalCondLabel,      gasIndx, patch, m_gac, 2 );
  old_dw->get( rhoG, lb->rho_CCLabel,           gasIndx, patch, m_gac, 2 );
  new_dw->get( cvG,  lb->specific_heatLabel,    gasIndx, patch, m_gac, 2 );
  new_dw->get( vfG,  lb->vol_frac_CCLabel,      gasIndx, patch, m_gac, 3 );

  std::vector< constCCVariable<double> > Ts( numMPM ), rhoS( numMPM ), vfS( numMPM );
  std::vector<double> kS( numMPM ), cvS( numMPM );
  for( unsigned int i = 0; i < numMPM; i++ ){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", i );
    const int indx = mpm_matl->getDWIndex();
    new_dw->get( Ts[i],   lb->temp_CCLabel,     indx, patch, m_gac, 3 );
    new_dw->get( rhoS[i], lb->rho_CCLabel,      indx, patch, m_gac, 2 );
    new_dw->get( vfS[i],  lb->vol_frac_CCLabel, indx, patch, m_gac, 3 );
    kS[i]  = mpm_matl->getThermalConductivity();
    cvS[i] = mpm_matl->getSpecificHeat();
  }

  //__________________________________
  //  index windows: the patch's cells grown by g layers, not past a
  //  non-periodic domain face
  const Level* level = patch->getLevel();
  IntVector loI, hiI;
  level->findInteriorCellIndexRange( loI, hiI );
  const IntVector periodic = level->getPeriodicBoundaries();

  auto inDomain = [&]( const IntVector & c ) -> bool {
    for( int k = 0; k < 3; k++ ){
      if( periodic[k] == 0 && ( c[k] < loI[k] || c[k] >= hiI[k] ) ){
        return false;
      }
    }
    return true;
  };
  auto window = [&]( int g, IntVector & l, IntVector & h ){
    l = patch->getCellLowIndex()  - IntVector( g, g, g );
    h = patch->getCellHighIndex() + IntVector( g, g, g );
    for( int k = 0; k < 3; k++ ){
      if( periodic[k] == 0 ){
        l[k] = std::max( l[k], loI[k] );
        h[k] = std::min( h[k], hiI[k] );
      }
    }
  };
  IntVector l0, h0, l1, h1, l2, h2;
  window( 0, l0, h0 );
  window( 1, l1, h1 );
  window( 2, l2, h2 );
  auto inBox = []( const IntVector & c, const IntVector & l, const IntVector & h ) -> bool {
    return c.x() >= l.x() && c.y() >= l.y() && c.z() >= l.z() &&
           c.x() <  h.x() && c.y() <  h.y() && c.z() <  h.z();
  };
  auto inW0 = [&]( const IntVector & c ){ return inBox( c, l0, h0 ); };
  auto inW1 = [&]( const IntVector & c ){ return inBox( c, l1, h1 ); };

  //__________________________________
  //  the pairs of owner cell b that have a node where need() holds:
  //  fn( gas cell, solid cell, MPM material, G )
  auto forPairs = [&]( const IntVector & b, auto && need, auto && fn ){
    forInterfacePairs( b, s, inDomain,
      [&]( const IntVector & g, const IntVector & sc, double w, int d, const Vector & /*n*/,
           double xg1, double wg1, const IntVector & gFar, double xg2,
           double xs1, double ws1, const IntVector & sFar, double xs2 ){
        if( !( w > 0.0 ) || !inDomain( sc ) || !need( g, sc ) ){
          return;
        }
        const double h = dx[d];
        const double k = kg[g];
        if( !( k > 0.0 ) ){
          return;
        }
        double vfSum = 0.0, vfSumFar = 0.0;
        for( unsigned int i = 0; i < numMPM; i++ ){
          vfSum += vfS[i][sc];
          if( inDomain( sFar ) ){
            vfSumFar += vfS[i][sFar];
          }
        }
        if( !( vfSum > 0.0 ) ){
          return;
        }
        //  far nodes count only if they hold their side's material
        const bool gasFarOK   = inDomain( gFar ) && vfG[gFar] > 0.5;
        const bool solidFarOK = inDomain( sFar ) && vfSumFar > 0.5;
        const double TgFar    = gasFarOK ? Tg[gFar] : Tg[g];
        const double alphaG   = k/std::max( rhoG[g]*cvG[g], tiny );

        for( unsigned int i = 0; i < numMPM; i++ ){
          const double share = vfS[i][sc]/vfSum;
          if( !( share > 0.0 && kS[i] > 0.0 ) ){
            continue;
          }
          const double TsFar  = solidFarOK ? Ts[i][sFar] : Ts[i][sc];
          const double alphaS = kS[i]/std::max( rhoS[i][sc]*cvS[i], tiny );
          const double dT     = Tg[g] - Ts[i][sc];

          //  half-cell (two-point) conductance: the layer model's fallback and
          //  the form it reduces to for a linear profile
          const double Ghalf = w*share*area[d]/( xg1*h/k + xs1*h/kS[i] );
          double G = Ghalf;
          if( std::fabs( dT ) > 1.0e-12*( std::fabs( Tg[g] ) + std::fabs( Ts[i][sc] ) ) ){
            const double Ti = interfaceTemp( Tg[g], TgFar, xg1*h, wg1*h, xg2*h, k, alphaG, gasFarOK,
                                             Ts[i][sc], TsFar, xs1*h, ws1*h, xs2*h, kS[i], alphaS, solidFarOK,
                                             delT );
            const double q  = layerFlux( Ti, Tg[g], TgFar, xg1*h, wg1*h, xg2*h, k, alphaG, gasFarOK, delT );
            G = w*share*area[d]*q/dT;
            if( !( G > 0.0 ) ){
              G = Ghalf;
            }
          }
          fn( g, sc, i, G );
        }
      });
  };

  //__________________________________
  //  1) every node's total conductance (nodes up to 1 cell outside the
  //     patch: owners up to 2 cells out)
  CCVariable<double> sumGg;
  std::vector< CCVariable<double> > sumGs( numMPM );
  new_dw->allocateTemporary( sumGg, patch, m_gac, 1 );
  sumGg.initialize( 0.0 );
  for( unsigned int i = 0; i < numMPM; i++ ){
    new_dw->allocateTemporary( sumGs[i], patch, m_gac, 1 );
    sumGs[i].initialize( 0.0 );
  }
  auto needW1 = [&]( const IntVector & g, const IntVector & sc ){ return inW1( g ) || inW1( sc ); };
  for( CellIterator it( l2, h2 ); !it.done(); it++ ){
    forPairs( *it, needW1, [&]( const IntVector & g, const IntVector & sc, unsigned int i, double G ){
      if( inW1( g ) ){
        sumGg[g] += G;
      }
      if( inW1( sc ) ){
        sumGs[i][sc] += G;
      }
    });
  }

  //__________________________________
  //  2) the energy each pair moves, booked on this patch's nodes
  auto needW0 = [&]( const IntVector & g, const IntVector & sc ){ return inW0( g ) || inW0( sc ); };
  for( CellIterator it( l1, h1 ); !it.done(); it++ ){
    forPairs( *it, needW0, [&]( const IntVector & g, const IntVector & sc, unsigned int i, double G ){
      const double Cg   = std::max( rhoG[g]*vol*cvG[g],      tiny );
      const double Cs   = std::max( rhoS[i][sc]*vol*cvS[i],  tiny );
      const double damp = 1.0/( 1.0 + delT*sumGg[g]/Cg + delT*sumGs[i][sc]/Cs );
      const double dE   = G*delT*( Tg[g] - Ts[i][sc] )*damp;
      if( inW0( g ) ){
        gasSrc[g] -= dE;
      }
      if( inW0( sc ) ){
        solidSrc[i][sc] += dE;
      }
    });
  }
}

/* _____________________________________________________________________
 Function:  SPICE::interfaceShear
 Purpose:   <CentralUpwind><interfaceShear> model (FSPICE).  The wall shear
            between the gas and the solid over one time step: momentum for
            the gas (gasMom, kg m/s) and for each MPM material (solidMom); the
            gas loses exactly what the solids gain.  gasHeat (J, optional):
            the kinetic energy the shear dissipates, booked as heat in the
            gas cell.

 The momentum form of SPICE::interfaceConduction, on the same interface pairs
 (forInterfacePairs).  Per pair, the gas's velocity relative to the solid
 node, tangential to the surface (n = grad s/|grad s|):
     u_t = (v_g - v_s) - ((v_g - v_s).n) n,   u1 = |u_t|,  e = u_t/u1,
 and the far gas node's, along e: u2.  The interface is no-slip (the gas at
 the surface moves with the solid), so the shear is layerFlux with interface
 value 0, k -> the gas viscosity mu (viscosity_CC: the ICE value or
 gasCombustion's mixture viscosity) and alpha -> nu = mu/rho:
   - momentum layer thicker than the first cell: second-order gradient;
   - thinner: the momentum-integral (Stokes layer) estimate
         delta = (pi/2) w1 (u2 - u1)/u2,   tau = mu u2/delta,
     exact for Stokes' first problem, no stored history, delta held at or
     above one step's diffusion length sqrt(pi nu dt);
   - no far gas node: tau = mu u1/x1.
 The drag coefficient G = w A tau/u1 (kg/s) is applied point implicitly,
     J = G dt u_t / ( 1 + dt SumG_g/m_g + dt SumG_s/m_s ),
 m = the node's mass: a near-wall pocket with little gas is brought to the
 solid's velocity without reversing.  No shear where the gas viscosity is 0.
 Needs <interfaceShear> slip in the momentum exchange: the cell in front of
 the solid keeps the exchange only along n, so its gas slides and this
 shear is the only tangential coupling.  ICE's own viscous stress carries
 nothing across the interface face (its face viscosity is the harmonic mean
 of mu*vol_frac, ~0 against the trace gas inside the solid).
 Heat: D = J.u_t - |J|^2 (1/m_g + 1/m_s)/2 >= 0, the kinetic energy the pair
 loses, goes into the gas cell, so the shear conserves energy.
 Inputs are start-of-step values (gas velocity and density from the old DW,
 solid velocity from interpolateNCToCC_0), so accumulateMomentumSourceSinks
 and accumulateEnergySourceSinks (each calls this) and every PITCH outer
 iteration see the same pairs.  Patch-layout independent as interfaceConduction.
 _____________________________________________________________________  */
void SPICE::interfaceShear( const Patch                       * patch,
                          DataWarehouse                     * old_dw,
                          DataWarehouse                     * new_dw,
                          const double                        delT,
                          CCVariable<Vector>                & gasMom,
                          std::vector< CCVariable<Vector> > & solidMom,
                          CCVariable<double>                * gasHeat )
{
  ICEMaterial* gasMatl = (ICEMaterial*) m_materialManager->getMaterial( "ICE", 0 );
  const int gasIndx = gasMatl->getDWIndex();
  const unsigned int numMPM = m_materialManager->getNumMatls( "MPM" );

  const Vector dx   = patch->dCell();
  const double vol  = dx.x()*dx.y()*dx.z();
  const double area[3] = { dx.y()*dx.z(), dx.x()*dx.z(), dx.x()*dx.y() };
  const double tiny = 1.0e-100;

  //  ghosts as interfaceConduction: velocities and fractions 3, the rest 2
  constCCVariable<double> s, mu, rhoG, vfG;
  constCCVariable<Vector> velG;
  new_dw->get( s,    Slb->solidVolFracP_CCLabel, gasIndx, patch, m_gac, 3 );
  old_dw->get( velG, lb->vel_CCLabel,           gasIndx, patch, m_gac, 3 );
  new_dw->get( mu,   lb->viscosityLabel,        gasIndx, patch, m_gac, 2 );
  old_dw->get( rhoG, lb->rho_CCLabel,           gasIndx, patch, m_gac, 2 );
  new_dw->get( vfG,  lb->vol_frac_CCLabel,      gasIndx, patch, m_gac, 3 );

  std::vector< constCCVariable<Vector> > velS( numMPM );
  std::vector< constCCVariable<double> > rhoS( numMPM ), vfS( numMPM );
  for( unsigned int i = 0; i < numMPM; i++ ){
    const int indx = m_materialManager->getMaterial( "MPM", i )->getDWIndex();
    new_dw->get( velS[i], lb->vel_CCLabel,      indx, patch, m_gac, 3 );
    new_dw->get( rhoS[i], lb->rho_CCLabel,      indx, patch, m_gac, 2 );
    new_dw->get( vfS[i],  lb->vol_frac_CCLabel, indx, patch, m_gac, 3 );
  }

  //__________________________________
  //  index windows (as interfaceConduction)
  const Level* level = patch->getLevel();
  IntVector loI, hiI;
  level->findInteriorCellIndexRange( loI, hiI );
  const IntVector periodic = level->getPeriodicBoundaries();

  auto inDomain = [&]( const IntVector & c ) -> bool {
    for( int k = 0; k < 3; k++ ){
      if( periodic[k] == 0 && ( c[k] < loI[k] || c[k] >= hiI[k] ) ){
        return false;
      }
    }
    return true;
  };
  auto window = [&]( int g, IntVector & l, IntVector & h ){
    l = patch->getCellLowIndex()  - IntVector( g, g, g );
    h = patch->getCellHighIndex() + IntVector( g, g, g );
    for( int k = 0; k < 3; k++ ){
      if( periodic[k] == 0 ){
        l[k] = std::max( l[k], loI[k] );
        h[k] = std::min( h[k], hiI[k] );
      }
    }
  };
  IntVector l0, h0, l1, h1, l2, h2;
  window( 0, l0, h0 );
  window( 1, l1, h1 );
  window( 2, l2, h2 );
  auto inBox = []( const IntVector & c, const IntVector & l, const IntVector & h ) -> bool {
    return c.x() >= l.x() && c.y() >= l.y() && c.z() >= l.z() &&
           c.x() <  h.x() && c.y() <  h.y() && c.z() <  h.z();
  };
  auto inW0 = [&]( const IntVector & c ){ return inBox( c, l0, h0 ); };
  auto inW1 = [&]( const IntVector & c ){ return inBox( c, l1, h1 ); };

  //__________________________________
  //  the pairs of owner cell b that have a node where need() holds:
  //  fn( gas cell, solid cell, MPM material, G, u_t )
  auto forPairs = [&]( const IntVector & b, auto && need, auto && fn ){
    forInterfacePairs( b, s, inDomain,
      [&]( const IntVector & g, const IntVector & sc, double w, int d, const Vector & n,
           double xg1, double wg1, const IntVector & gFar, double xg2,
           double /*xs1*/, double /*ws1*/, const IntVector & /*sFar*/, double /*xs2*/ ){
        if( !( w > 0.0 ) || !inDomain( sc ) || !need( g, sc ) ){
          return;
        }
        const double h = dx[d];
        const double m = mu[g];
        if( !( m > 0.0 ) ){
          return;
        }
        double vfSum = 0.0;
        for( unsigned int i = 0; i < numMPM; i++ ){
          vfSum += vfS[i][sc];
        }
        if( !( vfSum > 0.0 ) ){
          return;
        }
        //  the far gas node counts only if it holds gas
        const bool gasFarOK = inDomain( gFar ) && vfG[gFar] > 0.5;
        const double nu     = m/std::max( rhoG[g], tiny );

        for( unsigned int i = 0; i < numMPM; i++ ){
          const double share = vfS[i][sc]/vfSum;
          if( !( share > 0.0 ) ){
            continue;
          }
          const Vector rel = velG[g] - velS[i][sc];
          const Vector ut  = rel - Dot( rel, n )*n;
          const double u1  = ut.length();
          if( !( u1 > 1.0e-12*( velG[g].length() + velS[i][sc].length() ) ) ){
            continue;
          }
          const Vector e = ut/u1;
          double u2 = u1;
          if( gasFarOK ){
            const Vector relFar = velG[gFar] - velS[i][sc];
            u2 = Dot( relFar - Dot( relFar, n )*n, e );
          }
          //  half-cell (two-point) drag: the fallback
          const double Ghalf = w*share*area[d]*m/( xg1*h );
          const double tau   = layerFlux( 0.0, u1, u2, xg1*h, wg1*h, xg2*h, m, nu, gasFarOK, delT );
          double G = w*share*area[d]*tau/u1;
          if( !( G > 0.0 ) ){
            G = Ghalf;
          }
          fn( g, sc, i, G, ut );
        }
      });
  };

  //__________________________________
  //  1) every node's total drag coefficient (nodes up to 1 cell outside the
  //     patch: owners up to 2 cells out)
  CCVariable<double> sumGg;
  std::vector< CCVariable<double> > sumGs( numMPM );
  new_dw->allocateTemporary( sumGg, patch, m_gac, 1 );
  sumGg.initialize( 0.0 );
  for( unsigned int i = 0; i < numMPM; i++ ){
    new_dw->allocateTemporary( sumGs[i], patch, m_gac, 1 );
    sumGs[i].initialize( 0.0 );
  }
  auto needW1 = [&]( const IntVector & g, const IntVector & sc ){ return inW1( g ) || inW1( sc ); };
  for( CellIterator it( l2, h2 ); !it.done(); it++ ){
    forPairs( *it, needW1, [&]( const IntVector & g, const IntVector & sc, unsigned int i, double G,
                                const Vector & /*ut*/ ){
      if( inW1( g ) ){
        sumGg[g] += G;
      }
      if( inW1( sc ) ){
        sumGs[i][sc] += G;
      }
    });
  }

  //__________________________________
  //  2) the momentum each pair moves (and the heat it makes), booked on this
  //     patch's nodes
  auto needW0 = [&]( const IntVector & g, const IntVector & sc ){ return inW0( g ) || inW0( sc ); };
  for( CellIterator it( l1, h1 ); !it.done(); it++ ){
    forPairs( *it, needW0, [&]( const IntVector & g, const IntVector & sc, unsigned int i, double G,
                                const Vector & ut ){
      const double mg   = std::max( rhoG[g]*vol,     tiny );
      const double ms   = std::max( rhoS[i][sc]*vol, tiny );
      const double damp = 1.0/( 1.0 + delT*sumGg[g]/mg + delT*sumGs[i][sc]/ms );
      const Vector J    = ut*( G*delT*damp );
      if( inW0( g ) ){
        gasMom[g] -= J;
        if( gasHeat ){
          ( *gasHeat )[g] += Dot( J, ut ) - 0.5*J.length2()*( 1.0/mg + 1.0/ms );
        }
      }
      if( inW0( sc ) ){
        solidMom[i][sc] += J;
      }
    });
  }
}

/* _____________________________________________________________________
  Task:  computeLagrangianSpecificVolume--
  purpose:  Compute the larganginan specific volume  mass * spec_vol
 _____________________________________________________________________  */
void SPICE::computeLagrangianSpecificVolume(const ProcessorGroup  *,
                                          const PatchSubset     * patches,
                                          const MaterialSubset  * /*matls*/,
                                          DataWarehouse         * old_dw,
                                          DataWarehouse         * new_dw)
{
  const Level* level = getLevel(patches);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, m_spice_tasks, "SPICE::computeLagrangianSpecificVolume" );

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, level);

    unsigned int numALLMatls = m_materialManager->getNumMatls();
    Vector  dx = patch->dCell();
    double vol = dx.x()*dx.y()*dx.z();

    std::vector<constCCVariable<double> > Tdot(numALLMatls);
    std::vector<constCCVariable<double> > vol_frac(numALLMatls);
    std::vector<constCCVariable<double> > Temp_CC(numALLMatls);
    std::vector<CCVariable<double> > alpha(numALLMatls);

    constCCVariable<double> rho_CC;
    constCCVariable<double> f_theta;
    constCCVariable<double> sp_vol_CC;
    constCCVariable<double> cv;
    constCCVariable<double> delP;
    constCCVariable<double> P;
    constCCVariable<double> TMV_CC;
    CCVariable<double> sum_therm_exp;
    vector<double> if_mpm_matl_ignore(numALLMatls);

    new_dw->allocateTemporary(sum_therm_exp,patch);
    new_dw->get( delP, lb->delP_DilatateLabel, 0,patch,m_gn,0 );
    new_dw->get( P,    lb->press_CCLabel,      0,patch,m_gn,0 );
    sum_therm_exp.initialize(0.);

    if(d_ice->d_with_mpm){
      new_dw->get( TMV_CC, lb->TMV_CCLabel, 0,patch,m_gn,0 );
    }
    else {
      CCVariable<double>  TMV_create;
      new_dw->allocateTemporary(TMV_create,  patch);
      TMV_create.initialize(vol);
      TMV_CC = TMV_create; // reference created data
    }

    for(unsigned int m = 0; m < numALLMatls; m++) {
      Material* matl = m_materialManager->getMaterial( m );
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      int indx = matl->getDWIndex();

      new_dw->get( Tdot[m],    lb->Tdot_CCLabel,    indx,patch,m_gn,0 );
      new_dw->get( vol_frac[m],lb->vol_frac_CCLabel,indx,patch,m_gac,1 );
      new_dw->allocateTemporary(alpha[m],patch);

      if (ice_matl) {
        old_dw->get( Temp_CC[m], lb->temp_CCLabel,  indx,patch,m_gn,0 );
      }
      if (mpm_matl) {
        new_dw->get( Temp_CC[m],lb->temp_CCLabel,   indx,patch,m_gn,0 );
      }
    }

    //__________________________________
    // Sum of thermal expansion
    // ignore contributions from mpm_matls
    // UNTIL we have temperature dependent EOS's for the solids
    for(unsigned int m = 0; m < numALLMatls; m++) {
      Material* matl = m_materialManager->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      int indx = matl->getDWIndex();

      if (ice_matl) {
       if_mpm_matl_ignore[m]=1.0;
       new_dw->get( sp_vol_CC, lb->sp_vol_CCLabel,    indx,patch,m_gn,0 );
       new_dw->get( cv,        lb->specific_heatLabel,indx,patch,m_gn,0 );

       for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
          IntVector c = *iter;
          alpha[m][c]=
            ice_matl->getEOS()->getAlpha(Temp_CC[m][c], sp_vol_CC[c], P[c], cv[c]);
          sum_therm_exp[c] += vol_frac[m][c] * alpha[m][c] * Tdot[m][c];
        }
      }
      else {
        if_mpm_matl_ignore[m]=0.0;
        alpha[m].initialize(0.0);
      }
    }

    //__________________________________
    for(unsigned int m = 0; m < numALLMatls; m++) {
      Material* matl = m_materialManager->getMaterial( m );
      int indx = matl->getDWIndex();

      CCVariable<double> sp_vol_L;
      CCVariable<double> sp_vol_src;
      constCCVariable<double> kappa;

      new_dw->allocateAndPut(sp_vol_L,  lb->sp_vol_L_CCLabel,   indx,patch);
      new_dw->allocateAndPut(sp_vol_src,lb->sp_vol_src_CCLabel, indx,patch);

      sp_vol_src.initialize(0.);

      double tiny_rho = 1.e-12;
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);

      if (ice_matl) {
        tiny_rho = ice_matl->getTinyRho();
      }

      new_dw->get( sp_vol_CC,  lb->sp_vol_CCLabel,     indx,patch,m_gn,0 );
      new_dw->get( rho_CC,     lb->rho_CCLabel,        indx,patch,m_gn,0 );
      new_dw->get( f_theta,    lb->f_theta_CCLabel,    indx,patch,m_gn,0 );
      new_dw->get( kappa,      lb->compressibilityLabel,indx,patch,m_gn,0 );

      //__________________________________
      //  compute sp_vol_L * mass
      for(CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        sp_vol_L[c] = (rho_CC[c] * vol)*sp_vol_CC[c];
      }

      //__________________________________
      //   Contributions from models
      constCCVariable<double> Modelsp_vol_src;
      if(d_ice->d_models.size() > 0){
        new_dw->get( Modelsp_vol_src, lb->modelVol_srcLabel, indx, patch,m_gn,0 );

        for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
         IntVector c = *iter;
         sp_vol_L[c] += Modelsp_vol_src[c];
        }
      }

      //__________________________________
      //  add the sources to sp_vol_L
      for(CellIterator iter=patch->getCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        //__________________________________
        //  term1: compression.  delP_Dilatate is the net face volume flux
        //  over (vol*sumKappa), so vol_frac*kappa*vol*delP is this matl's
        //  f_theta share of that flux.  Keep vol here.
        //
        //  term2: thermal expansion, a fraction of the matl's own volume
        //  mass*sp_vol = vol_frac*TMV (FSPICE and MPMICE normalize vol_frac by TMV,
        //  pure ICE has TMV = vol).  With vol in its place the change was
        //  scaled by vol/TMV: gas cooled by the wall in a cell with TMV < vol
        //  lost more than its whole volume (sp_vol_L < 0,
        //  detonationChannelMPM).  With TMV an ideal gas gives
        //  sp_vol_L/sp_vol_old = T_new/T_old, always > 0.
        double term1 = -vol_frac[m][c] * kappa[c] * vol * delP[c];
        double term2 = delT * TMV_CC[c] *
                             (vol_frac[m][c] * alpha[m][c] * Tdot[m][c] -
                                               f_theta[c] * sum_therm_exp[c]);

        // This is actually mass * sp_vol
        double src    = term1 + if_mpm_matl_ignore[m] * term2;
        sp_vol_L[c]  += src;
        sp_vol_src[c] = src/(rho_CC[c] * vol);
      }

      if(d_ice->d_clampSpecificVolume){
        for(CellIterator iter=patch->getCellIterator();!iter.done();iter++){
          IntVector c = *iter;
/*`==========TESTING==========*/
          sp_vol_L[c] = max(sp_vol_L[c], tiny_rho * vol * sp_vol_CC[c]);
/*==========TESTING==========`*/
        }
      }

      //__________________________________
      // Apply boundary conditions
      setSpecificVolBC(sp_vol_L, "SpecificVol", true ,rho_CC,vol_frac[m],
                       patch,m_materialManager, indx);


      //____ B U L L E T   P R O O F I N G----
      // ignore BP if recompute time step has already been requested
      IntVector neg_cell;
      bool rts = new_dw->recomputeTimeStep();

      if (!areAllValuesPositive(sp_vol_L, neg_cell) && !rts) {
        cout << "\nICE:WARNING......Negative specific Volume"<< endl;
        cout << "cell              "<< neg_cell << " level " <<  level->getIndex() << endl;
        cout << "matl              "<< indx << endl;
        cout << "sum_thermal_exp   "<< sum_therm_exp[neg_cell] << endl;
        cout << "sp_vol_src        "<< sp_vol_src[neg_cell] << endl;
        cout << "mass sp_vol_L     "<< sp_vol_L[neg_cell] << endl;
        cout << "mass sp_vol_L_old "
             << (rho_CC[neg_cell]*vol*sp_vol_CC[neg_cell]) << endl;
        cout << "-----------------------------------"<<endl;
//        ostringstream warn;
//        int L = level->getIndex();
//        warn<<"ERROR ICE:("<<L<<"):computeLagrangianSpecificVolumeRF, mat "<<indx
//            << " cell " <<neg_cell << " sp_vol_L is negative\n";
//        throw InvalidValue(warn.str(), __FILE__, __LINE__);

        new_dw->put( bool_or_vartype(true), VarLabel::find(abortTimeStep_name));
        new_dw->put( bool_or_vartype(true), VarLabel::find(recomputeTimeStep_name));
      }
    }  // end numALLMatl loop
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
      constSFCXVariable<double> gasFracX;
      constSFCYVariable<double> gasFracY;
      constSFCZVariable<double> gasFracZ;
      if( d_ktInterfaceDA ){
        new_dw->get( gasFracX, Slb->gasFaceFrac_FCXLabel, indx,patch,m_gac,2 );
        new_dw->get( gasFracY, Slb->gasFaceFrac_FCYLabel, indx,patch,m_gac,2 );
        new_dw->get( gasFracZ, Slb->gasFaceFrac_FCZLabel, indx,patch,m_gac,2 );
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

        //  FSPICE no-penetration faces carry the donor's own specific values
        //  (first order): a donor may lose up to half its gas there in one
        //  step, and a reconstructed face value would overdraw its energy
        if( d_ktInterfaceDA ){
          const SFCXVariable<double>& gasFracXRef = gasFracX;
          const SFCYVariable<double>& gasFracYRef = gasFracY;
          const SFCZVariable<double>& gasFracZRef = gasFracZ;
          ktAdvector->setInterfaceFaces( &gasFracXRef, &gasFracYRef, &gasFracZRef );
        }
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
 Task:      SPICE::pitchCopyModelSources
 Purpose:   PITCH sub-graph: copy the (frozen) model sources computed by
            the main graph's ICE::zeroModelSources + the models'
            scheduleComputeModelSources out of Task::ParentNewDW, exactly
            mirroring the label set ICE::zeroModelSources computes.
 _____________________________________________________________________  */
void SPICE::pitchCopyModelSources(const ProcessorGroup  *,
                                 const PatchSubset     * patches,
                                 const MaterialSubset  * matls,
                                 DataWarehouse         * /*old_dw*/,
                                 DataWarehouse         * new_dw)
{
  DataWarehouse* pNewDW = new_dw->getOtherDataWarehouse(Task::ParentNewDW);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, m_spice_tasks, "SPICE::pitchCopyModelSources" );

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
 Task:      SPICE::capturePitchAdv0
 Purpose:   Snapshot the main graph's (k=0) mass_adv/mom_adv/eng_adv/
            sp_vol_adv/tvar->var_adv into the "*_adv0" labels.
 _____________________________________________________________________  */
void SPICE::capturePitchAdv0(const ProcessorGroup  *,
                            const PatchSubset     * patches,
                            const MaterialSubset  * matls,
                            DataWarehouse         * /*old_dw*/,
                            DataWarehouse         * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, m_spice_tasks, "SPICE::capturePitchAdv0" );

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
          const VarLabel* adv0 = pitchTvarAdv0Label(idx);

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
 Task:      SPICE::pitchBlend
 Purpose:   PITCH sub-graph, outer iteration k: Crank-Nicolson theta
            blend of the conserved state,
              X^k = X^n + (1-theta)*(X_adv0 - X^n) + theta*(X_sub - X^{k-1})
            with energy blended as TOTAL energy (int_eng + kinetic) so the
            blend doesn't create/destroy heat through the KE nonlinearity
            (reactingPimpleCentralFoam, ddt CrankNicolson correspondence:
            theta=0.5, nOuter=2 => Heun/SSP-RK2; theta=1 => Euler-implicit
            Picard).
 _____________________________________________________________________  */
void SPICE::pitchBlend(const ProcessorGroup  *,
                      const PatchSubset     * patches,
                      const MaterialSubset  * matls,
                      DataWarehouse         * old_dw,
                      DataWarehouse         * new_dw)
{
  DataWarehouse* pOldDW = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
  DataWarehouse* pNewDW = new_dw->getOtherDataWarehouse(Task::ParentNewDW);

  const double th = d_pitchTheta;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, m_spice_tasks, "SPICE::pitchBlend" );

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
          const VarLabel* adv0 = pitchTvarAdv0Label(idx);

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
 Task:      SPICE::pitchResidual
 Purpose:   PITCH sub-graph, after conservedtoPrimitive_Vars: residual =
            max over interior cells of
            max(|rho_new-rho_old|/rho_new, |T_new-T_old|/T_new).
 _____________________________________________________________________  */
void SPICE::pitchResidual(const ProcessorGroup  *,
                         const PatchSubset     * patches,
                         const MaterialSubset  * matls,
                         DataWarehouse         * old_dw,
                         DataWarehouse         * new_dw)
{
  double residual = 0.0;

  DataWarehouse* pNewDW = d_ice->d_with_mpm ? new_dw->getOtherDataWarehouse(Task::ParentNewDW)
                                     : nullptr;
  const double fSolid = 1.0 - d_ktSolidCellFrac;

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, m_spice_tasks, "SPICE::pitchResidual" );

    for(int m=0;m<matls->size();m++){
      int indx = matls->get(m);

      constCCVariable<double> rho_old, temp_old, rho_new, temp_new;
      old_dw->get( rho_old,  lb->rho_CCLabel,  indx, patch, m_gn,0 );
      old_dw->get( temp_old, lb->temp_CCLabel, indx, patch, m_gn,0 );
      new_dw->get( rho_new,  lb->rho_CCLabel,  indx, patch, m_gn,0 );
      new_dw->get( temp_new, lb->temp_CCLabel, indx, patch, m_gn,0 );

      constCCVariable<double> vol_frac;
      if( d_ice->d_with_mpm ){
        pNewDW->get( vol_frac, lb->vol_frac_CCLabel, indx, patch, m_gn,0 );
      }

      for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
        IntVector c = *iter;

        if( d_ice->d_with_mpm && vol_frac[c] < fSolid ){
          continue;
        }

        double rRho  = std::fabs(rho_new[c]  - rho_old[c] ) / rho_new[c];
        double rTemp = std::fabs(temp_new[c] - temp_old[c]) / temp_new[c];

        residual = std::max( residual, std::max(rRho, rTemp) );
      }
    }
  }

  new_dw->put( max_vartype(residual), Slb->pitchResidualLabel );
}

//  Replace q by its time-n value (pOldDW) in solid cells, in subOldDW.  The
//  seeded q_k may share memory with ParentNewDW (transferFrom), so write a
//  fresh copy and put it with replace.
template<class T>
static void pitchRestoreSolidCells( const Patch                  * patch,
                                     const VarLabel               * label,
                                     const int                      indx,
                                     const constCCVariable<double>& vol_frac,
                                     const double                   fSolid,
                                     DataWarehouse                * pOldDW,
                                     DataWarehouse                * subOldDW )
{
  constCCVariable<T> q_n, q_k;
  pOldDW->get(   q_n, label, indx, patch, Ghost::None, 0 );
  subOldDW->get( q_k, label, indx, patch, Ghost::None, 0 );

  CCVariable<T> q;
  q.allocate( q_k.getLowIndex(), q_k.getHighIndex() );
  q.copyData( q_k );

  for( CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++ ){
    const IntVector& c = *iter;
    if( vol_frac[c] < fSolid ){
      q[c] = q_n[c];
    }
  }

  subOldDW->put( q, label, indx, patch, true );   // replace
}

/* _____________________________________________________________________
 Task:      SPICE::pitchRestartSolidCells
 Purpose:   FSPICE only.  Before each outer iteration, reset the ICE
            material's rho/temp/sp_vol/vel and transported variables in
            solid cells (gas vol_frac < 1 - solidCellFrac, k=0 value) of the
            sub OldDW to their time-n values.
 _____________________________________________________________________  */
void SPICE::pitchRestartSolidCells( const PatchSubset    * patches,
                                   const MaterialSubset * ice_mss,
                                   DataWarehouse        * pOldDW,
                                   DataWarehouse        * pNewDW,
                                   DataWarehouse        * subOldDW )
{
  const double fSolid = 1.0 - d_ktSolidCellFrac;   // gas vol_frac below this => solid cell

  d_pitchSched->setInitTimestep(true);

  for( int p = 0; p < patches->size(); p++ ){
    const Patch* patch = patches->get(p);

    for( int m = 0; m < ice_mss->size(); m++ ){
      const int indx = ice_mss->get(m);

      constCCVariable<double> vol_frac;   // k=0 value, fixed for the step
      pNewDW->get( vol_frac, lb->vol_frac_CCLabel, indx, patch, Ghost::None, 0 );

      pitchRestoreSolidCells<double>( patch, lb->rho_CCLabel,    indx, vol_frac, fSolid, pOldDW, subOldDW );
      pitchRestoreSolidCells<double>( patch, lb->temp_CCLabel,   indx, vol_frac, fSolid, pOldDW, subOldDW );
      pitchRestoreSolidCells<double>( patch, lb->sp_vol_CCLabel, indx, vol_frac, fSolid, pOldDW, subOldDW );
      pitchRestoreSolidCells<Vector>( patch, lb->vel_CCLabel,    indx, vol_frac, fSolid, pOldDW, subOldDW );

      //  The pressure too (press matl).  The gas's EqPress pressure follows
      //  from its restored state, but in a solid cell EqPress returns
      //  essentially the pressure it starts from, so starting from the seeded
      //  k=0 result adds the solid's dilatation pressure change once per
      //  outer iteration (nOuter 2 doubled the swing of a deforming wall's
      //  cell pressure and collapsed it to zero on unloading).
      if( m == 0 ){
        const int press_indx = d_ice->d_press_matl->get(0);
        pitchRestoreSolidCells<double>( patch, lb->press_CCLabel, press_indx, vol_frac, fSolid, pOldDW, subOldDW );
      }

      for( auto& model : d_ice->d_models ){
        FluidsBasedModel* fb_model = dynamic_cast<FluidsBasedModel*>( model );
        if( fb_model ){
          for( auto& tvar : fb_model->d_transVars ){
            if( tvar->matls->contains(indx) ){
              pitchRestoreSolidCells<double>( patch, tvar->var, indx, vol_frac, fSolid, pOldDW, subOldDW );
            }
          }
        }
      }
    }
  }

  d_pitchSched->setInitTimestep(false);
}


/* _____________________________________________________________________
 Task:      SPICE::pitchOuterLoop
 Purpose:   Drive d_pitchSched through outer iterations k=1..nOuter-1,
            seeded from the main graph's own k=0 primitives, and copy the
            final iterate's primitives back.  Follows the structure of
            Uintah_TS's ICE::sdcSweep (DW slot mapping, recompile, seeding
            slot 3 via setInitTimestep(true)+advanceDataWarehouse+
            transferFrom, per-iteration advanceDataWarehouse, re-putting
            timeStep/simTime/delT bracketed by setInitTimestep, ScrubNone,
            recomputeTimeStep/abortTimeStep forwarding, copy-back).
 _____________________________________________________________________  */
void
SPICE::pitchOuterLoop( const ProcessorGroup * pg,
                      const PatchSubset    * patches,
                      const MaterialSubset * /*matls*/,
                      DataWarehouse        * ParentOldDW,
                      DataWarehouse        * ParentNewDW,
                      LevelP                 level )
{
  if( d_pitchOuter < 2 ){
    return;
  }

  // The k=0 pass already asked to recompute/abort this step: skip the outer
  // iterations (the step is discarded and its state may be invalid).  All
  // ranks must agree, since the sub-graph execute is collective.
  int localSkip = ( ParentNewDW->recomputeTimeStep() || ParentNewDW->abortTimeStep() ) ? 1 : 0;
  int skip      = 0;
  Uintah::MPI::Allreduce( &localSkip, &skip, 1, MPI_INT, MPI_MAX, pg->getComm() );
  if( skip ){
    return;
  }

  const GridP grid = level->getGrid();

  DataWarehouse::ScrubMode ParentOldDW_scrubmode =
                           ParentOldDW->setScrubbing(DataWarehouse::ScrubNone);
  DataWarehouse::ScrubMode ParentNewDW_scrubmode =
                           ParentNewDW->setScrubbing(DataWarehouse::ScrubNone);

  const PatchSubset* patch_sub = patches;
  const MaterialSet* ice_matls = m_materialManager->allMaterials( "ICE" );
  const MaterialSubset* ice_mss = ice_matls->getUnion();

  d_pitchSched->setParentDWs(ParentOldDW, ParentNewDW);
  d_pitchSched->advanceDataWarehouse(grid);

  if( d_recompilePitchSched ){
    d_pitchSched->initialize(3, 1);
    // matches scheduleTimeAdvance's own ordering (maxMach BEFORE the
    // chain): computesRequires_CustomBCs wires maxMach_<face> into several
    // of scheduleFluidChain's own tasks when usingLodi, so it must be
    // computed in THIS (sub-graph) NewDW every recompile, not reused from
    // the main graph's k=0 value.
    d_ice->scheduleMaxMach_on_Lodi_BC_Faces(   d_pitchSched, level,  ice_matls );
    if( d_pitchChainHook ){
      d_pitchChainHook( d_pitchSched, level );   // FSPICE: gas-only chain, frozen solid
    } else {
      scheduleFluidChain(               d_pitchSched, level,  true );
    }
    schedulePitchBlend(                d_pitchSched, level,  ice_matls );
    // "afterAdvection" (not a literal "pitchOuterLoop" string):
    // conservedtoPrimitive_Vars gates its computesVar(temp_CC/vel_CC/
    // mach/species) on this exact string.  ice_matls (not all_matls): in
    // FSPICE the MPM materials must not get temp_CC/vel_CC/mach computed.
    d_ice->scheduleConservedtoPrimitive_Vars(  d_pitchSched, level->eachPatch(),
                                        ice_mss, ice_matls, "afterAdvection" );
    schedulePitchResidual(             d_pitchSched, level,  ice_matls );
    d_pitchSched->compile();
    d_recompilePitchSched = false;
  }

  // prime slot 3 and seed it with u^0 == u^n primitives (k=0 IS the main
  // graph's own, already-computed result)
  d_pitchSched->setInitTimestep(true);
  d_pitchSched->advanceDataWarehouse(grid);

  DataWarehouse* seedDW = d_pitchSched->get_dw(3);

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

  d_pitchSched->setInitTimestep(false);

  double residual = 0.0;

  for (int k = 1; k < d_pitchOuter; k++) {

    d_pitchSched->advanceDataWarehouse(grid);

    DataWarehouse* subOldDW = d_pitchSched->get_dw(2);
    DataWarehouse* subNewDW = d_pitchSched->get_dw(3);

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

    d_pitchSched->setInitTimestep(true);
    subOldDW->put( timeStepVar, lb->timeStepLabel );
    subOldDW->put( simTimeVar,  lb->simulationTimeLabel );
    subOldDW->put( delTVar,     lb->delTLabel, level.get_rep() );
    d_pitchSched->setInitTimestep(false);

    if( subOldDW_wasFinalized ){
      subOldDW->refinalize();
    }

    subOldDW->setScrubbing(DataWarehouse::ScrubNone);
    subNewDW->setScrubbing(DataWarehouse::ScrubNone);

    if( d_ice->d_with_mpm ){
      // FSPICE: cells containing solid restart from their time-n state every
      // outer iteration (solid is frozen at n), so pitchBlend gives them
      // (1-theta)*X^0 + theta*X_sub: conservative and positive for any theta.
      pitchRestartSolidCells( patch_sub, ice_mss, ParentOldDW, ParentNewDW, subOldDW );
    }

    d_pitchSched->execute();

    max_vartype residualVar;
    subNewDW->get( residualVar, Slb->pitchResidualLabel );
    residual = residualVar;

    if( pg->myRank() == 0 ){
      DOUT( m_spice_pitch, "  PITCH outer iteration " << k << " of "
            << (d_pitchOuter - 1) << "   timestep " << timeStepVar
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

    if( d_pitchTol > 0.0 && residual < d_pitchTol ){
      break;
    }
  }

  // copy the final outer iterate's primitives back to the main graph
  DataWarehouse* subNewDW = d_pitchSched->get_dw(3);

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

