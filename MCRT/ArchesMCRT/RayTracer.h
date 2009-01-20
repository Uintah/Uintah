//----- BackwardMCRTSolver.h  --------------------------------------------------

#ifndef Uintah_Component_Arches_RayTracer_h
#define Uintah_Component_Arches_RayTracer_h

/**
* @class RayTracer
* @author Xiaojing Sun
* @date Dec 11, 2008
*
* @brief Backward(Reverse) MonteCarlo Ray-Tracer for Radiation Heat Transfer
*        
*
*/



#include <Packages/Uintah/CCA/Components/Arches/Radiation/RadiationModel.h>
#include <Packages/Uintah/CCA/Components/Arches/Radiation/RadiationSolver.h>

namespace Uintah {

  /** @brief a ray initially emitted from a real surface */
  // a ray can be emitted from any part of a surface(boundary, entire or partial)
  void traceRayEmittedFromSurf();

  /** @brief a ray initially emitted from a control volume (medium) */
  // a ray can be emitted from any part of a control volume(media)
  void traceRayEmittedFromVol();
