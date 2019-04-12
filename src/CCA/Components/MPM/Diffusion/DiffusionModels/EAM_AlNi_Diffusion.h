/*
 * EAM_AlNi_Diffusion.h
 *
 *  Created on: Jan 29, 2019
 *      Author: jbhooper
 */

#ifndef CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONMODELS_EAM_ALNI_DIFFUSION_H_
#define CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONMODELS_EAM_ALNI_DIFFUSION_H_

#include <CCA/Components/MPM/Diffusion/FunctionInterpolators/FunctionInterpolator.h>

#include <tuple>
#include <algorithm>

namespace Uintah {

  enum EAM_AlNi_Region {AlRich, NiRich};

  class EAM_AlNi
  {
    public:
      static inline double getLiquidusAl(const double TinC) {
        return ((12590.1201379667 + TinC*(-57.8687048043292 + TinC*0.0298748335182435))
                /(TinC-1679.0));
      }

      static inline double getLiquidusNi(const double TinC)
      {
        // For now include the entire Ni region since it hasn't been fit yet.
        return 1.0;
      }

      static inline double getSolidusAl(const double TinC) {
        return ((-52405.0313512151 + TinC*(14.9434353522693 + TinC*0.00964488529853503))
                /(TinC-1679.0));

      }

      static inline double getSolidusNi(const double TinC) {
        // For now include the entire Ni region since it hasn't been fit yet.
        return 0.5;
      }

      static inline double getLiquidus( const double          Temp
                                      , const EAM_AlNi_Region phaseSide )
      {
        double liquidusConcentration = 0.0;
        // Convert Temp from K to C, and clamp at 600C if < 600C
        double T_Celsius = std::max(600.0,Temp-273.15);
//        double T_Celsius = ((Temp - 273.15) < 600 ? 600 : (Temp - 273.15));
        double denomInv = 1.0/(T_Celsius - 1679.0);
        if (phaseSide == EAM_AlNi_Region::AlRich) {
           liquidusConcentration = denomInv *
             (12590.1201379667 +
                 T_Celsius*(-57.8687048043292 + T_Celsius*0.0298748335182435));
        } else { // Ni Rich Region
           liquidusConcentration = 1.0; // For now include the entire Ni region since it hasn't been fit
        }

        return(liquidusConcentration);
      }

      static inline double getSolidus( const double           Temp
                                     , const EAM_AlNi_Region  phaseSide )
      {
        double solidusConcentration = 0.0;
        // Convert Temp from K to C, and clamp at 600C if < 600C
        double T_Celsius = ((Temp - 273.15) < 600 ? 600 : (Temp - 273.15));
        double denomInv = 1.0/(T_Celsius - 1679.0);
        if (phaseSide == EAM_AlNi_Region::AlRich) {
          solidusConcentration = denomInv *
              (-52405.0313512151 +
                  T_Celsius * (14.9434353522693 + T_Celsius * 0.00964488529853503));
        } else { // Ni Rich region
          solidusConcentration = 0.5; // For now include the entire Ni region since it hasn't been fit.
        }

        return(solidusConcentration);
      }

      static inline double Diffusivity(const double & T
                                      ,const double & C
                                      ,const Vector & gradC
                                      ,const double & minGuestConc
                                      ,const EAM_AlNi_Region & regionType
                                      ,const FunctionInterpolator * interpolator
                                      ,const double & D0_liquid
                                      ,const double & D0_solid) // Input temp & conc
      {
          bool NiAllLiquid = true;
          // Hack flag for the time being to set the Nickle rich portion of the phase
          //   diagram to always be liquid-like.
          double RTinv = 1.0/(8.3144598*T); // 1/RT with R in J/(mol*K)
          double TinC = std::max(600.0, T-273.15);
          // Value from our paper.
          double E0_liquid = 68000.0; // Activation energy for diffusion in liquid Al state (J/mol)

          // For now, leave the same curve and just constantly depress the solid state diffusivity by a
          //   multiplicative factor via D0_solid.
          double E0_solid = 68000.0; // Activation energy for diffusion in solid B2 AlNi

//          double liqConc = getLiquidus(T,regionType);
//          double solConc = getSolidus(T,regionType);
          double D_liq = D0_liquid * std::exp(-E0_liquid * RTinv);
          double D_sol = D0_solid  * std::exp(-E0_solid * RTinv);
          // Assuming the concentration of the guest is properly passed in, minConcReached tells
          //   us that the mininum overall system concentration is at least high enough to have
          //   reached the BEGINNING of the solidification region if it is greater than the
          //   liquidus concentration.
          double liqConc, solConc;
          bool minConcReached;
          typedef std::tuple<double,double> interpPoint;
          interpPoint leftPoint, rightPoint;
          if (regionType == EAM_AlNi_Region::AlRich) {
            liqConc = getLiquidusAl(TinC);
            solConc = getSolidusAl(TinC);
            leftPoint = std::make_tuple(liqConc,D_liq);
            rightPoint = std::make_tuple(solConc, D_sol);
          } else // Ni rich
          {
            solConc = getSolidusNi(TinC);
            liqConc = getLiquidusNi(TinC);
            rightPoint = std::make_tuple(liqConc, D_liq);

            // TODO FIXME JBH:  We should probably invert the concentration (from C_Ni to C_Al) and then
            //   we could use the same type of interpolator along the Ni extrema as well.
            if (NiAllLiquid) {
              leftPoint = std::make_tuple(solConc, D_liq);
            } else {
              leftPoint = std::make_tuple(solConc, D_sol);
            }
          }
          minConcReached = (minGuestConc > liqConc);
          double D_out, C_out;
          std::tie(C_out,D_out) = interpolator->interpolate(leftPoint,rightPoint,C, gradC, minConcReached);
          return(D_out);
      }
  };
}




#endif /* CCA_COMPONENTS_MPM_DIFFUSION_DIFFUSIONMODELS_EAM_ALNI_DIFFUSION_H_ */
