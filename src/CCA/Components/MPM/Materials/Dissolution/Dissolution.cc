/*
 * Copyright © 2026 by Geocosm LLC
 */

#include <CCA/Components/MPM/Materials/Dissolution/Dissolution.h>

using namespace Uintah;
using namespace std;

Dissolution::Dissolution(const ProcessorGroup* myworld, MPMLabel* Mlb, 
                         ProblemSpecP ps, MPMFlags* flags)
                         : lb(Mlb), flag(flags)
{
}

Dissolution::~Dissolution()
{
}

void Dissolution::setTemperature(double BHTemp)
{
  d_temperature = BHTemp;
}

void Dissolution::setPhase(std::string LCPhase)
{
  // phase is "ramp", "settle",  "hold" or "dissolution"  Only do dissolution
  // during the "dissolution" phase

  d_phase = LCPhase;
}

void Dissolution::setTimeConversionFactor(const double tcf)
{
  // This is the factor to convert 1 Uintah time unit (probably a
  // microsecond) to a Ma.  i.e., if tcf=10, 1 microsecond = 10 Ma
  d_timeConversionFactor = tcf;
}

void Dissolution::setGrowthFractionRate(const double QGVF)
{
  // Rate at which the growth vector is achieved at this load level
  d_growthFractionRate = QGVF;
}
