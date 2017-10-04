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

#include <Core/Grid/SimulationTime.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/StringUtil.h>

#include <cfloat>
#include <climits>
#include <string>
#include <iostream>

using namespace Uintah;

SimulationTime::SimulationTime( const ProblemSpecP & params )
{
  m_delt_factor = 1.0;
  
  ProblemSpecP time_ps = params->findBlock( "Time" );
  time_ps->require( "maxTime", m_max_time );
  time_ps->require( "initTime", m_init_time );
  time_ps->require( "delt_min", m_delt_min );
  time_ps->require( "delt_max", m_delt_max );
  time_ps->require( "timestep_multiplier", m_delt_factor );

  if( !time_ps->get( "delt_init", m_max_initial_delt) &&
      !time_ps->get("max_initial_delt", m_max_initial_delt ) ) {
    m_max_initial_delt = 0;
  }
  
  if( !time_ps->get( "initial_delt_range", m_initial_delt_range ) ) {
    m_initial_delt_range = 0;
  }
  else if( m_initial_delt_range < 0 ) {
    std::cerr << "Negative initial_delt_range is not allowed.\n";
    std::cerr << "resetting to 0 (i.e. the value is ignored)\n";
    m_initial_delt_range = 0;
  }
  
  if( !time_ps->get( "max_delt_increase", m_max_delt_increase ) ) {
    m_max_delt_increase = 0;
  }
  else if( m_max_wall_time < 0 ) {
    std::cerr << "Negative max_wall_time is not allowed.\n";
    std::cerr << "resetting to 0 (i.e. the value is ignored)\n";
    m_max_delt_increase = 0;
  }
    
  if( !time_ps->get( "override_restart_delt", m_override_restart_delt) ) {
    m_override_restart_delt = 0.0;
  }
  else if( m_max_timestep < 0 ) {
    std::cerr << "Negative override_restart_delt is not allowed.\n";
    std::cerr << "resetting to 0 (i.e. the value is ignored)\n";
    m_max_timestep = 0;
  }

  if( !time_ps->get( "max_Timesteps", m_max_timestep ) ) {
    m_max_timestep = 0;
  }
  else if( m_max_timestep < 0 ) {
    std::cerr << "Negative maxTimesteps is not allowed.\n";
    std::cerr << "resetting to 0 (i.e. the value is ignored)\n";
    m_max_timestep = 0;
  }
  
  if( !time_ps->get( "max_wall_time", m_max_wall_time ) ) {
    m_max_wall_time = 0;
  }
  else if( m_max_wall_time < 0 ) {
    std::cerr << "Negative max_wall_time is not allowed.\n";
    std::cerr << "resetting to 0 (i.e. the value is ignored)\n";
    m_max_wall_time = 0;
  }
  
  if ( !time_ps->get( "clamp_time_to_output", m_clamp_time_to_output ) ) {
    m_clamp_time_to_output = false;
  }
  if ( !time_ps->get( "end_at_max_time_exactly", m_end_at_max_time ) ) {
    m_end_at_max_time = false;
  }

  if ( time_ps->get( "clamp_timestep_to_output", m_clamp_time_to_output ) ) {
    throw ProblemSetupException("ERROR SimulationTime \n"
				"clamp_timestep_to_output has been deprecated "
				"and has been replaced by clamp_time_to_output.",
                                __FILE__, __LINE__);
  }

  if ( time_ps->get( "end_on_max_time_exactly", m_end_at_max_time ) ) {
    throw ProblemSetupException("ERROR SimulationTime \n"
				"end_on_max_time_exactly has been deprecated "
				"and has been replaced by end_at_max_time_exactly.",
                                __FILE__, __LINE__);
  }
}

//__________________________________
//  This only called by the switcher component

void
SimulationTime::problemSetup( const ProblemSpecP & params )
{
  proc0cout << "  Reading <Time> section from: " <<
  Uintah::basename(params->getFile()) << "\n";
  ProblemSpecP time_ps = params->findBlock("Time");
  time_ps->require("delt_min", m_delt_min);
  time_ps->require("delt_max", m_delt_max);
  time_ps->require("timestep_multiplier", m_delt_factor);
  
  if( !time_ps->get("delt_init", m_max_initial_delt) &&
      !time_ps->get("max_initial_delt", m_max_initial_delt) ) {
    m_max_initial_delt = 0;
  }

  if( !time_ps->get("initial_delt_range", m_initial_delt_range) ) {
    m_initial_delt_range = 0;
  }
  else if( m_initial_delt_range < 0 ) {
    std::cerr << "Negative initial_delt_range is not allowed.\n";
    std::cerr << "resetting to 0 (i.e. the value is ignored)\n";
    m_initial_delt_range = 0;
  }

  if( !time_ps->get("max_delt_increase", m_max_delt_increase) ) {
    m_max_delt_increase = 0;
  }
  else if( m_max_wall_time < 0 ) {
    std::cerr << "Negative max_wall_time is not allowed.\n";
    std::cerr << "resetting to 0 (i.e. the value is ignored)\n";
    m_max_delt_increase = 0;
  }
  
  time_ps->get( "override_restart_delt", m_override_restart_delt);
}
