/**
 *  \file   ArchesParticlesHelper.h
 *  \date   June, 2014
 *  \author "Tony Saad"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2017 The University of Utah
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

#ifndef ArchesParticles_Helper_h
#define ArchesParticles_Helper_h

//-- stl includes --//
#include <string>

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>
#include <CCA/Components/Wasatch/ParticlesHelper.h>

// forward declarations
namespace Uintah{

  class ProcessorGroup;
  class DataWarehouse;
  class ParticlesHelper;
  class Arches; 

  /**
   *  \class  ArchesParticlesHelper
   *  \author Tony Saad
   *  \date   June, 2014
   *  \brief  This class provides support for creating particles.
   */
  class ArchesParticlesHelper: public Uintah::ParticlesHelper
  {
  public:
    
    ArchesParticlesHelper();
    ~ArchesParticlesHelper();

    /*
     *  \brief Initializes particle memory. This must be called at schedule initialize. DO NOT call
     it on restarts. Use schedule_restart_initialize instead
     */
    void schedule_initialize ( const Uintah::LevelP& level,
                               Uintah::SchedulerP& sched   );

    void sync_with_arches( Arches* const arches );
    
  private:
    void initialize( const Uintah::ProcessorGroup*,
                    const Uintah::PatchSubset* patches, const Uintah::MaterialSubset* matls,
                    Uintah::DataWarehouse* old_dw, Uintah::DataWarehouse* new_dw);

    Arches* arches_; 
    bool archesSync_; 

  };

} /* namespace Arches */
#endif /* ArchesParticles_Helper_H */
