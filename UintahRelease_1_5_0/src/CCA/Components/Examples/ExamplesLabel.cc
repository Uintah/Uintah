/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <CCA/Components/Examples/ExamplesLabel.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Disclosure/TypeUtils.h>

namespace Uintah {

ExamplesLabel::ExamplesLabel()
{
  // For SimpleCFD
  bctype = VarLabel::create("bctype",
			    NCVariable<int>::getTypeDescription(),
			    IntVector(1,1,1));
  xvelocity = VarLabel::create("xvelocity",
			       SFCXVariable<double>::getTypeDescription(),
			       IntVector(1,1,1));
  yvelocity = VarLabel::create("yvelocity",
			       SFCYVariable<double>::getTypeDescription(),
			       IntVector(1,1,1));
  zvelocity = VarLabel::create("zvelocity",
			       SFCZVariable<double>::getTypeDescription(),
			       IntVector(1,1,1));
  density = VarLabel::create("density",
			     CCVariable<double>::getTypeDescription(),
			     IntVector(1,1,1));
  temperature = VarLabel::create("temperature",
				 CCVariable<double>::getTypeDescription(),
				 IntVector(1,1,1));
  xvelocity_matrix = VarLabel::create("xvelocity_matrix",
				      SFCXVariable<Stencil7>::getTypeDescription());
  xvelocity_rhs = VarLabel::create("xvelocity_rhs",
				   SFCXVariable<double>::getTypeDescription());
  yvelocity_matrix = VarLabel::create("yvelocity_matrix",
				      SFCYVariable<Stencil7>::getTypeDescription());
  yvelocity_rhs = VarLabel::create("yvelocity_rhs",
				   SFCYVariable<double>::getTypeDescription());
  zvelocity_matrix = VarLabel::create("zvelocity_matrix",
				      SFCZVariable<Stencil7>::getTypeDescription());
  zvelocity_rhs = VarLabel::create("zvelocity_rhs",
				   SFCZVariable<double>::getTypeDescription());
  density_matrix = VarLabel::create("density_matrix",
				    CCVariable<Stencil7>::getTypeDescription());
  density_rhs = VarLabel::create("density_rhs",
				 CCVariable<double>::getTypeDescription());
  temperature_matrix = VarLabel::create("temperature_matrix",
					CCVariable<Stencil7>::getTypeDescription());
  temperature_rhs = VarLabel::create("temperature_rhs",
				     CCVariable<double>::getTypeDescription());

  pressure_matrix = VarLabel::create("pressure_matrix",
				     CCVariable<Stencil7>::getTypeDescription());
  pressure_rhs = VarLabel::create("pressure_rhs",
				  CCVariable<double>::getTypeDescription());
  pressure = VarLabel::create("pressure",
			      CCVariable<double>::getTypeDescription());

  ccvelocity = VarLabel::create("ccvelocity",
				CCVariable<Vector>::getTypeDescription());
  ccvorticity = VarLabel::create("ccvorticity",
				 CCVariable<Vector>::getTypeDescription());
  ccvorticitymag = VarLabel::create("ccvorticitymag",
				    CCVariable<double>::getTypeDescription());
  vcforce = VarLabel::create("vcforce",
			     CCVariable<Vector>::getTypeDescription());
  NN = VarLabel::create("N",
			CCVariable<Vector>::getTypeDescription());

  // For AMRSimpleCFD
  pressure2_matrix = VarLabel::create("pressure2_matrix",
				      CCVariable<Stencil7>::getTypeDescription());
  pressure2_rhs = VarLabel::create("pressure2_rhs",
				  CCVariable<double>::getTypeDescription());
  pressure2 = VarLabel::create("pressure2",
			       CCVariable<double>::getTypeDescription());

  density_gradient_mag = VarLabel::create("density_gradient_magnitude",
					  CCVariable<double>::getTypeDescription());
  temperature_gradient_mag = VarLabel::create("temperature_gradient_magnitude",
					      CCVariable<double>::getTypeDescription());
  pressure_gradient_mag = VarLabel::create("pressure_gradient_magnitude",
					   CCVariable<double>::getTypeDescription());

  pXLabel = VarLabel::create("p.x",
                             ParticleVariable<Point>::getTypeDescription() );

  pXLabel_preReloc = VarLabel::create( "p.x+",
			ParticleVariable<Point>::getTypeDescription(),
			IntVector(0,0,0),
			VarLabel::PositionVariable);

  pParticleIDLabel = VarLabel::create("p.particleID",
			ParticleVariable<long64>::getTypeDescription() );

  pParticleIDLabel_preReloc = VarLabel::create("p.particleID+",
			ParticleVariable<long64>::getTypeDescription() );

  pMassLabel_preReloc = VarLabel::create( "p.mass+",
			ParticleVariable<double>::getTypeDescription() );

  pMassLabel = VarLabel::create( "p.mass",
			ParticleVariable<double>::getTypeDescription() );



}

ExamplesLabel::~ExamplesLabel()
{
  VarLabel::destroy(bctype);
  VarLabel::destroy(xvelocity);
  VarLabel::destroy(yvelocity);
  VarLabel::destroy(zvelocity);
  VarLabel::destroy(density);
  VarLabel::destroy(temperature);
  VarLabel::destroy(xvelocity_matrix);
  VarLabel::destroy(xvelocity_rhs);
  VarLabel::destroy(yvelocity_matrix);
  VarLabel::destroy(yvelocity_rhs);
  VarLabel::destroy(zvelocity_matrix);
  VarLabel::destroy(zvelocity_rhs);
  VarLabel::destroy(density_matrix);
  VarLabel::destroy(density_rhs);
  VarLabel::destroy(temperature_matrix);
  VarLabel::destroy(temperature_rhs);
  VarLabel::destroy(pressure_matrix);
  VarLabel::destroy(pressure_rhs);
  VarLabel::destroy(pressure);
  VarLabel::destroy(ccvelocity);
  VarLabel::destroy(ccvorticity);
  VarLabel::destroy(ccvorticitymag);
  VarLabel::destroy(vcforce);
  VarLabel::destroy(NN);
  VarLabel::destroy(pressure2);
  VarLabel::destroy(pressure2_matrix);
  VarLabel::destroy(pressure2_rhs);
  VarLabel::destroy(pressure_gradient_mag);
  VarLabel::destroy(temperature_gradient_mag);
  VarLabel::destroy(density_gradient_mag);
  VarLabel::destroy(pXLabel);
  VarLabel::destroy(pXLabel_preReloc);
  VarLabel::destroy(pMassLabel);
  VarLabel::destroy(pMassLabel_preReloc);
  VarLabel::destroy(pParticleIDLabel);
  VarLabel::destroy(pParticleIDLabel_preReloc);
}

} // end namespace uintah 

