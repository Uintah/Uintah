# 
# 
# The MIT License
# 
# Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
# Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
# University of Utah.
# 
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
# 
# 
# 
# 
# Makefile fragment for this subdirectory

SRCDIR := CCA/Components/MPM/ConstitutiveModel

SRCS   += \
        $(SRCDIR)/RigidMaterial.cc              \
        $(SRCDIR)/CompMooneyRivlin.cc           \
        $(SRCDIR)/ConstitutiveModelFactory.cc   \
        $(SRCDIR)/ConstitutiveModel.cc          \
        $(SRCDIR)/ImplicitCM.cc                 \
        $(SRCDIR)/MPMMaterial.cc                \
        $(SRCDIR)/CNH_MMS.cc                    \
        $(SRCDIR)/TransIsoHyper.cc              \
        $(SRCDIR)/TransIsoHyperImplicit.cc      \
        $(SRCDIR)/ViscoTransIsoHyper.cc         \
        $(SRCDIR)/ViscoTransIsoHyperImplicit.cc \
        $(SRCDIR)/ViscoScram.cc                 \
        $(SRCDIR)/ViscoSCRAMHotSpot.cc          \
        $(SRCDIR)/HypoElastic.cc                \
        $(SRCDIR)/HypoElasticImplicit.cc        \
        $(SRCDIR)/ViscoScramImplicit.cc         \
        $(SRCDIR)/MWViscoElastic.cc             \
        $(SRCDIR)/IdealGasMP.cc                 \
        $(SRCDIR)/Membrane.cc                   \
        $(SRCDIR)/ShellMaterial.cc              \
        $(SRCDIR)/ElasticPlastic.cc             \
        $(SRCDIR)/ElasticPlasticHP.cc           \
        $(SRCDIR)/Water.cc                      \
        $(SRCDIR)/ViscoPlastic.cc               \
        $(SRCDIR)/MurnaghanMPM.cc               \
        $(SRCDIR)/ProgramBurn.cc                \
        $(SRCDIR)/JWLppMPM.cc                   \
        $(SRCDIR)/UCNH.cc                       \
        $(SRCDIR)/P_Alpha.cc                    \
        $(SRCDIR)/SoilFoam.cc	             \
        $(SRCDIR)/NonLocalDruckerPrager.cc      \
        $(SRCDIR)/Arenisca.cc

ifneq ($(NO_FORTRAN),yes)
  SRCS   += \
       $(SRCDIR)/Diamm.cc                      \
       $(SRCDIR)/HypoElasticFortran.cc         \
       $(SRCDIR)/Kayenta.cc                    
endif

SUBDIRS := \
        $(SRCDIR)/PlasticityModels \

ifneq ($(NO_FORTRAN),yes)
  SUBDIRS += $(SRCDIR)/fortran
endif

include $(SCIRUN_SCRIPTS)/recurse.mk
