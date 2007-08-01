//----- ExtraScalarSrc.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/ExtraScalarSrc.h>

using namespace std;
using namespace Uintah;

//****************************************************************************
// Interface constructor for ExtraScalarSrc
//****************************************************************************
ExtraScalarSrc::ExtraScalarSrc(const ArchesLabel* label, 
			       const MPMArchesLabel* MAlb):
                               d_lab(label), d_MAlab(MAlb)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
ExtraScalarSrc::~ExtraScalarSrc()
{
}

