
#ifndef UINTAH_HOMEBREW_ChemistryInterface_H
#define UINTAH_HOMEBREW_ChemistryInterface_H

#include <Packages/Uintah/CCA/Ports/ChemistryInterfaceP.h>
#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>

using namespace Uintah;

class ChemistryInterface : public RefCounted {
public:
    ChemistryInterface();
    virtual ~ChemistryInterface();

    // This is a fake method, since we don't know how to do this yet.
    virtual void calculateChemistryEffects() = 0;
private:
    ChemistryInterface(const ChemistryInterface&);
    ChemistryInterface& operator=(const ChemistryInterface&);
};

#endif
