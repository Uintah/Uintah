
#ifndef UINTAH_HOMEBREW_ChemistryInterface_H
#define UINTAH_HOMEBREW_ChemistryInterface_H

#include <Packages/Uintah/Interface/ChemistryInterfaceP.h>
#include <Packages/Uintah/Grid/Handle.h>
#include <Packages/Uintah/Grid/RefCounted.h>

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
