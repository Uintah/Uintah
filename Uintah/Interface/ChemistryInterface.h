
#ifndef UINTAH_HOMEBREW_ChemistryInterface_H
#define UINTAH_HOMEBREW_ChemistryInterface_H

#include "ChemistryInterfaceP.h"
#include "Handle.h"
#include "RefCounted.h"

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
