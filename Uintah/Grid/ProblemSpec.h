
#ifndef UINTAH_HOMEBREW_ProblemSpec_H
#define UINTAH_HOMEBREW_ProblemSpec_H

#include "Handle.h"
#include "RefCounted.h"
class TypeDescription;

// This is the "base" problem spec.  There should be ways of breaking
// this up

class ProblemSpec : public RefCounted {
public:
    ProblemSpec();
    virtual ~ProblemSpec();

    double getStartTime() const;
    double getMaximumTime() const;

    static const TypeDescription* getTypeDescription();
private:
    ProblemSpec(const ProblemSpec&);
    ProblemSpec& operator=(const ProblemSpec&);
};

#endif
