
#ifndef UINTAH_HOMEBREW_ProblemSpecP_H
#define UINTAH_HOMEBREW_ProblemSpecP_H

namespace Uintah {
    namespace Grid {
	template<class T> class Handle;
    }
    namespace Interface {
	class ProblemSpec;
	typedef Uintah::Grid::Handle<ProblemSpec> ProblemSpecP;
    }
}

#endif
