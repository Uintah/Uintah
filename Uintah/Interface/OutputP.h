
#ifndef UINTAH_HOMEBREW_OutputP_H
#define UINTAH_HOMEBREW_OutputP_H

namespace Uintah {
    namespace Grid {
	template<class T> class Handle;
    }
    namespace Interface {
	class Output;
	typedef Uintah::Grid::Handle<Output> OutputP;
    }
}

#endif
