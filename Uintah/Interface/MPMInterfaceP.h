
#ifndef UINTAH_HOMEBREW_MPMInterfaceP_H
#define UINTAH_HOMEBREW_MPMInterfaceP_H

namespace Uintah {
    namespace Grid {
	template<class T> class Handle;
    }
    namespace Interface {
	class MPMInterface;
	typedef Uintah::Grid::Handle<MPMInterface> MPMInterfaceP;
    }
}

#endif
