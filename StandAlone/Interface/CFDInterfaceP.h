
#ifndef UINTAH_HOMEBREW_CFDInterfaceP_H
#define UINTAH_HOMEBREW_CFDInterfaceP_H

namespace Uintah {
    namespace Grid {
	template<class T> class Handle;
    }
    namespace Interface {
	class CFDInterface;
	typedef Uintah::Grid::Handle<CFDInterface> CFDInterfaceP;
    }
}

#endif
