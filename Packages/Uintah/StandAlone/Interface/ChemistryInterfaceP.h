
#ifndef UINTAH_HOMEBREW_ChemistryInterfaceP_H
#define UINTAH_HOMEBREW_ChemistryInterfaceP_H

namespace Uintah {
    namespace Grid {
	template<class T> class Handle;
    }
    namespace Interface {
	class ChemistryInterface;
	typedef Uintah::Grid::Handle<ChemistryInterface> ChemistryInterfaceP;
    }
}

#endif
