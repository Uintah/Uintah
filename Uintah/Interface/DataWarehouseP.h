
#ifndef UINTAH_HOMEBREW_DataWarehouseP_H
#define UINTAH_HOMEBREW_DataWarehouseP_H

namespace Uintah {
    namespace Grid {
	template<class T> class Handle;
    }
    namespace Interface {
	class DataWarehouse;
	typedef Uintah::Grid::Handle<DataWarehouse> DataWarehouseP;
    }
}

#endif
