#ifndef Uintah_Component_Arches_KokkosTools_h
#define Uintah_Component_Arches_KokkosTools_h

namespace Uintah{ namespace ArchesCore {


#define KOKKOS_INITIALIZE_TO_CONSTANT_EXTRA_CELL(phi, const) \
  Uintah::BlockRange range(patch->getExtraCellLowIndex(), patch->getExtraCellHighIndex() ); \
  Uintah::parallel_for( range, [&](int i, int j, int k){ \
    phi(i,j,k) = const; \
  });

#define KOKKOS_INITIALIZE_TO_CONSTANT_INTERIOR_CELL(phi, const) \
  Uintah::BlockRange range(patch->getCellLowIndex(), patch->getCellHighIndex() ); \
  Uintah::parallel_for( range, [&](int i, int j, int k){ \
    phi(i,j,k) = const; \
  });

}} //end namespace Uintah::ArchesCore
#endif
