add_library( TestBoxGrouper
        TestBoxGrouper.cc
        Box.cc
        BoxRangeQuerier.cc )
target_link_libraries( TestBoxGrouper PUBLIC Core_Exceptions Core_Geometry Core_Containers TestSuite )
