add_library( TestSuite Suite.cc SuiteTree.cc Test.cc )
target_link_libraries( TestSuite Core_Exceptions )
# jcs Z_LIBRARY