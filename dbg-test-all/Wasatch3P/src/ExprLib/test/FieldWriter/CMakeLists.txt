nebo_cuda_prep_dir()

nebo_add_executable( fieldWriterTest FieldWriterTest.cpp )
target_link_libraries( fieldWriterTest exprlib ${TPL_LIBRARIES} )
add_test( field_writer fieldWriterTest )
