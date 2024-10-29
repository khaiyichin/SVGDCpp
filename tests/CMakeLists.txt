# Define common source for test files (required by doctest)
set(DOCTEST_MAIN_SRC doctest_main.cpp)

# Add Model test
add_executable(test_model
    ${DOCTEST_MAIN_SRC}
    test_model.cpp
)

target_link_libraries(test_model
    ${PROJECT_NAME}
)

add_test(NAME TestModelClass COMMAND test_model)

# Add Kernel test
add_executable(test_kernel
    ${DOCTEST_MAIN_SRC}
    test_kernel.cpp
)

target_link_libraries(test_kernel
    ${PROJECT_NAME}
)

add_test(NAME TestKernelClass COMMAND test_kernel)

# Add SVGD test
add_executable(test_svgd
    ${DOCTEST_MAIN_SRC}
    test_svgd.cpp
)

target_link_libraries(test_svgd
    ${PROJECT_NAME}
)

add_test(NAME TestSVGDClass COMMAND test_svgd)