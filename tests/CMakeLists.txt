# Define common source for test files (required by doctest)
set(DOCTEST_MAIN_SRC doctest_main.cpp)

# Add model test
add_executable(test_model
    ${DOCTEST_MAIN_SRC}
    test_model.cpp
)

target_link_libraries(test_model
    ${PROJECT_NAME}
)

add_test(NAME TestModelClass COMMAND test_model)

# Add kernel test
add_executable(test_kernel
    ${DOCTEST_MAIN_SRC}
    test_kernel.cpp
)

target_link_libraries(test_kernel
    ${PROJECT_NAME}
)

add_test(NAME TestKernelClass COMMAND test_kernel)