# Create GMM example executable
add_executable(gmm_example gaussian_mixture_model/gmm_example.cpp)

target_link_libraries(gmm_example
    PRIVATE
    ${PROJECT_NAME}
)

install(TARGETS gmm_example
    DESTINATION ${PROJECT_SOURCE_DIR}/examples/gaussian_mixture_model/
)

# Create MVN example executable
add_executable(mvn_example multivariate_normal/mvn_example.cpp)

target_link_libraries(mvn_example
    PRIVATE
    ${PROJECT_NAME}
)

install(TARGETS mvn_example
    DESTINATION ${PROJECT_SOURCE_DIR}/examples/multivariate_normal/
)