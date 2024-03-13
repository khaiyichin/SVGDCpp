project(SVGDCpp)

set(CMAKE_CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.15)

# Define variables
set(SVGDCPP_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib/cmake/${PROJECT_NAME}")
set(SVGDCPP_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}") # assign include destination to variable

# Build library
find_package(Eigen3 CONFIG REQUIRED)

add_library(${PROJECT_NAME}
    INTERFACE
)
target_include_directories(${PROJECT_NAME}
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/>
)
target_link_libraries(${PROJECT_NAME}
    INTERFACE
    cppad_lib
    Eigen3::Eigen
)

# Install library header files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${SVGDCPP_INCLUDE_INSTALL_DIR}
)

# Associate library target with an export target
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
)

# Install export target
install(EXPORT ${PROJECT_NAME}Targets
    DESTINATION "${SVGDCPP_CMAKE_INSTALL_DIR}" # creates ${CMAKE_INSTALL_PREFIX}/lib/cmake/${PROJECT_NAME}Targets.cmake
    NAMESPACE SVGDCpp::
)

# Configure config file
include(CMakePackageConfigHelpers)

configure_package_config_file(cmake/SVGDCppConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION "${SVGDCPP_CMAKE_INSTALL_DIR}"
    PATH_VARS SVGDCPP_INCLUDE_INSTALL_DIR
)

# Create config files
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}"
    COMPATIBILITY AnyNewerVersion
)

# Install config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION "${SVGDCPP_CMAKE_INSTALL_DIR}"
)