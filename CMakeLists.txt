cmake_minimum_required(VERSION 3.15)
project(SVGDCpp)

set(CMAKE_CXX_STANDARD 17)

# Define variables
set(SVGDCPP_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/cmake")
set(SVGDCPP_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}") # assign include destination to variable
set(API_DOCUMENTATION_DIR ${CMAKE_INSTALL_PREFIX}/doc/${PROJECT_NAME})

# Build and install API documentation
if(CMAKE_BUILD_TYPE MATCHES Release OR CMAKE_CONFIGURATION_TYPES MATCHES Release)
    find_package(Doxygen)

    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN doc/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

        # Configure the file
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        message(STATUS "Doxygen build started")

        # Build doxygen documentation
        add_custom_target(Doxygen_documentation ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen" VERBATIM
        )

        # Install built documentation
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/html DESTINATION ${API_DOCUMENTATION_DIR})
    else()
        message(WARNING "Doxygen need to be installed to generate the doxygen documentation")
    endif()
endif()

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

# Install examples
add_subdirectory(examples)