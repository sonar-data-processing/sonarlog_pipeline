cmake_minimum_required (VERSION 2.8)
project (sonarlog_pipeline)

include ( "${WORKSPACE_DIR}/base/cmake/Dependencies.cmake"  )
include ( "${WORKSPACE_DIR}/third_party/cmake/Gnuplot.cmake"  )

add_project_dependency (base                  ${WORKSPACE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
add_project_dependency (rock_util             ${WORKSPACE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
add_project_dependency (sonar_processing ${WORKSPACE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

setup_headers ( sonar_processing ${sonar_processing_SOURCE_DIR} )
setup_headers ( ${PROJECT_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/src" )

include ( FindPkgConfig  )
include ( BoostTest )

find_package ( Boost COMPONENTS system filesystem program_options iostreams unit_test_framework REQUIRED )
find_package ( Eigen3 REQUIRED )
find_package ( OpenCV REQUIRED )

set (
    ${PROJECT_NAME}_INCLUDE_DIR
    ${PROJECT_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/include
    CACHE PATH "Sonar's log Target Tracking Include Directory"
    FORCE
)

file ( GLOB SRCS RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/*.cpp )
list ( REMOVE_ITEM SRCS "src/main.cpp" )

set (
    SRCS
    ${SRCS}
    ${sonar_processing_SOURCES}
)

set ( SRCS ${SRCS} ${sonar_processing_SOURCES} )

pkg_check_modules(pocolog_cpp pocolog_cpp)

include_directories (
    ${sonarlog_pipeline_INCLUDE_DIR}
    ${base_INCLUDE_DIR}
    ${sonar_util_INCLUDE_DIR}
    ${rock_util_INCLUDE_DIR}
    ${sonar_processing_INCLUDE_DIR}
    ${gnuplot_INCLUDE_DIR}
    ${ROCK_ROBOTICS_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

link_directories (
    ${ROCK_ROBOTICS_LIBRARY_DIR}
)

set (
    LIBS
    rock_util
    base
    ${Boost_LIBRARIES}
    ${pocolog_cpp_LIBRARIES}
    ${OpenCV_LIBS}
)

add_executable (
    sonarlog_pipeline
    src/main.cpp
    ${SRCS}
)

target_link_libraries (
    sonarlog_pipeline
    ${LIBS}
)

if (COMPILE_EXAMPLES)
    file ( GLOB EXAMPLES "${PROJECT_SOURCE_DIR}/examples/*.cpp")

    foreach(FILENAME ${EXAMPLES})
        get_filename_component (FILE ${FILENAME} NAME)
        get_filename_component (FILE_WE ${FILENAME} NAME_WE)
        add_executable (${FILE_WE} examples/${FILE} ${SRCS})
        target_link_libraries (${FILE_WE} ${LIBS})
    endforeach()

endif(COMPILE_EXAMPLES)

boost_test_init()

add_boost_test (
    test_ArgumentParser
    SOURCES test/test_ArgumentParser.cpp src/ArgumentParser.cpp
    LIBRARIES ${Boost_LIBRARIES}
)

configure_file (
    ${PROJECT_SOURCE_DIR}/scripts/example0.sh.in
    ${PROJECT_BINARY_DIR}/scripts/example0.sh
    @ONLY
)

configure_file (
    ${PROJECT_SOURCE_DIR}/scripts/example1.sh.in
    ${PROJECT_BINARY_DIR}/scripts/example1.sh
    @ONLY
)

configure_file (
    ${PROJECT_SOURCE_DIR}/scripts/example2.sh.in
    ${PROJECT_BINARY_DIR}/scripts/example2.sh
    @ONLY
)
