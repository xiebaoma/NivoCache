cmake_minimum_required(VERSION 3.20)
project(NivoCache LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)
set(nivo_top_level OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(nivo_top_level ON)
endif()
option(NIVO_BUILD_TESTS "Build contract tests" ${nivo_top_level})
option(NIVO_BUILD_MEMORY_TESTS "Build RSS integration tests (run without sanitizers)" OFF)
option(NIVO_BUILD_EXAMPLES "Build examples" ${nivo_top_level})
option(NIVO_BUILD_BENCHMARKS "Build Google Benchmark performance tests" OFF)
option(NIVO_BENCHMARK_WITH_CACHELIB "Include CacheLib in performance tests" OFF)
option(NIVO_BENCHMARK_FETCH_GOOGLE_BENCHMARK
  "Fetch the pinned Google Benchmark instead of using an installed package" ON)
set(NIVO_BENCHMARK_CACHELIB_REVISION "unknown" CACHE STRING
  "CacheLib revision recorded in benchmark output")

add_library(nivo_cache src/cache.cpp)
add_library(Nivo::cache ALIAS nivo_cache)
set_target_properties(nivo_cache PROPERTIES EXPORT_NAME cache CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON)
target_compile_features(nivo_cache PUBLIC cxx_std_20)
target_include_directories(nivo_cache PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(nivo_cache PUBLIC Threads::Threads)
if(WIN32)
  target_link_libraries(nivo_cache PRIVATE psapi)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(nivo_cache PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
endif()

if(NIVO_BUILD_TESTS)
  enable_testing()
  add_executable(nivo_contract_tests tests/cache_test.cpp)
  target_link_libraries(nivo_contract_tests PRIVATE Nivo::cache)
  add_test(NAME nivo.contract COMMAND nivo_contract_tests)
  set_tests_properties(nivo.contract PROPERTIES TIMEOUT 60)
endif()
if(NIVO_BUILD_MEMORY_TESTS)
  enable_testing()
  add_executable(nivo_memory_rss_tests tests/memory_rss_test.cpp)
  target_link_libraries(nivo_memory_rss_tests PRIVATE Nivo::cache)
  foreach(scenario IN ITEMS slabs pinned reserve compaction compaction_pinned)
    add_test(NAME nivo.rss.${scenario} COMMAND nivo_memory_rss_tests ${scenario})
    set_tests_properties(nivo.rss.${scenario} PROPERTIES TIMEOUT 60 LABELS memory RUN_SERIAL TRUE)
  endforeach()
endif()
if(NIVO_BUILD_EXAMPLES)
  add_executable(nivo_basic examples/basic.cpp)
  target_link_libraries(nivo_basic PRIVATE Nivo::cache)
  if(NIVO_BUILD_TESTS)
    add_test(NAME nivo.example COMMAND nivo_basic)
  endif()
endif()
if(NIVO_BUILD_BENCHMARKS)
  if(NOT TARGET benchmark::benchmark)
    if(NIVO_BENCHMARK_FETCH_GOOGLE_BENCHMARK)
      include(FetchContent)
      set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
      set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
      FetchContent_Declare(google_benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG 31c66f47f60d06790a4885a261d682ba339bd100
        GIT_SHALLOW TRUE)
      FetchContent_MakeAvailable(google_benchmark)
    else()
      find_package(benchmark 1.8 CONFIG REQUIRED)
    endif()
  endif()

  add_executable(nivo_cache_bench benchmarks/src/cache_bench.cpp)
  target_link_libraries(nivo_cache_bench PRIVATE Nivo::cache benchmark::benchmark)

  execute_process(
    COMMAND git rev-parse --short=12 HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE nivo_benchmark_revision
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(NOT nivo_benchmark_revision)
    set(nivo_benchmark_revision unknown)
  endif()
  target_compile_definitions(nivo_cache_bench PRIVATE
    NIVO_BENCHMARK_NIVO_REVISION="${nivo_benchmark_revision}"
    NIVO_BENCHMARK_CACHELIB_REVISION="${NIVO_BENCHMARK_CACHELIB_REVISION}"
    NIVO_BENCHMARK_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")

  if(NIVO_BENCHMARK_WITH_CACHELIB)
    find_package(GTest CONFIG REQUIRED)
    find_package(NUMA REQUIRED)
    find_package(magic_enum CONFIG REQUIRED)
    find_package(cachelib CONFIG REQUIRED)
    get_filename_component(cachelib_include_root "${CACHELIB_INCLUDE_DIR}" DIRECTORY)
    target_include_directories(nivo_cache_bench SYSTEM PRIVATE
      "${cachelib_include_root}")
    target_compile_definitions(nivo_cache_bench PRIVATE
      NIVO_BENCHMARK_WITH_CACHELIB=1)
    if(TARGET cachelib::cachelib)
      target_link_libraries(nivo_cache_bench PRIVATE cachelib::cachelib)
    elseif(TARGET cachelib)
      target_link_libraries(nivo_cache_bench PRIVATE cachelib)
    else()
      message(FATAL_ERROR "The CacheLib package did not export a cachelib target")
    endif()
  else()
    target_compile_definitions(nivo_cache_bench PRIVATE
      NIVO_BENCHMARK_WITH_CACHELIB=0)
  endif()
endif()

set(nivo_package_dir ${CMAKE_INSTALL_LIBDIR}/cmake/NivoCache)
install(TARGETS nivo_cache EXPORT NivoCacheTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT NivoCacheTargets NAMESPACE Nivo:: DESTINATION ${nivo_package_dir})
configure_package_config_file(cmake/NivoCacheConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/NivoCacheConfig.cmake
  INSTALL_DESTINATION ${nivo_package_dir})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/NivoCacheConfig.cmake
  DESTINATION ${nivo_package_dir})
