# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

INCLUDE(ExternalProject)

SET(CBLAS_SOURCES_DIR ${THIRD_PARTY_PATH}/openblas)
SET(CBLAS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/openblas)
IF(NOT WIN32)
    SET(CBLAS_LIBRARIES
        "${CBLAS_INSTALL_DIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}openblas${CMAKE_STATIC_LIBRARY_SUFFIX}"
        CACHE FILEPATH "openblas library." FORCE)
    SET(CBLAS_INC_DIR "${CBLAS_INSTALL_DIR}/include" CACHE PATH "openblas include directory." FORCE)
    SET(OPENBLAS_CC "${CMAKE_C_COMPILER} -Wno-unused-but-set-variable -Wno-unused-variable")
    SET(OPENBLAS_COMMIT "v0.2.20")

    IF(APPLE)
        SET(OPENBLAS_CC "${CMAKE_C_COMPILER} -isysroot ${CMAKE_OSX_SYSROOT}")
    ENDIF()
    SET(OPTIONAL_ARGS "")
    IF(CMAKE_SYSTEM_PROCESSOR MATCHES "^x86(_64)?$")
        SET(OPTIONAL_ARGS DYNAMIC_ARCH=1 NUM_THREADS=64)
    ENDIF()

    SET(COMMON_ARGS CC=${OPENBLAS_CC} NO_SHARED=1 NO_LAPACK=1 libs)
    ExternalProject_Add(
        extern_openblas
        ${EXTERNAL_PROJECT_LOG_ARGS}
        ${SHALLOW_CLONE}
        GIT_REPOSITORY      https://github.com/xianyi/OpenBLAS.git
        GIT_TAG             ${OPENBLAS_COMMIT}
        PREFIX              ${CBLAS_SOURCES_DIR}
        INSTALL_DIR         ${CBLAS_INSTALL_DIR}
        BUILD_IN_SOURCE     1
        BUILD_COMMAND       ${CMAKE_MAKE_PROGRAM} ${COMMON_ARGS} ${OPTIONAL_ARGS}
        INSTALL_COMMAND     ${CMAKE_MAKE_PROGRAM} install NO_SHARED=1 NO_LAPACK=1 PREFIX=<INSTALL_DIR> 
                            && rm -r ${CBLAS_INSTALL_DIR}/lib/cmake ${CBLAS_INSTALL_DIR}/lib/pkgconfig
        UPDATE_COMMAND      ""
        CONFIGURE_COMMAND   ""
    )
ELSE(NOT WIN32)
    SET(CBLAS_LIBRARIES
        "${CBLAS_INSTALL_DIR}/lib/openblas${CMAKE_STATIC_LIBRARY_SUFFIX}"
        CACHE FILEPATH "openblas library." FORCE)
    SET(CBLAS_INC_DIR "${CBLAS_INSTALL_DIR}/include/openblas" CACHE PATH "openblas include directory." FORCE)
    ExternalProject_Add(
        extern_openblas
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY      https://github.com/xianyi/OpenBLAS.git
        GIT_TAG            "v0.3.7"
        PREFIX              ${CBLAS_SOURCES_DIR}
        INSTALL_DIR         ${CBLAS_INSTALL_DIR}
        BUILD_IN_SOURCE     0
        UPDATE_COMMAND      ""
        CMAKE_ARGS          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                            -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
                            -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                            -DCMAKE_INSTALL_PREFIX=${CBLAS_INSTALL_DIR}
                            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                            -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
                            -DMSVC_STATIC_CRT=${MSVC_STATIC_CRT}
                            ${EXTERNAL_OPTIONAL_ARGS}
        CMAKE_CACHE_ARGS    -DCMAKE_INSTALL_PREFIX:PATH=${CBLAS_INSTALL_DIR}
                            -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
                            -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
        )
ENDIF(NOT WIN32)
