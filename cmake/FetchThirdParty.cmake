# FetchThirdParty.cmake - Fetch third-party source / prebuilt to cache (component-owned for standalone build).
# Usage: include() from component root, then call fetch_thirdparty(...).
#
# Supported modes:
#   - GIT_REPO   : clone source repository (optional GIT_REF / GIT_COMMIT / GIT_RECURSIVE)
#   - ARCHIVE_URL: download tarball (.tgz / .tar.gz) and extract (no compile, prebuilt binaries)
#
# Cache root: env SROBOTIS_THIRDPARTY_CACHE > $HOME/.cache/thirdparty > ${CMAKE_BINARY_DIR}/.cache/thirdparty
# Set SROBOTIS_THIRDPARTY_FETCH_OFF=ON to disable network fetch (offline / pre-populated cache only).

if(DEFINED _FETCH_THIRDPARTY_LOADED)
  return()
endif()
set(_FETCH_THIRDPARTY_LOADED ON)

if(DEFINED ENV{SROBOTIS_THIRDPARTY_CACHE})
  set(_FETCH_TP_CACHE_ROOT "$ENV{SROBOTIS_THIRDPARTY_CACHE}")
elseif(DEFINED ENV{HOME})
  set(_FETCH_TP_CACHE_ROOT "$ENV{HOME}/.cache/thirdparty")
else()
  set(_FETCH_TP_CACHE_ROOT "${CMAKE_BINARY_DIR}/.cache/thirdparty")
endif()

if(DEFINED SROBOTIS_THIRDPARTY_FETCH_OFF)
  set(_FETCH_TP_DISABLED "${SROBOTIS_THIRDPARTY_FETCH_OFF}")
else()
  set(_FETCH_TP_DISABLED OFF)
endif()

function(fetch_thirdparty)
  cmake_parse_arguments(ARG
    "GIT_RECURSIVE"
    "NAME;GIT_REPO;GIT_REF;GIT_COMMIT;ARCHIVE_URL;ARCHIVE_SUBDIR;SUBDIR;OUT_SOURCE_DIR"
    ""
    ${ARGN})

  if(NOT ARG_NAME)
    message(FATAL_ERROR "fetch_thirdparty: NAME is required")
  endif()
  if(NOT ARG_GIT_REPO AND NOT ARG_ARCHIVE_URL)
    message(FATAL_ERROR "fetch_thirdparty: GIT_REPO or ARCHIVE_URL is required")
  endif()
  if(ARG_GIT_REPO AND ARG_ARCHIVE_URL)
    message(FATAL_ERROR "fetch_thirdparty: GIT_REPO and ARCHIVE_URL are mutually exclusive")
  endif()

  set(_dir "${_FETCH_TP_CACHE_ROOT}/${ARG_NAME}")

  if(ARG_ARCHIVE_SUBDIR)
    set(_check_path "${_dir}/${ARG_ARCHIVE_SUBDIR}")
  elseif(ARG_SUBDIR)
    set(_check_path "${_dir}/${ARG_SUBDIR}")
  else()
    set(_check_path "${_dir}")
  endif()

  set(_need_fetch OFF)
  if(NOT EXISTS "${_check_path}")
    set(_need_fetch ON)
  endif()
  if(_need_fetch AND _FETCH_TP_DISABLED)
    message(FATAL_ERROR "fetch_thirdparty: ${ARG_NAME} not found at ${_check_path} and fetch is disabled (SROBOTIS_THIRDPARTY_FETCH_OFF)")
  endif()

  if(_need_fetch)
    file(MAKE_DIRECTORY "${_FETCH_TP_CACHE_ROOT}")

    if(ARG_ARCHIVE_URL)
      # tarball mode: download + extract, no compile
      get_filename_component(_archive_name "${ARG_ARCHIVE_URL}" NAME)
      set(_archive_file "${_FETCH_TP_CACHE_ROOT}/${_archive_name}")

      message(STATUS "Fetching ${ARG_NAME}: downloading ${ARG_ARCHIVE_URL}")
      file(DOWNLOAD "${ARG_ARCHIVE_URL}" "${_archive_file}"
           STATUS _dl_status
           SHOW_PROGRESS
           TLS_VERIFY ON)
      list(GET _dl_status 0 _dl_code)
      if(NOT _dl_code EQUAL 0)
        list(GET _dl_status 1 _dl_msg)
        file(REMOVE "${_archive_file}")
        message(FATAL_ERROR "fetch_thirdparty: download failed (${_dl_code}): ${_dl_msg}\n  url: ${ARG_ARCHIVE_URL}")
      endif()

      file(MAKE_DIRECTORY "${_dir}")
      message(STATUS "Fetching ${ARG_NAME}: extracting to ${_dir}")
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${_archive_file}"
        WORKING_DIRECTORY "${_dir}"
        RESULT_VARIABLE _tar_res)
      if(NOT _tar_res EQUAL 0)
        message(FATAL_ERROR "fetch_thirdparty: extract failed (${_tar_res}): ${_archive_file}")
      endif()
      file(REMOVE "${_archive_file}")

    else()
      # git clone mode
      set(_recursive_flag "")
      if(ARG_GIT_RECURSIVE)
        set(_recursive_flag "--recursive")
      endif()

      message(STATUS "Fetching ${ARG_NAME}: git clone ${ARG_GIT_REPO}")
      if(ARG_GIT_COMMIT)
        execute_process(
          COMMAND git clone --depth 1 ${_recursive_flag} "${ARG_GIT_REPO}" "${_dir}"
          WORKING_DIRECTORY "${_FETCH_TP_CACHE_ROOT}"
          RESULT_VARIABLE _res)
        if(_res EQUAL 0)
          execute_process(COMMAND git checkout "${ARG_GIT_COMMIT}"
                          WORKING_DIRECTORY "${_dir}"
                          RESULT_VARIABLE _res)
        endif()
      elseif(ARG_GIT_REF)
        execute_process(
          COMMAND git clone --depth 1 ${_recursive_flag} "${ARG_GIT_REPO}" -b "${ARG_GIT_REF}" "${_dir}"
          WORKING_DIRECTORY "${_FETCH_TP_CACHE_ROOT}"
          RESULT_VARIABLE _res)
      else()
        execute_process(
          COMMAND git clone --depth 1 ${_recursive_flag} "${ARG_GIT_REPO}" "${_dir}"
          WORKING_DIRECTORY "${_FETCH_TP_CACHE_ROOT}"
          RESULT_VARIABLE _res)
      endif()
      if(NOT _res EQUAL 0)
        message(FATAL_ERROR "fetch_thirdparty: failed to clone ${ARG_NAME} (${_res})")
      endif()
    endif()
  endif()

  if(NOT EXISTS "${_check_path}")
    message(FATAL_ERROR "fetch_thirdparty: path does not exist after fetch: ${_check_path}")
  endif()

  if(ARG_OUT_SOURCE_DIR)
    set(${ARG_OUT_SOURCE_DIR} "${_check_path}" PARENT_SCOPE)
  endif()
endfunction()
