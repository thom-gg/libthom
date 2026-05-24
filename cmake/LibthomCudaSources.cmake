function(add_arch_cuda_source target selected_sources_var source_name)
    set(supported_archs ${ARGN})

    if(NOT supported_archs)
        message(FATAL_ERROR "Missing CUDA architectures for ${source_name}")
    endif()

    string(TOUPPER "${source_name}" source_macro_name)
    set(source_file "${source_name}.cu")
    set(source_compile_definition "LIB_HAS_${source_macro_name}")

    if(ARCH STREQUAL "all" OR "all" IN_LIST supported_archs OR ARCH IN_LIST supported_archs)
        target_sources(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
        target_compile_definitions(${target} PUBLIC ${source_compile_definition}=1)
        set(current_selected_sources "${${selected_sources_var}}")
        list(APPEND current_selected_sources "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
        set(${selected_sources_var} "${current_selected_sources}" PARENT_SCOPE)
    endif()
endfunction()
