if(OPENSSL_FOUND)
message(STATUS "  OpenSSL:              ${OPENSSL_FOUND}"                               )
add_subdirectory 		(aes)
endif()

